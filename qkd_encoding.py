from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

try:
    from .qkd_model import QKDNetwork
except ImportError:  # pragma: no cover
    from qkd_model import QKDNetwork

try:
    from qiskit_optimization.problems import QuadraticProgram
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Missing optional dependency: qiskit_optimization. Install with `pip install qiskit-optimization`."
    ) from e


@dataclass(frozen=True)
class QKDPath:
    """Candidate path for a single demand."""
    nodes: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    latency: float
    relays: int  # number of intermediate nodes (trusted relays)
    is_drop: bool = False


@dataclass(frozen=True)
class QKDRoutingEncoding:
    """Encoding for QKD routing as one-hot path selection per demand."""
    index_to_key: Dict[int, Tuple[str, Tuple[int, ...]]]
    key_to_index: Dict[Tuple[str, Tuple[int, ...]], int]
    groups: List[List[int]]
    num_vars: int

    # QKD specific
    demand_groups: Dict[int, List[int]]       # demand_id -> variable indices (one-hot)
    paths_by_demand: Dict[int, List[QKDPath]] # demand_id -> list of candidate paths
    edge_ids_in_path: Dict[Tuple[int, int], Tuple[int, ...]]  # (demand_id, path_idx) -> edge_ids


def _enumerate_simple_paths(
    adj: Dict[int, List[Tuple[int, int]]],
    src: int,
    dst: int,
    max_hops: int,
    limit: int,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Enumerate simple paths up to max_hops; return list[(nodes, edge_ids)]."""
    out: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    stack: List[Tuple[int, List[int], List[int], set]] = [(src, [src], [], {src})]
    while stack and len(out) < limit:
        node, path_nodes, path_edges, seen = stack.pop()
        if len(path_edges) > max_hops:
            continue
        if node == dst:
            out.append((tuple(path_nodes), tuple(path_edges)))
            continue
        for nb, eid in adj.get(node, []):
            if nb in seen:
                continue
            stack.append((nb, path_nodes + [nb], path_edges + [eid], seen | {nb}))
    return out


def build_candidate_paths(
    net: QKDNetwork,
    k_paths: int = 3,
    max_hops: int = 5,
    per_demand_limit: int = 2000,
    allow_drop: bool = True,
) -> Dict[int, List[QKDPath]]:
    """Build K candidate paths per demand using simple-path enumeration (small graphs).

    SLA handling (important):
    - If a demand has dem.max_latency set, we FILTER OUT any path whose latency exceeds it.
      This enforces the per-demand latency constraint without adding <= constraints to the QP
      (which would blow up the qubit count via slack variables).

    Capacity handling:
    - Link capacities are NOT enforced in the QP (to keep the QUBO small enough for simulation).
      They are enforced later via repair_qkd_routing() after decoding.
    """
    adj = net.adjacency()

    paths_by_d: Dict[int, List[QKDPath]] = {}
    for di, dem in enumerate(net.demands):
        raw = _enumerate_simple_paths(
            adj=adj,
            src=int(dem.src),
            dst=int(dem.dst),
            max_hops=int(max_hops),
            limit=int(per_demand_limit),
        )

        cand: List[QKDPath] = []
        for nodes, eids in raw:
            lat = 0.0
            for eid in eids:
                lat += float(net.edges[eid].latency)

            # Enforce per-demand max_latency by construction (no <= constraint in QP)
            if dem.max_latency is not None and float(lat) > float(dem.max_latency) + 1e-9:
                continue

            relays = max(0, len(nodes) - 2)
            cand.append(
                QKDPath(
                    nodes=nodes,
                    edge_ids=eids,
                    latency=float(lat),
                    relays=int(relays),
                    is_drop=False,
                )
            )

        # Sort by latency then fewer relays then fewer hops
        cand.sort(key=lambda p: (p.latency, p.relays, len(p.edge_ids)))

        # Keep top K (if none exist, we'll rely on drop)
        cand = cand[: max(1, int(k_paths))] if cand else []

        if allow_drop:
            cand.append(
                QKDPath(
                    nodes=(int(dem.src), int(dem.dst)),
                    edge_ids=(),
                    latency=0.0,
                    relays=0,
                    is_drop=True,
                )
            )

        paths_by_d[di] = cand

    return paths_by_d


def build_qkd_routing_qp(
    net: QKDNetwork,
    w_unmet: float,
    w_latency: float,
    w_resource: float,
    paths_by_demand: Optional[Dict[int, List[QKDPath]]] = None,
    k_paths: int = 3,
    max_hops: int = 5,
    allow_drop: bool = True,
    relay_penalty: float = 0.5,
) -> Tuple[QuadraticProgram, QKDRoutingEncoding]:
    """Build a QKD routing QuadraticProgram with 3-objective scalarization.

    Decision variables:
      x[d,p] = 1 if demand d is routed on candidate path p (one-hot per demand).

    Constraints in the QP:
      - One-hot per demand: sum_p x[d,p] == 1

    Important modeling choice (to keep simulation feasible):
      - We do NOT add <= capacity constraints or <= SLA constraints to the QP.
        * SLA is enforced by filtering candidate paths in build_candidate_paths().
        * Capacity is enforced by repair_qkd_routing() after decoding.

    Objectives (minimize):
      1) unmet demand (drop option carries unmet = demand_d)
      2) total latency (sum over selected paths)
      3) resource / risk proxy:
            total key consumption (demand * hops) + relay_penalty * (#relays)

    Scalarized objective:
      minimize w1*f1 + w2*f2 + w3*f3
    """
    if paths_by_demand is None:
        paths_by_demand = build_candidate_paths(net, k_paths=k_paths, max_hops=max_hops, allow_drop=allow_drop)

    qp = QuadraticProgram(name="qkd_routing_3obj")

    index_to_key: Dict[int, Tuple[str, Tuple[int, ...]]] = {}
    key_to_index: Dict[Tuple[str, Tuple[int, ...]], int] = {}
    groups: List[List[int]] = []
    demand_groups: Dict[int, List[int]] = {}
    edge_ids_in_path: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    idx = 0

    # Variables and one-hot groups per demand
    for d_id, paths in paths_by_demand.items():
        g: List[int] = []
        for p_id, p in enumerate(paths):
            name = f"x_{d_id}_{p_id}"
            qp.binary_var(name)
            key = ("x", (int(d_id), int(p_id)))
            index_to_key[idx] = key
            key_to_index[key] = idx
            g.append(idx)
            edge_ids_in_path[(int(d_id), int(p_id))] = tuple(int(e) for e in p.edge_ids)
            idx += 1
        groups.append(g)
        demand_groups[int(d_id)] = g

        # One-hot equality (integer coefficients)
        qp.linear_constraint(
            {f"x_{d_id}_{p_id}": 1 for p_id in range(len(paths))},
            sense="==",
            rhs=1,
            name=f"demand_onehot_{d_id}",
        )

    num_vars = idx

    # Objective (linear coefficients) — floats are fine here
    linear = [0.0] * num_vars

    for d_id, dem in enumerate(net.demands):
        d = float(dem.demand)
        paths = paths_by_demand[d_id]
        for p_id, p in enumerate(paths):
            var_idx = key_to_index[("x", (int(d_id), int(p_id)))]
            unmet = d if p.is_drop else 0.0
            latency = float(p.latency)
            resource = d * float(len(p.edge_ids)) + float(relay_penalty) * float(p.relays)

            linear[var_idx] += float(w_unmet) * unmet
            linear[var_idx] += float(w_latency) * latency
            linear[var_idx] += float(w_resource) * resource

    qp.minimize(linear=linear)

    enc = QKDRoutingEncoding(
        index_to_key=index_to_key,
        key_to_index=key_to_index,
        groups=groups,
        num_vars=num_vars,
        demand_groups=demand_groups,
        paths_by_demand=paths_by_demand,
        edge_ids_in_path=edge_ids_in_path,
    )
    return qp, enc


def decode_qkd_choice_from_bitstring(bitstring_le: str, enc: QKDRoutingEncoding) -> Dict[int, int]:
    """Decode little-endian bitstring into chosen path index per demand."""
    bits = [1 if c == "1" else 0 for c in bitstring_le]
    chosen: Dict[int, int] = {}
    for d_id, g in enc.demand_groups.items():
        pick = None
        for qi in g:
            if qi < len(bits) and bits[qi] == 1:
                pick = qi
                break
        if pick is None:
            pick = g[0]
        _, (_d, p_id) = enc.index_to_key[pick]
        chosen[int(d_id)] = int(p_id)
    return chosen


def repair_qkd_routing(
    chosen: Dict[int, int],
    net: QKDNetwork,
    enc: QKDRoutingEncoding,
) -> Dict[int, int]:
    """Greedy repair to satisfy link capacities by rerouting or dropping demands.

    NOTE:
    - Because we do not include capacity <= constraints in the QP->QUBO (to keep qubit count small),
      decoded bitstrings may violate capacities. This repair enforces feasibility before evaluation.
    """
    # Precompute alternative paths (excluding the current choice) per demand, sorted by latency then relays.
    alts: Dict[int, List[int]] = {}
    for d_id, paths in enc.paths_by_demand.items():
        order = list(range(len(paths)))
        order.sort(key=lambda pid: (paths[pid].is_drop, paths[pid].latency, paths[pid].relays, len(paths[pid].edge_ids)))
        alts[d_id] = order

    def compute_load(ch: Dict[int, int]) -> List[float]:
        loads = [0.0 for _ in net.edges]
        for d_id, pid in ch.items():
            dem = float(net.demands[d_id].demand)
            p = enc.paths_by_demand[d_id][pid]
            for eid in p.edge_ids:
                loads[int(eid)] += dem
        return loads

    def feasible(loads: List[float]) -> bool:
        for eid, ld in enumerate(loads):
            if ld > float(net.edges[eid].key_capacity) + 1e-9:
                return False
        return True

    # Iteratively fix the most violated edge by adjusting one demand that uses it.
    ch = {int(k): int(v) for k, v in chosen.items()}
    for _ in range(200):  # small safety cap
        loads = compute_load(ch)
        if feasible(loads):
            return ch

        # Find most violated edge
        viol = []
        for eid, ld in enumerate(loads):
            cap = float(net.edges[eid].key_capacity)
            if ld > cap + 1e-9:
                viol.append((ld - cap, eid))
        if not viol:
            return ch
        viol.sort(reverse=True)
        _excess, bad_eid = viol[0]

        # Find demands that use this edge; prefer rerouting the largest demand first
        users = []
        for d_id, pid in ch.items():
            p = enc.paths_by_demand[d_id][pid]
            if bad_eid in p.edge_ids:
                users.append((float(net.demands[d_id].demand), d_id))
        if not users:
            return ch
        users.sort(reverse=True)

        fixed = False
        for _dem_sz, d_id in users:
            cur = ch[d_id]
            # try alternate paths that avoid bad edge
            for pid in alts[d_id]:
                if pid == cur:
                    continue
                p = enc.paths_by_demand[d_id][pid]
                if bad_eid in p.edge_ids:
                    continue
                trial = dict(ch)
                trial[d_id] = pid
                if feasible(compute_load(trial)):
                    ch = trial
                    fixed = True
                    break
            if fixed:
                break

        if not fixed:
            # As last resort, force-drop the largest demand that uses the violated edge
            drop_target = None
            drop_pid = None
            for _dem_sz, cand_id in users:
                for pid, p in enumerate(enc.paths_by_demand[cand_id]):
                    if p.is_drop:
                        drop_target = int(cand_id)
                        drop_pid = int(pid)
                        break
                if drop_target is not None:
                    break

            if drop_target is not None and drop_pid is not None and drop_pid != ch[drop_target]:
                ch[drop_target] = int(drop_pid)
            else:
                return ch

    return ch


def evaluate_qkd_objectives(
    chosen: Dict[int, int],
    net: QKDNetwork,
    enc: QKDRoutingEncoding,
    relay_penalty: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute 3 objectives (minimize):
      1) total unmet demand
      2) total latency
      3) resource/risk proxy: key-consumption + relay_penalty * relays
    """
    unmet = 0.0
    latency = 0.0
    resource = 0.0

    for d_id, pid in chosen.items():
        dem = float(net.demands[d_id].demand)
        p = enc.paths_by_demand[d_id][pid]
        if p.is_drop:
            unmet += dem
            continue
        latency += float(p.latency)
        resource += dem * float(len(p.edge_ids)) + float(relay_penalty) * float(p.relays)

    return float(unmet), float(latency), float(resource)


def compute_edge_loads(
    chosen: Dict[int, int],
    net: QKDNetwork,
    paths_by_demand: Dict[int, List[QKDPath]],
) -> List[float]:
    """Compute per-edge key consumption and return loads[eid]."""
    loads = [0.0 for _ in net.edges]
    for d_id, pid in chosen.items():
        dem = float(net.demands[int(d_id)].demand)
        p = paths_by_demand[int(d_id)][int(pid)]
        if p.is_drop:
            continue
        for eid in p.edge_ids:
            loads[int(eid)] += dem
    return loads