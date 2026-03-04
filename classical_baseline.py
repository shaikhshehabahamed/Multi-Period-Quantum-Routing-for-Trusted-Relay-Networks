#!/usr/bin/env python3
"""
classical_myopic_baseline.py

A fully-classical (non-quantum) baseline for the QKD trusted-relay routing problem:
- Builds candidate paths per demand (simple-path enumeration, SLA filtering).
- For each period, solves multiple scalarizations using a *myopic* greedy rule.
- Enforces link-capacity feasibility via a greedy repair step (reroute / drop).
- Evaluates the 3 objectives used in the quantum codebase:
    (1) total unmet demand
    (2) total latency
    (3) resource/risk proxy: demand * hops + relay_penalty * relays
- Saves a 3D Pareto scatter figure (all points + non-dominated points) per period.

No Qiskit, no qiskit-optimization. Pure Python + numpy + matplotlib.

Run a demo:
    python classical_myopic_baseline.py --demo

Typical usage in your experiments:
    - Build `base_net` (with key-pool dynamics fields set on edges if desired)
    - Build `demands_by_t` as a list of per-period demand lists
    - Call `run_multiperiod_myopic(...)`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import argparse
import csv
import heapq
import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data model (minimal, self-contained)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    key_capacity: float
    latency: float
    trusted: bool = True

    # Multi-period key-pool dynamics
    key_pool_init: Optional[float] = None
    key_gen_rate: float = 0.0
    key_storage_cap: Optional[float] = None
    key_decay: float = 0.0


@dataclass(frozen=True)
class Demand:
    src: int
    dst: int
    demand: float
    max_latency: Optional[float] = None


@dataclass
class Network:
    n_nodes: int
    edges: List[Edge]
    demands: List[Demand]

    def adjacency(self) -> Dict[int, List[Tuple[int, int]]]:
        """node -> list[(neighbor, edge_id)]"""
        adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.n_nodes)}
        for eid, e in enumerate(self.edges):
            adj[e.u].append((e.v, eid))
            adj[e.v].append((e.u, eid))
        return adj


# ---------------------------------------------------------------------------
# Candidate paths (pure Python)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Path:
    nodes: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    latency: float
    relays: int
    is_drop: bool = False


def _enumerate_simple_paths(
    adj: Dict[int, List[Tuple[int, int]]],
    src: int,
    dst: int,
    max_hops: int,
    limit: int,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Enumerate simple paths up to max_hops (small graphs)."""
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
    net: Network,
    k_paths: int = 3,
    max_hops: int = 6,
    per_demand_limit: int = 5000,
    allow_drop: bool = True,
) -> Dict[int, List[Path]]:
    """Build up to k_paths candidate paths per demand (plus optional drop)."""
    adj = net.adjacency()
    paths_by_d: Dict[int, List[Path]] = {}

    for d_id, dem in enumerate(net.demands):
        raw = _enumerate_simple_paths(
            adj=adj,
            src=int(dem.src),
            dst=int(dem.dst),
            max_hops=int(max_hops),
            limit=int(per_demand_limit),
        )

        cand: List[Path] = []
        for nodes, eids in raw:
            lat = sum(float(net.edges[eid].latency) for eid in eids)

            if dem.max_latency is not None and lat > float(dem.max_latency) + 1e-9:
                continue

            relays = max(0, len(nodes) - 2)
            cand.append(
                Path(
                    nodes=nodes,
                    edge_ids=eids,
                    latency=float(lat),
                    relays=int(relays),
                    is_drop=False,
                )
            )

        # Sort by latency, then fewer relays, then fewer hops
        cand.sort(key=lambda p: (p.latency, p.relays, len(p.edge_ids)))

        # Keep top K
        cand = cand[: max(1, int(k_paths))] if cand else []

        if allow_drop:
            cand.append(
                Path(nodes=(int(dem.src), int(dem.dst)), edge_ids=(), latency=0.0, relays=0, is_drop=True)
            )

        paths_by_d[d_id] = cand

    return paths_by_d


# ---------------------------------------------------------------------------
# Myopic solver + repair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightConfig3:
    w_unmet: float
    w_latency: float
    w_resource: float


def weight_grid_simplex(n: int = 7, cap: int = 12) -> List[WeightConfig3]:
    """
    Generate weight vectors on the 2-simplex (3 weights that sum to 1) using an
    unbiased integer lattice, then (optionally) downsample to `cap` *diverse*
    weights using farthest-point sampling.

    Why this exists:
      - Your original implementation built weights in nested-loop order and then
        returned ws[:cap]. That truncation is order-biased and can systematically
        under-sample high w_unmet settings, causing the greedy solver to choose
        "drop everything" for many/all weight vectors.
      - This implementation covers the simplex evenly and, if capped, keeps a
        spread-out subset.
    """
    n = int(n)
    cap = int(cap)
    if n < 2:
        raise ValueError("n must be >= 2")
    if cap < 1:
        raise ValueError("cap must be >= 1")

    # Lattice resolution: number of segments along each axis.
    # With res = n-1, number of simplex lattice points is n*(n+1)/2.
    res = n - 1

    # 1) Full deterministic lattice: (i, j, k)/res with i+j+k=res
    ws: List[WeightConfig3] = []
    for i in range(res + 1):
        for j in range(res - i + 1):
            k = res - i - j
            a = i / res
            b = j / res
            c = k / res
            ws.append(WeightConfig3(float(a), float(b), float(c)))

    if not ws:
        return [WeightConfig3(1.0 / 3, 1.0 / 3, 1.0 / 3)]

    if cap >= len(ws):
        return ws

    # 2) Diverse downsampling via farthest-point sampling
    arr = np.array([(w.w_unmet, w.w_latency, w.w_resource) for w in ws], dtype=float)

    # Seed with simplex anchors (corners + centroid), mapped to nearest lattice points.
    anchors = np.array(
        [
            [1.0, 0.0, 0.0],            # unmet-focused
            [0.0, 1.0, 0.0],            # latency-focused
            [0.0, 0.0, 1.0],            # resource-focused
            [1.0 / 3, 1.0 / 3, 1.0 / 3] # balanced
        ],
        dtype=float,
    )

    selected: List[int] = []
    for a in anchors:
        idx = int(np.argmin(np.linalg.norm(arr - a, axis=1)))
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= cap:
            break

    if not selected:
        selected = [0]

    remaining = set(range(len(ws))) - set(selected)

    while len(selected) < cap and remaining:
        rem = np.array(list(remaining), dtype=int)
        sel = arr[np.array(selected, dtype=int)]

        # Distance from each remaining point to its nearest selected point
        dists = np.linalg.norm(arr[rem][:, None, :] - sel[None, :, :], axis=2)
        min_d = dists.min(axis=1)

        # Add the farthest point from the selected set
        next_idx = int(rem[int(np.argmax(min_d))])
        selected.append(next_idx)
        remaining.remove(next_idx)

    selected.sort()
    return [ws[i] for i in selected]


def evaluate_objectives(
    chosen: Dict[int, int],
    net: Network,
    paths_by_demand: Dict[int, List[Path]],
    relay_penalty: float = 0.5,
) -> Tuple[float, float, float]:
    unmet = 0.0
    latency = 0.0
    resource = 0.0

    for d_id, p_id in chosen.items():
        dem = float(net.demands[d_id].demand)
        p = paths_by_demand[d_id][p_id]
        if p.is_drop:
            unmet += dem
            continue
        latency += float(p.latency)
        resource += dem * float(len(p.edge_ids)) + float(relay_penalty) * float(p.relays)

    return float(unmet), float(latency), float(resource)


def compute_edge_loads(
    chosen: Dict[int, int],
    net: Network,
    paths_by_demand: Dict[int, List[Path]],
) -> List[float]:
    loads = [0.0 for _ in net.edges]
    for d_id, p_id in chosen.items():
        dem = float(net.demands[d_id].demand)
        p = paths_by_demand[d_id][p_id]
        if p.is_drop:
            continue
        for eid in p.edge_ids:
            loads[int(eid)] += dem
    return loads


def repair_capacity_greedy(
    chosen: Dict[int, int],
    net: Network,
    paths_by_demand: Dict[int, List[Path]],
    *,
    max_iters: int = 250,
) -> Dict[int, int]:
    """Greedy repair: reroute or drop to satisfy capacities.

    Strategy:
      - Find the most violated edge.
      - Identify demands that use it (largest demand first).
      - Try alternate paths that avoid that edge; if none works, drop the largest demand.

    This mirrors the repair idea used in the quantum codebase, but is standalone.
    """
    # Pre-sort alternative path indices per demand
    alts: Dict[int, List[int]] = {}
    for d_id, paths in paths_by_demand.items():
        order = list(range(len(paths)))
        order.sort(key=lambda pid: (paths[pid].is_drop, paths[pid].latency, paths[pid].relays, len(paths[pid].edge_ids)))
        alts[d_id] = order

    def feasible(loads: List[float]) -> bool:
        for eid, ld in enumerate(loads):
            if ld > float(net.edges[eid].key_capacity) + 1e-9:
                return False
        return True

    ch = {int(k): int(v) for k, v in chosen.items()}

    for _ in range(int(max_iters)):
        loads = compute_edge_loads(ch, net, paths_by_demand)
        if feasible(loads):
            return ch

        # Most violated edge
        viol = []
        for eid, ld in enumerate(loads):
            cap = float(net.edges[eid].key_capacity)
            if ld > cap + 1e-9:
                viol.append((ld - cap, eid))
        if not viol:
            return ch
        viol.sort(reverse=True)
        _excess, bad_eid = viol[0]

        # Demands that use it, largest demand first
        users: List[Tuple[float, int]] = []
        for d_id, p_id in ch.items():
            p = paths_by_demand[d_id][p_id]
            if bad_eid in p.edge_ids:
                users.append((float(net.demands[d_id].demand), int(d_id)))
        if not users:
            return ch
        users.sort(reverse=True)

        fixed = False
        for _dem_sz, d_id in users:
            cur = ch[d_id]
            for p_id in alts[d_id]:
                if p_id == cur:
                    continue
                p = paths_by_demand[d_id][p_id]
                if bad_eid in p.edge_ids:
                    continue
                trial = dict(ch)
                trial[d_id] = int(p_id)
                if feasible(compute_edge_loads(trial, net, paths_by_demand)):
                    ch = trial
                    fixed = True
                    break
            if fixed:
                break

        if fixed:
            continue

        # Drop largest user as last resort
        drop_pid = None
        for pid, p in enumerate(paths_by_demand[users[0][1]]):
            if p.is_drop:
                drop_pid = int(pid)
                break
        if drop_pid is None:
            return ch
        ch[users[0][1]] = drop_pid

    return ch


def solve_myopic_for_weight(
    net: Network,
    paths_by_demand: Dict[int, List[Path]],
    w: WeightConfig3,
    *,
    relay_penalty: float = 0.5,
) -> Tuple[Dict[int, int], Tuple[float, float, float]]:
    """Myopic greedy selection per demand for a given weight vector, then repair."""
    chosen: Dict[int, int] = {}

    for d_id, dem in enumerate(net.demands):
        d = float(dem.demand)
        best_pid = 0
        best_score = float("inf")

        for pid, p in enumerate(paths_by_demand[d_id]):
            unmet = d if p.is_drop else 0.0
            lat = float(p.latency)
            res = d * float(len(p.edge_ids)) + float(relay_penalty) * float(p.relays)
            score = float(w.w_unmet) * unmet + float(w.w_latency) * lat + float(w.w_resource) * res
            if score < best_score:
                best_score = score
                best_pid = int(pid)

        chosen[int(d_id)] = int(best_pid)

    chosen = repair_capacity_greedy(chosen, net, paths_by_demand)
    obj = evaluate_objectives(chosen, net, paths_by_demand, relay_penalty=relay_penalty)
    return chosen, obj


# ---------------------------------------------------------------------------
# Pareto utilities + 3D plotting (saves images)
# ---------------------------------------------------------------------------

def pareto_filter(points: Sequence[Sequence[float]]) -> List[Tuple[float, ...]]:
    """Non-dominated subset (minimization)."""
    pts = [tuple(map(float, p)) for p in points]
    keep: List[Tuple[float, ...]] = []
    for i, a in enumerate(pts):
        dominated = False
        for j, b in enumerate(pts):
            if i == j:
                continue
            if all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a)):
                dominated = True
                break
        if not dominated:
            keep.append(a)

    # unique
    uniq: List[Tuple[float, ...]] = []
    seen = set()
    for p in keep:
        key = tuple(round(x, 12) for x in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def save_pareto_3d(
    points: Sequence[Sequence[float]],
    out_path: str,
    labels: Tuple[str, str, str] = ("Unmet demand", "Total latency", "Resource/Risk"),
    title: str = "Pareto set (3D, minimization)",
    dpi: int = 600,
) -> str:
    arr = np.array(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points; got shape={arr.shape}")
    nd = np.array(pareto_filter(arr), dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], marker="o", alpha=0.35, label="All points")
    if len(nd) > 0:
        ax.scatter(nd[:, 0], nd[:, 1], nd[:, 2], marker="x", label="Non-dominated")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])  # removed negative labelpad that can overlap ticks
    ax.set_title(title)
    ax.legend()

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.92)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Multi-period key-pool dynamics (standalone)
# ---------------------------------------------------------------------------

def init_key_pools(net: Network) -> List[float]:
    return [float(e.key_pool_init) if e.key_pool_init is not None else float(e.key_capacity) for e in net.edges]


def available_capacities(net: Network, pools: List[float]) -> List[float]:
    """Capacity snapshot for current step (after decay+gen+cap, before consumption)."""
    caps: List[float] = []
    for i, e in enumerate(net.edges):
        pool = float(pools[i])
        pool = pool * (1.0 - float(e.key_decay)) + float(e.key_gen_rate)
        if e.key_storage_cap is not None:
            pool = min(pool, float(e.key_storage_cap))
        caps.append(pool)
    return caps


def apply_consumption(net: Network, pools: List[float], edge_loads: List[float]) -> List[float]:
    caps = available_capacities(net, pools)
    return [max(0.0, float(caps[i]) - float(edge_loads[i])) for i in range(len(caps))]


def make_period_network(base_net: Network, demands_t: List[Demand], capacities: List[float]) -> Network:
    if len(capacities) != len(base_net.edges):
        raise ValueError("capacities length must match number of edges")
    edges: List[Edge] = []
    for i, e in enumerate(base_net.edges):
        edges.append(
            Edge(
                u=int(e.u),
                v=int(e.v),
                key_capacity=float(capacities[i]),
                latency=float(e.latency),
                trusted=bool(e.trusted),
                key_pool_init=e.key_pool_init,
                key_gen_rate=float(e.key_gen_rate),
                key_storage_cap=e.key_storage_cap,
                key_decay=float(e.key_decay),
            )
        )
    return Network(n_nodes=int(base_net.n_nodes), edges=edges, demands=list(demands_t))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class PeriodResult:
    t: int
    pools_before: List[float]
    capacities: List[float]

    # Pareto approximation from the myopic heuristic across weights
    pareto_points: List[Tuple[float, float, float]] = field(default_factory=list)
    pareto_fig_path: Optional[str] = None

    # Executed operating solution
    chosen: Dict[int, int] = field(default_factory=dict)
    objectives: Tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))
    edge_loads: List[float] = field(default_factory=list)
    pools_after: List[float] = field(default_factory=list)


def pick_operating_solution(
    solutions: List[Tuple[Dict[int, int], Tuple[float, float, float], WeightConfig3]],
    policy: Optional[WeightConfig3] = None,
) -> Tuple[Dict[int, int], Tuple[float, float, float]]:
    if not solutions:
        return {}, (float("nan"), float("nan"), float("nan"))

    if policy is None:
        # Lexicographic by (unmet, latency, resource)
        solutions.sort(key=lambda s: (s[1][0], s[1][1], s[1][2]))
        return dict(solutions[0][0]), tuple(map(float, solutions[0][1]))

    w1, w2, w3 = float(policy.w_unmet), float(policy.w_latency), float(policy.w_resource)
    solutions.sort(
        key=lambda s: (
            w1 * s[1][0] + w2 * s[1][1] + w3 * s[1][2],
            s[1][0], s[1][1], s[1][2],
        )
    )
    return dict(solutions[0][0]), tuple(map(float, solutions[0][1]))


def run_multiperiod_myopic(
    base_net: Network,
    demands_by_t: List[List[Demand]],
    *,
    policy: Optional[WeightConfig3] = None,
    weights_grid_n: int = 7,
    num_weights_cap: int = 12,
    k_paths: int = 3,
    max_hops: int = 6,
    allow_drop: bool = True,
    relay_penalty: float = 0.6,
    save_pareto: bool = True,
    pareto_out_dir: str = "pareto_figs_classical",
    pareto_prefix: str = "pareto_t",
    pareto_dpi: int = 600,
    pareto_fmt: str = "png",
    pareto_labels: Tuple[str, str, str] = ("Unmet demand", "Total latency", "Resource/Risk"),
) -> List[PeriodResult]:
    pools = init_key_pools(base_net)
    hist: List[PeriodResult] = []

    weights = weight_grid_simplex(n=int(weights_grid_n), cap=max(1, int(num_weights_cap)))

    for t, demands_t in enumerate(demands_by_t):
        pools_before = list(pools)
        caps = available_capacities(base_net, pools_before)

        net_t = make_period_network(base_net, demands_t, caps)

        # Build candidates once per period
        paths_by_d = build_candidate_paths(
            net_t,
            k_paths=int(k_paths),
            max_hops=int(max_hops),
            allow_drop=bool(allow_drop),
        )

        solutions: List[Tuple[Dict[int, int], Tuple[float, float, float], WeightConfig3]] = []
        points: List[Tuple[float, float, float]] = []

        for w in weights:
            ch, obj = solve_myopic_for_weight(net_t, paths_by_d, w, relay_penalty=float(relay_penalty))
            solutions.append((ch, obj, w))
            points.append(tuple(map(float, obj)))

        chosen, obj = pick_operating_solution(solutions, policy=policy)

        loads = compute_edge_loads(chosen, net_t, paths_by_d)
        pools_after = apply_consumption(base_net, pools_before, loads)

        fig_path = None
        if save_pareto and points:
            os.makedirs(str(pareto_out_dir), exist_ok=True)
            fig_path = os.path.join(str(pareto_out_dir), f"{pareto_prefix}{t:03d}_3d.{pareto_fmt.lstrip('.')}")
            save_pareto_3d(
                points,
                out_path=fig_path,
                labels=pareto_labels,
                title="Pareto set (3D, minimization)",
                dpi=int(pareto_dpi),
            )

        hist.append(
            PeriodResult(
                t=int(t),
                pools_before=pools_before,
                capacities=list(caps),
                pareto_points=list(points),
                pareto_fig_path=fig_path,
                chosen=dict(chosen),
                objectives=tuple(map(float, obj)),
                edge_loads=list(loads),
                pools_after=list(pools_after),
            )
        )

        pools = pools_after

    return hist


# ---------------------------------------------------------------------------
# Demo instance (mirrors the scale/shape of your tiny demo)
# ---------------------------------------------------------------------------

def tiny_demo_network_with_pools(seed: int = 7) -> Network:
    rng = random.Random(int(seed))
    n = 6

    edges = [
        Edge(0, 1, key_capacity=120.0, latency=2.0),
        Edge(1, 2, key_capacity=80.0, latency=2.0),
        Edge(2, 3, key_capacity=70.0, latency=2.5),
        Edge(0, 4, key_capacity=90.0, latency=2.2),
        Edge(4, 2, key_capacity=60.0, latency=2.0),
        Edge(4, 5, key_capacity=50.0, latency=1.6),
        Edge(5, 3, key_capacity=55.0, latency=1.9),
        Edge(1, 4, key_capacity=65.0, latency=2.1),
    ]

    demands = [
        Demand(src=0, dst=3, demand=55.0, max_latency=8.0),
        Demand(src=0, dst=2, demand=40.0, max_latency=7.5),
        Demand(src=4, dst=2, demand=35.0, max_latency=7.0),
        Demand(src=1, dst=3, demand=45.0, max_latency=8.0),
    ]

    # Perturb to avoid ties (deterministic with seed)
    pert_edges: List[Edge] = []
    for e in edges:
        cap = max(0.0, e.key_capacity * (0.9 + 0.2 * rng.random()))
        lat = e.latency * (0.95 + 0.1 * rng.random())
        pert_edges.append(
            Edge(
                e.u,
                e.v,
                key_capacity=float(cap),
                latency=float(lat),
                trusted=e.trusted,
                key_pool_init=float(cap),                 # start full
                key_gen_rate=0.08 * float(cap),           # regenerate each step
                key_storage_cap=1.20 * float(cap),        # limited storage
                key_decay=0.02,                           # small decay
            )
        )

    return Network(n_nodes=n, edges=pert_edges, demands=demands)


def demo_demands_by_t(base: List[Demand], T: int = 5, seed: int = 7) -> List[List[Demand]]:
    rng = random.Random(int(seed))
    out: List[List[Demand]] = []
    for _t in range(int(T)):
        period: List[Demand] = []
        for d in base:
            mult = 0.9 + 0.2 * rng.random()
            period.append(Demand(src=d.src, dst=d.dst, demand=float(d.demand) * float(mult), max_latency=d.max_latency))
        out.append(period)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run a small demo and save figures")
    ap.add_argument("--T", type=int, default=5, help="number of periods for demo")
    ap.add_argument("--outdir", type=str, default="pareto_figs_classical", help="output dir for pareto images")
    args = ap.parse_args()

    if args.demo:
        base_net = tiny_demo_network_with_pools(seed=7)
        demands_by_t = demo_demands_by_t(base_net.demands, T=int(args.T), seed=7)

        hist = run_multiperiod_myopic(
            base_net,
            demands_by_t,
            policy=None,  # lexicographic operating point
            weights_grid_n=7,
            num_weights_cap=12,
            k_paths=3,
            max_hops=6,
            allow_drop=True,
            relay_penalty=0.6,
            save_pareto=True,
            pareto_out_dir=str(args.outdir),
            pareto_prefix="pareto_t",
            pareto_dpi=600,
            pareto_fmt="png",
        )

        print("\n=== Classical myopic baseline results ===")
        for r in hist:
            print(
                f"t={r.t}  obj(unmet,lat,res)={tuple(round(x, 3) for x in r.objectives)}  "
                f"min_pool_after={min(r.pools_after):.3f}"
            )
            if r.pareto_fig_path:
                print(f"    Pareto figure: {r.pareto_fig_path}")
        return

    ap.error("Nothing to do: pass --demo (or integrate run_multiperiod_myopic in your experiments).")


if __name__ == "__main__":
    main()