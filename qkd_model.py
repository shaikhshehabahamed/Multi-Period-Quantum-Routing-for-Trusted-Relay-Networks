from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import random


@dataclass(frozen=True)
class QKDEdge:
    """Undirected QKD link.

    This codebase uses a *per-time-step* capacity snapshot (``key_capacity``) when building the routing QP.
    For multi-period simulations, you can additionally provide key-pool dynamics parameters; a runner can
    then compute the per-step capacity snapshot from the current pool state.

    Attributes
    ----------
    u, v : int
        Endpoints (node indices).
    key_capacity : float
        Available key units for the current time window (e.g., kbits). In a multi-period simulation, this
        is typically overwritten each step using :func:`available_key_capacities`.
    latency : float
        Link latency/cost (arbitrary units).
    trusted : bool
        Optional tag indicating trusted-relay infrastructure.

    Multi-period parameters (optional; backward compatible)
    ------------------------------------------------------
    key_pool_init : Optional[float]
        Initial pool amount. If None, defaults to ``key_capacity``.
    key_gen_rate : float
        Keys generated per step and added to the pool.
    key_storage_cap : Optional[float]
        Max storable keys; if None, unbounded.
    key_decay : float
        Fraction of the pool lost each step, in [0, 1).
    """
    u: int
    v: int
    key_capacity: float
    latency: float
    trusted: bool = True

    # --- Multi-period key-pool parameters (optional; backward compatible) ---
    key_pool_init: Optional[float] = None
    key_gen_rate: float = 0.0
    key_storage_cap: Optional[float] = None
    key_decay: float = 0.0


@dataclass(frozen=True)
class QKDDemand:
    """A key-demand between a source and destination in a time window."""
    src: int
    dst: int
    demand: float  # key units required (e.g., kbits)
    max_latency: Optional[float] = None  # optional SLA constraint


@dataclass
class QKDNetwork:
    """Trusted-relay QKD network instance.

    The routing model itself is single-window. Multi-period simulations can be built by repeatedly
    creating per-step snapshots where each edge's ``key_capacity`` is set to the available keys for that step.
    """
    n_nodes: int
    edges: List[QKDEdge]
    demands: List[QKDDemand]

    def __post_init__(self) -> None:
        if self.n_nodes <= 1:
            raise ValueError("n_nodes must be >= 2")
        for e in self.edges:
            if not (0 <= e.u < self.n_nodes and 0 <= e.v < self.n_nodes):
                raise ValueError("edge endpoint out of range")
            if e.u == e.v:
                raise ValueError("self-loops not supported")
            if e.key_capacity < 0:
                raise ValueError("key_capacity must be >= 0")
            if e.latency <= 0:
                raise ValueError("latency must be > 0")

            # multi-period validation (optional fields)
            if e.key_gen_rate < 0:
                raise ValueError("key_gen_rate must be >= 0")
            if e.key_storage_cap is not None and e.key_storage_cap < 0:
                raise ValueError("key_storage_cap must be >= 0")
            if not (0.0 <= float(e.key_decay) < 1.0):
                raise ValueError("key_decay must be in [0, 1)")
            if e.key_pool_init is not None and e.key_pool_init < 0:
                raise ValueError("key_pool_init must be >= 0")

        for d in self.demands:
            if not (0 <= d.src < self.n_nodes and 0 <= d.dst < self.n_nodes):
                raise ValueError("demand endpoint out of range")
            if d.src == d.dst:
                raise ValueError("src != dst required")
            if d.demand < 0:
                raise ValueError("demand must be >= 0")

    def adjacency(self) -> Dict[int, List[Tuple[int, int]]]:
        """Return adjacency: node -> list[(neighbor, edge_id)]."""
        adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.n_nodes)}
        for eid, e in enumerate(self.edges):
            adj[e.u].append((e.v, eid))
            adj[e.v].append((e.u, eid))
        return adj


def tiny_qkd_demo_network(seed: int = 7) -> QKDNetwork:
    """A tiny QKD-routing demo instance (6 nodes, 8 edges, 4 demands).

    This is intentionally small so statevector simulation is feasible.
    """
    rng = random.Random(seed)
    n = 6

    # Hand-crafted backbone-ish topology
    # 0-1-2-3 chain plus shortcuts and an alternate branch
    edges = [
        QKDEdge(0, 1, key_capacity=120.0, latency=2.0),
        QKDEdge(1, 2, key_capacity=80.0, latency=2.0),
        QKDEdge(2, 3, key_capacity=70.0, latency=2.5),
        QKDEdge(0, 4, key_capacity=90.0, latency=2.2),
        QKDEdge(4, 2, key_capacity=60.0, latency=2.0),
        QKDEdge(4, 5, key_capacity=50.0, latency=1.6),
        QKDEdge(5, 3, key_capacity=55.0, latency=1.9),
        QKDEdge(1, 4, key_capacity=65.0, latency=2.1),
    ]

    demands = [
        QKDDemand(src=0, dst=3, demand=55.0, max_latency=8.0),
        QKDDemand(src=0, dst=2, demand=40.0, max_latency=7.5),
        QKDDemand(src=4, dst=2, demand=35.0, max_latency=7.0),
        QKDDemand(src=1, dst=3, demand=45.0, max_latency=8.0),
    ]

    # Randomly perturb capacities/latencies a bit to avoid ties (still deterministic by seed)
    pert_edges: List[QKDEdge] = []
    for e in edges:
        cap = max(0.0, e.key_capacity * (0.9 + 0.2 * rng.random()))
        lat = e.latency * (0.95 + 0.1 * rng.random())
        pert_edges.append(
            QKDEdge(
                e.u,
                e.v,
                key_capacity=float(cap),
                latency=float(lat),
                trusted=e.trusted,
                key_pool_init=e.key_pool_init,
                key_gen_rate=float(e.key_gen_rate),
                key_storage_cap=e.key_storage_cap,
                key_decay=float(e.key_decay),
            )
        )

    return QKDNetwork(n_nodes=n, edges=pert_edges, demands=demands)


# ---------------------------------------------------------------------------
# Multi-period key-pool utilities
# ---------------------------------------------------------------------------

def init_key_pools(net: QKDNetwork) -> List[float]:
    """Initialize per-edge pools for multi-period runs."""
    pools: List[float] = []
    for e in net.edges:
        pools.append(float(e.key_pool_init) if e.key_pool_init is not None else float(e.key_capacity))
    return pools


def available_key_capacities(net: QKDNetwork, pools: List[float]) -> List[float]:
    """Capacity snapshot for the current step (after decay+gen+cap, before consumption)."""
    if len(pools) != len(net.edges):
        raise ValueError("pools length must match number of edges")
    caps: List[float] = []
    for i, e in enumerate(net.edges):
        pool = float(pools[i])
        pool = pool * (1.0 - float(e.key_decay)) + float(e.key_gen_rate)
        if e.key_storage_cap is not None:
            pool = min(pool, float(e.key_storage_cap))
        caps.append(pool)
    return caps


def apply_key_consumption(net: QKDNetwork, pools: List[float], edge_loads: List[float]) -> List[float]:
    """Advance pools one step using the same dynamics, then subtract consumption."""
    if len(edge_loads) != len(net.edges):
        raise ValueError("edge_loads length must match number of edges")
    caps = available_key_capacities(net, pools)
    return [max(0.0, float(caps[i]) - float(edge_loads[i])) for i in range(len(caps))]