from __future__ import annotations

"""Multi-period EP-QMOO-MPC for QKD routing.

This module is the **main algorithm** entrypoint for the project.

Core idea:
- Maintain per-edge key pools over time (generation/decay/storage).
- At each time step t:
    1) compute the available per-edge capacities from pools
    2) build a per-period snapshot network with those capacities and current demands
    3) solve EP-QMOO (Quantum Pareto QAOA) for that snapshot
    4) execute ONE chosen routing
    5) update pools by subtracting the realized edge loads

The single-window EP-QMOO solver remains as an internal subroutine (qkd_solver.EPQMOO_QKDRouting).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import random

try:
    # Package-style imports
    from .qkd_model import (
        QKDNetwork,
        QKDEdge,
        QKDDemand,
        tiny_qkd_demo_network,
        init_key_pools,
        available_key_capacities,
        apply_key_consumption,
    )
    from .qkd_solver import EPQMOO_QKDRouting, RunConfigQKD, WeightConfig3
    from .qkd_encoding import compute_edge_loads, evaluate_qkd_objectives
except ImportError:  # pragma: no cover
    # Script-style (same-folder) imports
    from qkd_model import (
        QKDNetwork,
        QKDEdge,
        QKDDemand,
        tiny_qkd_demo_network,
        init_key_pools,
        available_key_capacities,
        apply_key_consumption,
    )
    from qkd_solver import EPQMOO_QKDRouting, RunConfigQKD, WeightConfig3
    from qkd_encoding import compute_edge_loads, evaluate_qkd_objectives


@dataclass
class PeriodResult:
    t: int
    pools_before: List[float]
    capacities: List[float]
    chosen: Dict[int, int]  # demand_id -> path_idx
    objectives: Tuple[float, float, float]
    edge_loads: List[float]
    pools_after: List[float]

    # --- New: Pareto outputs (for paper-ready figures) ---
    pareto_points: List[Tuple[float, ...]] = field(default_factory=list)
    pareto_fig_paths: List[str] = field(default_factory=list)


def simple_weight_grid(n: int = 7, cap: int = 12) -> List[WeightConfig3]:
    """Diverse simplex grid over 3 weights (sums to 1), with unbiased truncation.

    This is used to generate the preference states for Quantum Pareto.
    It delegates to :func:`qkd_solver.weight_grid_simplex` so the classical/quantum
    pipelines share the exact same sampling logic.
    """
    try:
        # Package-style
        from .qkd_solver import weight_grid_simplex
    except Exception:  # pragma: no cover
        # Script-style
        from qkd_solver import weight_grid_simplex

    return weight_grid_simplex(n=int(n), cap=int(cap))


def make_period_network(base_net: QKDNetwork, demands_t: List[QKDDemand], capacities: List[float]) -> QKDNetwork:
    """Build a per-period snapshot network.

    Same topology; ``key_capacity`` is overwritten by the per-edge ``capacities`` computed from pools.
    """
    if len(capacities) != len(base_net.edges):
        raise ValueError("capacities length must match number of edges")

    edges: List[QKDEdge] = []
    for i, e in enumerate(base_net.edges):
        edges.append(
            QKDEdge(
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

    return QKDNetwork(n_nodes=int(base_net.n_nodes), edges=edges, demands=list(demands_t))


def run_multiperiod_mpc(
    base_net: QKDNetwork,
    demands_by_t: List[List[QKDDemand]],
    cfg: RunConfigQKD,
    policy: Optional[WeightConfig3] = None,
    weights_grid_n: int = 7,
    *,
    save_pareto: bool = True,
    pareto_out_dir: str = "pareto_figs",
    pareto_prefix: str = "pareto_t",
    pareto_dpi: int = 600,
    pareto_fmt: str = "png",
    pareto_labels: Tuple[str, str, str] = ("Unmet demand", "Total latency", "Resource/Risk"),
) -> List[PeriodResult]:
    """Run multi-period EP-QMOO-MPC (the main algorithm).

    Parameters
    ----------
    base_net:
        Network topology and key-pool dynamics parameters (key_pool_init/key_gen_rate/key_decay/...).
        Note: per-step capacities are computed from pools and substituted into per-period snapshots.
    demands_by_t:
        List of per-period demand lists.
    cfg:
        EP-QMOO solver configuration (QAOA + optimizer + encoding).
    policy:
        Optional scalarization weights used to choose ONE operating solution from the Pareto set each period.
        If None, selects lexicographically by (unmet, latency, resource).
    weights_grid_n:
        Controls the density of the preference-weight grid (before truncating to cfg.mo_num_weights).

    Returns
    -------
    history:
        List of PeriodResult (one per time step).
    """
    pools = init_key_pools(base_net)
    history: List[PeriodResult] = []

    weights = simple_weight_grid(n=int(weights_grid_n), cap=max(1, int(cfg.mo_num_weights)))

    for t, demands_t in enumerate(demands_by_t):
        pools_before = list(pools)
        caps = available_key_capacities(base_net, pools_before)

        # Solve this step with a capacity snapshot
        net_t = make_period_network(base_net, demands_t, caps)
        solver = EPQMOO_QKDRouting(net_t, cfg)
        solver._solve_quantum_pareto(weights)

        # --- Pareto FRONT export (3D-only, paper-ready PNG/JPG) ---
        pareto_points = solver.archive.as_points()
        pareto_fig_paths: List[str] = []
        if save_pareto and pareto_points:
            import os

            os.makedirs(str(pareto_out_dir), exist_ok=True)
            out_prefix = os.path.join(str(pareto_out_dir), f"{pareto_prefix}{t:03d}")

            # Preferred: use the solver helper (added in qkd_solver_updated.py)
            try:
                pareto_fig_paths = solver.save_pareto_front_image(
                    out_prefix=out_prefix,
                    labels=pareto_labels,
                    dpi=int(pareto_dpi),
                    fmt=str(pareto_fmt),
                )
            except AttributeError:
                # Fallback: call the plotter directly
                try:
                    from .pareto_plotter import save_pareto_figures  # package-style
                except Exception:  # pragma: no cover
                    from pareto_plotter import save_pareto_figures  # script-style

                pareto_fig_paths = save_pareto_figures(
                    pareto_points,
                    out_prefix=out_prefix,
                    labels=pareto_labels,
                    dpi=int(pareto_dpi),
                    fmt=str(pareto_fmt),
                )

        chosen = solver.pick_operating_solution(policy=policy)

        # Fallback: force-drop all demands if nothing decoded
        if chosen is None:
            chosen = {}
            for d_id, paths in solver.paths_by_demand.items():
                chosen[int(d_id)] = int(len(paths) - 1)  # drop option is appended last

        loads = compute_edge_loads(chosen, net_t, solver.paths_by_demand)
        pools_after = apply_key_consumption(base_net, pools_before, loads)

        enc = solver.last_encoding
        if enc is None:
            raise RuntimeError("solver.last_encoding not set; _solve_quantum_pareto must be called first.")

        obj = evaluate_qkd_objectives(chosen, net_t, enc, relay_penalty=float(cfg.relay_penalty))

        history.append(
            PeriodResult(
                t=int(t),
                pools_before=pools_before,
                capacities=list(caps),
                chosen=dict(chosen),
                objectives=tuple(float(x) for x in obj),
                edge_loads=list(loads),
                pools_after=pools_after,
                pareto_points=list(pareto_points),
                pareto_fig_paths=list(pareto_fig_paths),
            )
        )

        pools = pools_after

    return history


# ---------------------------------------------------------------------------
# Command-line entrypoint (Windows-friendly)
# ---------------------------------------------------------------------------

def _demo_demands_by_t(base: List[QKDDemand], T: int = 5, seed: int = 7) -> List[List[QKDDemand]]:
    """Create a small synthetic multi-period demand sequence for demos."""
    rng = random.Random(int(seed))
    out: List[List[QKDDemand]] = []
    for _t in range(int(T)):
        period: List[QKDDemand] = []
        for d in base:
            # +/- 10% fluctuation
            mult = 0.9 + 0.2 * rng.random()
            period.append(
                QKDDemand(
                    src=int(d.src),
                    dst=int(d.dst),
                    demand=float(d.demand) * float(mult),
                    max_latency=d.max_latency,
                )
            )
        out.append(period)
    return out


def _demo_network_with_pools(seed: int = 7) -> QKDNetwork:
    """Demo network with key-pool dynamics enabled on each edge."""
    net = tiny_qkd_demo_network(seed=seed)
    edges: List[QKDEdge] = []
    for e in net.edges:
        edges.append(
            QKDEdge(
                u=int(e.u),
                v=int(e.v),
                key_capacity=float(e.key_capacity),
                latency=float(e.latency),
                trusted=bool(e.trusted),
                key_pool_init=float(e.key_capacity),     # start full
                key_gen_rate=0.08 * float(e.key_capacity),  # regenerate each step
                key_storage_cap=1.20 * float(e.key_capacity),  # limited storage
                key_decay=0.02,  # small decay per step
            )
        )
    return QKDNetwork(n_nodes=int(net.n_nodes), edges=edges, demands=list(net.demands))


if __name__ == "__main__":
    # Main runnable example: multi-period EP-QMOO-MPC.
    base_net = _demo_network_with_pools(seed=7)
    demands_by_t = _demo_demands_by_t(base_net.demands, T=5, seed=7)

    cfg = RunConfigQKD(
        reps=1,
        maxiter=60,
        optimizer="SPSA",
        mo_num_weights=6,
        qp_train_objective="hypervolume",
        qp_train_shots=256,
        spsa_resamplings=2,
        ep_mode="penalty",
        ent_target=0.20,
        qp_pref_layers=1,
        k_paths=3,
        max_hops=5,
        allow_drop=True,
        relay_penalty=0.6,
    )

    hist = run_multiperiod_mpc(base_net, demands_by_t, cfg, policy=None, weights_grid_n=7)

    print("\n=== Multi-period EP-QMOO-MPC results ===")
    for r in hist:
        print(
            f"t={r.t}  obj(unmet,lat,res)={tuple(round(x,3) for x in r.objectives)}  "
            f"min_pool_after={min(r.pools_after):.3f}"
        )
        if r.pareto_fig_paths:
            print(f"    Pareto figures: {', '.join(r.pareto_fig_paths)}")