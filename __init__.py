"""EP-QMOO for QKD routing (multi-period MPC is the main algorithm).

Public API focuses on:
- Network + key-pool dynamics (qkd_model)
- Encoding + repair + objective evaluation (qkd_encoding)
- Multi-period runner (qkd_multiperiod.run_multiperiod_mpc)

The per-window quantum solver remains available internally in qkd_solver, but is not the
recommended entrypoint for experiments (use the MPC runner).
"""

from .archive import ParetoArchive
from .metrics import hypervolume_2d, hypervolume_3d, pareto_filter
from .mixer import build_onehot_xy_mixer

from .qkd_model import (
    QKDEdge,
    QKDDemand,
    QKDNetwork,
    tiny_qkd_demo_network,
    init_key_pools,
    available_key_capacities,
    apply_key_consumption,
)

from .qkd_encoding import (
    QKDPath,
    QKDRoutingEncoding,
    build_candidate_paths,
    build_qkd_routing_qp,
    decode_qkd_choice_from_bitstring,
    evaluate_qkd_objectives,
    repair_qkd_routing,
    compute_edge_loads,
)

from .qkd_solver import RunConfigQKD, WeightConfig3
from .qkd_multiperiod import PeriodResult, run_multiperiod_mpc, simple_weight_grid

__all__ = [
    "ParetoArchive",
    "hypervolume_2d",
    "hypervolume_3d",
    "pareto_filter",
    "build_onehot_xy_mixer",
    "QKDEdge",
    "QKDDemand",
    "QKDNetwork",
    "QKDPath",
    "tiny_qkd_demo_network",
    "QKDRoutingEncoding",
    "build_candidate_paths",
    "build_qkd_routing_qp",
    "decode_qkd_choice_from_bitstring",
    "evaluate_qkd_objectives",
    "repair_qkd_routing",
    "compute_edge_loads",
    "RunConfigQKD",
    "WeightConfig3",
    "init_key_pools",
    "available_key_capacities",
    "apply_key_consumption",
    "PeriodResult",
    "simple_weight_grid",
    "run_multiperiod_mpc",
]