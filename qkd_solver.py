from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import math

import numpy as np


@dataclass
class _MinimizeResult:
    """Lightweight result object matching the parts of Qiskit's optimizer result we use."""

    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    message: str = "OK"


class StableSPSA:
    """A numerically stable SPSA optimizer with an explicit gain schedule.

    This *keeps SPSA* but avoids Qiskit's SPSA auto-calibration failure mode where the
    estimated gradient magnitude becomes ~0 and internal division-by-zero yields NaNs.
    It also guards against NaN/Inf updates and clips parameters to a reasonable range.
    """

    def __init__(
        self,
        maxiter: int,
        *,
        seed: int = 7,
        a: float = 0.05,
        c: float = 0.10,
        A: Optional[float] = None,
        alpha: float = 0.602,
        gamma: float = 0.101,
        resamplings: int = 1,
        max_step_norm: float = 0.50,
        param_clip: float = math.pi,
    ) -> None:
        self.maxiter = int(maxiter)
        self.seed = int(seed)
        self.a = float(a)
        self.c = float(c)
        self.A = float(A) if A is not None else 0.1 * float(self.maxiter)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.resamplings = int(max(1, resamplings))
        self.max_step_norm = float(max_step_norm)
        self.param_clip = float(param_clip)

    @staticmethod
    def _is_finite_scalar(x: float) -> bool:
        return bool(np.isfinite(x))

    def _avg_eval(self, fun: Callable[[np.ndarray], float], x: np.ndarray) -> float:
        vals: List[float] = []
        for _ in range(self.resamplings):
            try:
                v = float(fun(x))
            except Exception:
                v = float("inf")
            if self._is_finite_scalar(v):
                vals.append(v)
        if not vals:
            return float("inf")
        return float(np.mean(vals))

    def minimize(self, fun: Callable[[np.ndarray], float], x0: np.ndarray) -> _MinimizeResult:
        rng = np.random.default_rng(self.seed)
        x = np.array(x0, dtype=float)

        # Ensure we start finite and within bounds
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -self.param_clip, self.param_clip)

        best_x = x.copy()
        best_f = self._avg_eval(fun, x)
        nfev = self.resamplings

        for k in range(1, self.maxiter + 1):
            ak = self.a / ((self.A + k) ** self.alpha)
            ck = self.c / ((k) ** self.gamma)

            delta = rng.choice([-1.0, 1.0], size=x.shape)

            x_plus = np.clip(x + ck * delta, -self.param_clip, self.param_clip)
            x_minus = np.clip(x - ck * delta, -self.param_clip, self.param_clip)

            f_plus = self._avg_eval(fun, x_plus)
            f_minus = self._avg_eval(fun, x_minus)
            nfev += 2 * self.resamplings

            if (not self._is_finite_scalar(f_plus)) or (not self._is_finite_scalar(f_minus)):
                continue

            ghat = ((f_plus - f_minus) / (2.0 * ck)) * delta
            if not np.all(np.isfinite(ghat)):
                continue

            step = ak * ghat
            step_norm = float(np.linalg.norm(step))
            if step_norm > self.max_step_norm and step_norm > 0.0:
                step = step * (self.max_step_norm / step_norm)

            x_new = np.clip(x - step, -self.param_clip, self.param_clip)
            f_new = self._avg_eval(fun, x_new)
            nfev += self.resamplings

            if self._is_finite_scalar(f_new):
                x = x_new
                if f_new < best_f:
                    best_f = float(f_new)
                    best_x = x.copy()

        return _MinimizeResult(x=best_x, fun=float(best_f), nit=int(self.maxiter), nfev=int(nfev))


try:
    from .archive import ParetoArchive
    from .metrics import hypervolume_3d
    from .mixer import build_onehot_xy_mixer

    from .qkd_model import QKDNetwork
    from .qkd_encoding import (
        QKDRoutingEncoding,
        build_candidate_paths,
        build_qkd_routing_qp,
        decode_qkd_choice_from_bitstring,
        evaluate_qkd_objectives,
        repair_qkd_routing,
    )
except ImportError:  # pragma: no cover
    from archive import ParetoArchive
    from metrics import hypervolume_3d
    from mixer import build_onehot_xy_mixer

    from qkd_model import QKDNetwork
    from qkd_encoding import (
        QKDRoutingEncoding,
        build_candidate_paths,
        build_qkd_routing_qp,
        decode_qkd_choice_from_bitstring,
        evaluate_qkd_objectives,
        repair_qkd_routing,
    )

# Optional dependencies (required for this solver)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector, partial_trace, entropy
    try:
        # Qiskit versions vary
        from qiskit.synthesis.evolution import LieTrotter
    except Exception:  # pragma: no cover
        from qiskit.synthesis import LieTrotter
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing optional dependency: qiskit. Install with `pip install qiskit` "
        "(and ensure your qiskit version includes PauliEvolutionGate)."
    ) from e

try:
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing optional dependency: qiskit-optimization. Install with `pip install qiskit-optimization`."
    ) from e


@dataclass
class RunConfigQKD:
    # QAOA / optimization
    reps: int = 1
    seed: int = 7
    maxiter: int = 80
    optimizer: str = "SPSA"  # supported: SPSA
    sampler: str = "statevector"

    # Simulation / circuit compilation (prevents dense matrix blow-ups)
    evolution_trotter_reps: int = 1
    transpile_basis_gates: Tuple[str, ...] = ("rz", "sx", "x", "cx")
    transpile_optimization_level: int = 0

    # QUBO conversion
    penalty: Optional[float] = None

    # Mixer
    mixer_topology: str = "ring"
    mixer_coeff: float = 1.0

    # Candidate paths
    k_paths: int = 3
    max_hops: int = 5
    allow_drop: bool = True
    relay_penalty: float = 0.5

    # Multiobjective configuration (Quantum Pareto only)
    mo_num_weights: int = 8

    # Preference register preparation
    qp_pref_layers: int = 1
    qp_pref_entangle: bool = True

    # HV-in-loop training
    qp_train_objective: str = "hypervolume"  # "hypervolume" or "expected_cost"
    qp_train_top_k: int = 28
    qp_hv_ref_scale: float = 1.15


    # Shot-based HV estimation during training (keeps the objective compatible with SPSA).
    qp_train_shots: int = 256

    # SPSA gain schedule + stability guards (used when optimizer='SPSA')
    spsa_a: float = 0.05
    spsa_c: float = 0.10
    spsa_A: Optional[float] = None
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_resamplings: int = 1
    spsa_max_step_norm: float = 0.50
    spsa_param_clip: float = math.pi

    # EP: entanglement constraint on decision qubits (statevector-only)
    ep_mode: str = "penalty"  # "none" | "penalty" | "aug_lagrangian"
    ep_alpha: float = 0.6
    ent_target: float = 0.25
    ep_outer_iters: int = 4
    ep_lambda0: float = 0.0
    ep_rho0: float = 5.0
    ep_rho_mult: float = 2.0
    ep_tol: float = 1e-3
    ep_inner_maxiter: Optional[int] = None

    # Archive and decoding
    archive_max_size: int = 300
    top_k_bitstrings: int = 80


@dataclass(frozen=True)
class WeightConfig3:
    w_unmet: float
    w_latency: float
    w_resource: float


def _lattice_simplex_points_3(res: int) -> np.ndarray:
    """Return all (a,b,c) on the 3-simplex lattice with step=1/res.

    The returned array has shape (N, 3) and rows sum to 1.
    """
    res = int(res)
    if res <= 0:
        return np.zeros((0, 3), dtype=float)

    pts = []
    for i in range(res + 1):
        for j in range(res + 1 - i):
            k = res - i - j
            pts.append((i / res, j / res, k / res))
    return np.asarray(pts, dtype=float)


def _farthest_point_sample(points: np.ndarray, k: int, start: Optional[np.ndarray] = None) -> List[int]:
    """Greedy farthest-point sampling in Euclidean space.

    Selects k points that are spread out (diversity sampling).
    Deterministic given the input ordering.
    """
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    k = int(k)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    if start is None:
        start = np.full(pts.shape[1], 1.0 / float(pts.shape[1]), dtype=float)
    start = np.asarray(start, dtype=float)

    # Start with the point closest to `start` (typically the barycenter).
    d0 = np.linalg.norm(pts - start[None, :], axis=1)
    first = int(np.argmin(d0))
    chosen = [first]

    # Track each point's distance to the nearest chosen point.
    min_d = np.linalg.norm(pts - pts[first][None, :], axis=1)

    for _ in range(1, k):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        d = np.linalg.norm(pts - pts[nxt][None, :], axis=1)
        min_d = np.minimum(min_d, d)

    return chosen


def weight_grid_simplex(n: int = 7, cap: int = 12) -> List[WeightConfig3]:
    """Diverse simplex grid over 3 weights (sums to 1), with unbiased truncation.

    - Builds a uniform lattice on the simplex (includes corners/extremes).
    - If more than `cap` points exist, it down-samples via farthest-point sampling
      so the selected weights remain well-spread (no ordering bias).
    """
    cap = int(cap)
    if cap <= 0:
        return []

    # Backwards-compatible: `n` controls lattice density; old code used `linspace(..., n)`.
    # Here, `res = n-1` means "n distinct values per axis" on the simplex.
    res = max(2, int(n) - 1)

    pts = _lattice_simplex_points_3(res)
    if pts.size == 0:
        return [WeightConfig3(1 / 3, 1 / 3, 1 / 3)]

    idxs = _farthest_point_sample(pts, k=min(cap, int(pts.shape[0])), start=np.array([1 / 3, 1 / 3, 1 / 3]))
    out = [WeightConfig3(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])) for i in idxs]

    # Safety: never return empty
    return out if out else [WeightConfig3(1 / 3, 1 / 3, 1 / 3)]


def _simple_weight_grid(n: int = 7, cap: int = 12) -> List[WeightConfig3]:
    """Backward-compatible alias for :func:`weight_grid_simplex`."""
    return weight_grid_simplex(n=n, cap=cap)


def _mean_single_qubit_entropy_subset(sv: Statevector, qubits: List[int]) -> float:
    n = sv.num_qubits
    entropies = []
    for qi in qubits:
        rho_i = partial_trace(sv, [q for q in range(n) if q != qi])
        entropies.append(float(entropy(rho_i, base=2)))
    return float(np.mean(entropies)) if entropies else 0.0


def _conditional_mean_single_qubit_entropy_decision_given_pref(
    sv: Statevector,
    pref_qubits: List[int],
    decision_qubits: List[int],
    prob_epsilon: float = 1e-15,
) -> float:
    """Expected mean single-qubit entropy on decision qubits, conditioned on preference outcomes."""
    if not pref_qubits:
        return _mean_single_qubit_entropy_subset(sv, decision_qubits)

    pref_qubits = sorted(pref_qubits)
    decision_qubits = sorted(decision_qubits)

    n_dec = len(decision_qubits)
    if decision_qubits == list(range(n_dec)) and pref_qubits == list(range(n_dec, n_dec + len(pref_qubits))):
        amps = np.asarray(sv.data, dtype=complex)
        K = 1 << len(pref_qubits)
        ent = 0.0
        idx_base = np.arange(1 << n_dec, dtype=np.int64)
        for k in range(K):
            idxs = idx_base + (np.int64(k) << np.int64(n_dec))
            sub = amps[idxs]
            pk = float(np.vdot(sub, sub).real)
            if pk <= prob_epsilon:
                continue
            sv_dec = Statevector(sub / math.sqrt(pk))
            ent += pk * _mean_single_qubit_entropy_subset(sv_dec, list(range(n_dec)))
        return float(ent)

    amps = np.asarray(sv.data, dtype=complex)
    K = 1 << len(pref_qubits)
    ent = 0.0
    idx_all = np.arange(amps.shape[0], dtype=np.int64)
    bits_pref = [(((idx_all >> q) & 1) == 1) for q in pref_qubits]

    for k in range(K):
        mask = np.ones(amps.shape[0], dtype=bool)
        for i in range(len(pref_qubits)):
            if (k >> i) & 1:
                mask &= bits_pref[i]
            else:
                mask &= ~bits_pref[i]
        if not mask.any():
            continue
        sub = amps[mask]
        pk = float(np.vdot(sub, sub).real)
        if pk <= prob_epsilon:
            continue
        proj = np.zeros_like(amps)
        proj[mask] = sub
        proj /= math.sqrt(pk)
        sv_k = Statevector(proj)
        ent_qubits = []
        for dq in decision_qubits:
            rho_i = partial_trace(sv_k, [q for q in range(sv.num_qubits) if q != dq])
            ent_qubits.append(float(entropy(rho_i, base=2)))
        ent += pk * (float(np.mean(ent_qubits)) if ent_qubits else 0.0)

    return float(ent)


class EPQMOO_QKDRouting:
    """EP-QMOO (HV-in-loop Quantum Pareto) for a trusted-relay QKD routing window."""

    def __init__(self, net: QKDNetwork, cfg: RunConfigQKD) -> None:
        self.net = net
        self.cfg = cfg
        self.archive = ParetoArchive(max_size=cfg.archive_max_size)

        self.paths_by_demand = build_candidate_paths(
            net,
            k_paths=int(cfg.k_paths),
            max_hops=int(cfg.max_hops),
            allow_drop=bool(cfg.allow_drop),
        )

        # Filled after running _solve_quantum_pareto()
        self.last_encoding: Optional[QKDRoutingEncoding] = None
        self.last_weights: List[WeightConfig3] = []
        self.best_objectives_per_pref: List[Optional[Tuple[float, float, float]]] = []
        self.best_solutions_per_pref: List[Optional[Dict[int, int]]] = []
        self.best_scores_per_pref: List[float] = []

    def _compile_statevector_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Compile/unroll the circuit so Statevector simulation never calls Gate.to_matrix().

        Without this, PauliEvolutionGate (and its controlled versions) can trigger a dense
        2^n x 2^n matrix construction inside Statevector.from_instruction, which is
        infeasible beyond tiny n.
        """
        basis = list(getattr(self.cfg, "transpile_basis_gates", ("rz", "sx", "x", "cx")))
        opt_level = int(getattr(self.cfg, "transpile_optimization_level", 0))
        try:
            return transpile(qc, basis_gates=basis, optimization_level=opt_level)
        except Exception:
            # Fallback: best-effort recursive decomposition (slower but robust)
            out = qc
            for _ in range(12):
                out = out.decompose()
            return out

    def _optimizer(self, maxiter: Optional[int] = None):
        """Return the optimizer used for parameter training.

        This codebase uses the stable SPSA implementation in this module.
        """
        it = self.cfg.maxiter if maxiter is None else int(maxiter)
        opt_name = str(getattr(self.cfg, 'optimizer', 'SPSA')).strip().upper()
        if opt_name not in ('SPSA', 'STABLESPSA'):
            raise ValueError("optimizer must be 'SPSA' (or 'STABLESPSA').")
        return StableSPSA(
            maxiter=it,
            seed=int(self.cfg.seed),
            a=float(getattr(self.cfg, 'spsa_a', 0.05)),
            c=float(getattr(self.cfg, 'spsa_c', 0.10)),
            A=getattr(self.cfg, 'spsa_A', None),
            alpha=float(getattr(self.cfg, 'spsa_alpha', 0.602)),
            gamma=float(getattr(self.cfg, 'spsa_gamma', 0.101)),
            resamplings=int(getattr(self.cfg, 'spsa_resamplings', 1)),
            max_step_norm=float(getattr(self.cfg, 'spsa_max_step_norm', 0.50)),
            param_clip=float(getattr(self.cfg, 'spsa_param_clip', math.pi)),
        )

    def _hv_reference_point_3d(self, enc: QKDRoutingEncoding) -> Tuple[float, float, float]:
        total_demand = float(sum(d.demand for d in self.net.demands))
        worst_lat = 0.0
        worst_res = 0.0
        for d_id, dem in enumerate(self.net.demands):
            paths = enc.paths_by_demand[d_id]
            lat_max = max((p.latency for p in paths), default=0.0)
            worst_lat += float(lat_max)
            max_hops = max((len(p.edge_ids) for p in paths), default=0)
            max_rel = max((p.relays for p in paths), default=0)
            worst_res += float(dem.demand) * float(max_hops) + float(self.cfg.relay_penalty) * float(max_rel)

        scale = float(getattr(self.cfg, "qp_hv_ref_scale", 1.15))
        eps = 1.0
        return (scale * total_demand + eps, scale * worst_lat + eps, scale * worst_res + eps)

    def _optimize_statevector_with_ep(
        self,
        params: List,
        eval_base_cost_and_sv,
        decision_qubits: List[int],
        pref_qubits: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Statevector]:
        if self.cfg.sampler != "statevector":
            raise ValueError("This implementation requires sampler='statevector'.")

        ep_mode = self.cfg.ep_mode
        if ep_mode not in ("none", "penalty", "aug_lagrangian"):
            raise ValueError("ep_mode must be 'none'|'penalty'|'aug_lagrangian'")

        rng0 = np.random.default_rng(int(self.cfg.seed))
        x0 = (np.zeros(len(params), dtype=float) + 0.2) + rng0.normal(
            loc=0.0, scale=0.02, size=len(params)
        )

        def eval_cost_ent(theta: np.ndarray) -> Tuple[float, float, Statevector]:
            base_cost, sv = eval_base_cost_and_sv(theta)
            if pref_qubits:
                ent = _conditional_mean_single_qubit_entropy_decision_given_pref(
                    sv=sv,
                    pref_qubits=pref_qubits,
                    decision_qubits=decision_qubits,
                )
            else:
                ent = _mean_single_qubit_entropy_subset(sv, decision_qubits)
            return float(base_cost), float(ent), sv

        if ep_mode == "none":
            opt = self._optimizer()

            def loss(th: np.ndarray) -> float:
                c, _e, _sv = eval_cost_ent(th)
                return float(c)

            res = opt.minimize(loss, x0)
            theta_opt = np.array(res.x, dtype=float)
            _c, _e, sv = eval_cost_ent(theta_opt)
            return theta_opt, sv

        if ep_mode == "penalty" or float(self.cfg.ep_alpha) <= 0.0:
            opt = self._optimizer()

            def loss(th: np.ndarray) -> float:
                c, e, _sv = eval_cost_ent(th)
                g = max(0.0, float(self.cfg.ent_target) - e)
                return float(c + float(self.cfg.ep_alpha) * (g ** 2))

            res = opt.minimize(loss, x0)
            theta_opt = np.array(res.x, dtype=float)
            _c, _e, sv = eval_cost_ent(theta_opt)
            return theta_opt, sv

        lam = float(self.cfg.ep_lambda0)
        rho = float(self.cfg.ep_rho0)
        theta = np.array(x0, dtype=float)

        inner_maxiter = self.cfg.ep_inner_maxiter
        if inner_maxiter is None:
            inner_maxiter = max(10, int(self.cfg.maxiter // max(1, int(self.cfg.ep_outer_iters))))

        sv_final: Optional[Statevector] = None
        for _outer in range(int(self.cfg.ep_outer_iters)):
            opt = self._optimizer(maxiter=inner_maxiter)

            def loss(th: np.ndarray) -> float:
                c, e, _sv = eval_cost_ent(th)
                g = max(0.0, float(self.cfg.ent_target) - e)
                return float(c + lam * g + 0.5 * rho * (g ** 2))

            res = opt.minimize(loss, theta)
            theta = np.array(res.x, dtype=float)

            c, e, sv_final = eval_cost_ent(theta)
            g = max(0.0, float(self.cfg.ent_target) - e)

            if g <= float(self.cfg.ep_tol):
                break

            lam = max(0.0, lam + rho * g)
            rho = rho * float(self.cfg.ep_rho_mult)

        if sv_final is None:
            _c, _e, sv_final = eval_cost_ent(theta)

        return theta, sv_final

    def _pad_weights_to_pow2(self, weights: List[WeightConfig3]) -> List[WeightConfig3]:
        if not weights:
            return [WeightConfig3(1 / 3, 1 / 3, 1 / 3)]
        k = len(weights)
        m = int(math.ceil(math.log2(k)))
        k2 = 1 << m
        if k2 == k:
            return weights

        # Important: do NOT pad by repeating the last weight (that biases preference states).
        # Instead, repeat weights round-robin so multiplicities differ by at most 1.
        out = list(weights)
        pad = k2 - k
        for t in range(pad):
            out.append(weights[t % k])
        return out

    def _build_quantum_pareto_circuit(
        self,
        num_decision_qubits: int,
        cost_ops: List,
        mixer_op,
        pref_qubits: int,
    ) -> Tuple[QuantumCircuit, List]:
        n = int(num_decision_qubits)
        m = int(pref_qubits)
        qc = QuantumCircuit(n + m)
        synth = LieTrotter(reps=int(getattr(self.cfg, "evolution_trotter_reps", 1)))

        pref_params: List = []
        L = int(getattr(self.cfg, "qp_pref_layers", 1))
        if m > 0:
            if L < 1:
                raise ValueError("qp_pref_layers must be >= 1.")
            phis = ParameterVector("phi_pref", L * m)
            psis = ParameterVector("psi_pref", L * m)
            pref_params = list(phis) + list(psis)
            for layer in range(L):
                for i in range(m):
                    qc.ry(phis[layer * m + i], n + i)
                    qc.rz(psis[layer * m + i], n + i)
                if bool(getattr(self.cfg, "qp_pref_entangle", True)) and m > 1:
                    for i in range(m - 1):
                        qc.cx(n + i, n + i + 1)
                    qc.cx(n + m - 1, n + 0)

        gammas = ParameterVector("gamma", self.cfg.reps)
        betas = ParameterVector("beta", self.cfg.reps)

        controls = list(range(n, n + m))
        targets = list(range(n))

        for layer in range(self.cfg.reps):
            gamma = gammas[layer]
            beta = betas[layer]

            for k, Hk in enumerate(cost_ops):
                flips = [controls[i] for i in range(m) if ((k >> i) & 1) == 0]
                if flips:
                    qc.x(flips)

                evol = PauliEvolutionGate(Hk, time=gamma, synthesis=synth)
                if m > 0:
                    cevol = evol.control(m)
                    qc.append(cevol, controls + targets)
                else:
                    qc.append(evol, targets)

                if flips:
                    qc.x(flips)

            mix_gate = PauliEvolutionGate(mixer_op, time=beta, synthesis=synth)
            qc.append(mix_gate, targets)

        params = list(pref_params) + list(gammas) + list(betas)
        return qc, params

    def _solve_quantum_pareto(self, weights: List[WeightConfig3]) -> None:
        weights = self._pad_weights_to_pow2(weights)
        K = len(weights)
        m = int(math.log2(K))
        assert (1 << m) == K

        enc0: Optional[QKDRoutingEncoding] = None
        cost_ops = []
        offsets = []

        for w in weights:
            qp, enc = build_qkd_routing_qp(
                self.net,
                w_unmet=w.w_unmet,
                w_latency=w.w_latency,
                w_resource=w.w_resource,
                paths_by_demand=self.paths_by_demand,
                k_paths=int(self.cfg.k_paths),
                max_hops=int(self.cfg.max_hops),
                allow_drop=bool(self.cfg.allow_drop),
                relay_penalty=float(self.cfg.relay_penalty),
            )
            if enc0 is None:
                enc0 = enc
            else:
                if enc.num_vars != enc0.num_vars or enc.key_to_index != enc0.key_to_index:
                    raise ValueError("Quantum Pareto requires identical encodings across weights.")
            qubo = QuadraticProgramToQubo(penalty=self.cfg.penalty).convert(qp)
            op, off = to_ising(qubo)
            cost_ops.append(op)
            offsets.append(float(off))

        assert enc0 is not None
        enc = enc0
        self.last_encoding = enc
        self.last_weights = list(weights)
        n = int(enc.num_vars)

        mixer_op = build_onehot_xy_mixer(
            num_qubits=n,
            groups=enc.groups,
            topology=self.cfg.mixer_topology,
            coefficient=self.cfg.mixer_coeff,
        )

        qc, params = self._build_quantum_pareto_circuit(
            num_decision_qubits=n,
            cost_ops=cost_ops,
            mixer_op=mixer_op,
            pref_qubits=m,
        )

        # Compile once to avoid dense-matrix fallback in Statevector simulation
        qc_sim = self._compile_statevector_circuit(qc)

        decision_qubits = list(range(n))
        pref_qubits = list(range(n, n + m))

        train_obj = str(getattr(self.cfg, "qp_train_objective", "hypervolume")).strip().lower()
        hv_ref = self._hv_reference_point_3d(enc)
        train_top_k_cfg = int(getattr(self.cfg, "qp_train_top_k", int(self.cfg.top_k_bitstrings)))
        train_shots = int(getattr(self.cfg, "qp_train_shots", 256))
        rng_train = np.random.default_rng(int(self.cfg.seed) + 9917)
        if train_top_k_cfg <= 0:
            train_top_k_cfg = 1

        _dec_obj_cache: Dict[int, Tuple[Tuple[float, float, float], Dict[int, int]]] = {}

        def _obj_choice_from_dec_int(dec_int: int) -> Tuple[Tuple[float, float, float], Dict[int, int]]:
            dec_int = int(dec_int)
            if dec_int in _dec_obj_cache:
                return _dec_obj_cache[dec_int]
            bits_le = "".join("1" if ((dec_int >> i) & 1) else "0" for i in range(n))
            choice = decode_qkd_choice_from_bitstring(bits_le, enc)
            choice = repair_qkd_routing(choice, self.net, enc)
            obj = evaluate_qkd_objectives(choice, self.net, enc, relay_penalty=float(self.cfg.relay_penalty))
            _dec_obj_cache[dec_int] = (obj, choice)
            return obj, choice

        def _approx_pareto_points_from_statevector(sv: Statevector) -> List[Tuple[float, float, float]]:
            amps = np.asarray(sv.data, dtype=complex)
            probs = np.abs(amps) ** 2

            psum = float(np.sum(probs))
            if psum <= 0.0 or not np.isfinite(psum):
                return []
            probs = probs / psum

            shots = int(max(1, train_shots))
            idxs = rng_train.choice(probs.size, size=shots, replace=True, p=probs).astype(np.int64)

            best_per_k: List[Tuple[float, Optional[Tuple[float, float, float]]]] = [
                (float("inf"), None) for _ in range(K)
            ]

            for idx in idxs:
                idx = int(idx)
                pref = idx >> n
                dec = idx & ((1 << n) - 1)

                obj, _choice = _obj_choice_from_dec_int(dec)
                w = weights[pref]
                score = float(w.w_unmet * obj[0] + w.w_latency * obj[1] + w.w_resource * obj[2])
                if score < best_per_k[pref][0]:
                    best_per_k[pref] = (score, obj)

            # If a preference state was not sampled, fall back to the conditional argmax in that block.
            for k in range(K):
                if best_per_k[k][1] is not None:
                    continue
                start = (k << n)
                end = ((k + 1) << n)
                if end <= start or end > probs.size:
                    continue
                local = probs[start:end]
                if local.size == 0:
                    continue
                idx_local = start + int(np.argmax(local))
                dec = idx_local & ((1 << n) - 1)
                obj, _choice = _obj_choice_from_dec_int(dec)
                w = weights[k]
                score = float(w.w_unmet * obj[0] + w.w_latency * obj[1] + w.w_resource * obj[2])
                best_per_k[k] = (score, obj)

            return [obj for _s, obj in best_per_k if obj is not None]

        def eval_base_cost_and_sv(theta: np.ndarray) -> Tuple[float, Statevector]:
            bound = qc_sim.assign_parameters({p: float(theta[i]) for i, p in enumerate(params)}, inplace=False)
            sv = Statevector.from_instruction(bound)

            if train_obj in ("expected_cost", "expected_energy", "energy"):
                psi = np.asarray(sv.data, dtype=complex)
                psi2 = psi.reshape((K, 1 << n))
                probs_pref = np.sum(np.abs(psi2) ** 2, axis=1).real

                base_cost = 0.0
                for k in range(K):
                    pk = float(probs_pref[k])
                    if pk <= 1e-15:
                        continue
                    v = psi2[k] / math.sqrt(pk)
                    sv_k = Statevector(v)
                    ek = float(np.real(sv_k.expectation_value(cost_ops[k])) + offsets[k])
                    base_cost += pk * ek
                return float(base_cost), sv

            if train_obj in ("hypervolume", "hv", "hv3"):
                pts = _approx_pareto_points_from_statevector(sv)
                hv = float(hypervolume_3d(pts, ref=hv_ref)) if pts else 0.0
                return float(-hv), sv

            raise ValueError("qp_train_objective must be 'expected_cost' or 'hypervolume'.")

        _theta_opt, sv = self._optimize_statevector_with_ep(
            params=params,
            eval_base_cost_and_sv=eval_base_cost_and_sv,
            decision_qubits=decision_qubits,
            pref_qubits=pref_qubits,
        )

        psi = np.asarray(sv.data, dtype=complex)
        probs = np.abs(psi) ** 2
        top_k_cfg = int(self.cfg.top_k_bitstrings)
        if top_k_cfg <= 0:
            idxs = np.array([int(np.argmax(probs))], dtype=np.int64)
        else:
            top_k = min(top_k_cfg, probs.size)
            idxs = np.argpartition(-probs, top_k - 1)[:top_k]
            idxs = idxs[np.argsort(-probs[idxs])]

        best_per_k: List[Tuple[float, Optional[Tuple[float, float, float]], Optional[Dict[int, int]]]] = [(float("inf"), None, None) for _ in range(K)]

        for idx in idxs:
            idx = int(idx)
            pref = idx >> n
            dec = idx & ((1 << n) - 1)

            obj, choice = _obj_choice_from_dec_int(dec)
            w = weights[pref]
            score = float(w.w_unmet * obj[0] + w.w_latency * obj[1] + w.w_resource * obj[2])

            if score < best_per_k[pref][0]:
                best_per_k[pref] = (score, obj, choice)

        for k in range(K):
            _score, obj, choice = best_per_k[k]
            if obj is None:
                continue
            self.archive.update(obj, payload={"pref_index": int(k), "choice": choice, "score": float(_score)})


        # Expose per-preference best solutions for downstream (e.g., multi-period MPC).
        self.best_scores_per_pref = [float(s) for (s, _o, _c) in best_per_k]
        self.best_objectives_per_pref = [o for (_s, o, _c) in best_per_k]
        self.best_solutions_per_pref = [c for (_s, _o, c) in best_per_k]

    def pick_operating_solution(self, policy: Optional[WeightConfig3] = None) -> Optional[Dict[int, int]]:
        """Pick ONE routing to execute from the last Quantum Pareto solve.

        Parameters
        ----------
        policy:
            If provided, selects argmin (policy · objectives) among available per-preference best solutions.
            If None, selects lexicographically by (unmet, latency, resource).

        Returns
        -------
        Optional[Dict[int,int]]
            demand_id -> chosen path_idx, or None if no solution is available.
        """
        if not self.best_solutions_per_pref:
            return None

        candidates: List[Tuple[Tuple[float, float, float], Dict[int, int]]] = []
        for obj, choice in zip(self.best_objectives_per_pref, self.best_solutions_per_pref):
            if obj is None or choice is None:
                continue
            candidates.append(((float(obj[0]), float(obj[1]), float(obj[2])), dict(choice)))

        if not candidates:
            return None

        if policy is None:
            candidates.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]))
            return candidates[0][1]

        w1, w2, w3 = float(policy.w_unmet), float(policy.w_latency), float(policy.w_resource)
        candidates.sort(
            key=lambda t: (
                w1 * t[0][0] + w2 * t[0][1] + w3 * t[0][2],
                t[0][0],
                t[0][1],
                t[0][2],
            )
        )
        return candidates[0][1]

    def save_pareto_front_image(
        self,
        out_prefix: str = "pareto_front",
        labels: Optional[Tuple[str, str, str]] = ("Unmet demand", "Total latency", "Resource/Risk"),
        dpi: int = 600,
        fmt: str = "png",
    ) -> List[str]:
        """Save the current Pareto front as an image file (PNG/JPG).

        This uses the non-dominated archive built by :meth:`_solve_quantum_pareto`.

        Parameters
        ----------
        out_prefix:
            Output path prefix **without** extension.
            Example: "figs/pareto_t000" -> saves "figs/pareto_t000_3d.png".
        labels:
            Axis labels (only used for 3-objective plots). Set to None to use generic labels.
        dpi:
            Figure DPI (600 recommended for journal-ready raster export).
        fmt:
            Image format/extension (e.g., "png" or "jpg").

        Returns
        -------
        List[str]
            Paths of saved image files. Empty if the archive is empty.
        """
        points = self.archive.as_points()
        if not points:
            return []

        # Local import keeps matplotlib as an optional dependency for solver-only use.
        try:
            from .pareto_plotter import save_pareto_figures  # package-style
        except Exception:  # pragma: no cover
            from pareto_plotter import save_pareto_figures  # script-style

        if labels is None:
            labels = ("Objective 1", "Objective 2", "Objective 3")

        return save_pareto_figures(
            points,
            out_prefix=str(out_prefix),
            labels=labels,
            dpi=int(dpi),
            fmt=str(fmt),
        )

# Note: This module provides the per-period EP-QMOO solver used by qkd_multiperiod.py.

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "Single-window execution has been removed. "
        "Run the multi-period algorithm instead:  python qkd_multiperiod.py"
    )