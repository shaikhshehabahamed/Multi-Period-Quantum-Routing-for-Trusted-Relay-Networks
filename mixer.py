from __future__ import annotations

from typing import Sequence, Tuple
from qiskit.quantum_info import SparsePauliOp


def _pauli_term(num_qubits: int, pauli: str, qubits: Tuple[int, int]) -> str:
    """Create a Pauli label for a 2-qubit term acting on (i,j).

    Qiskit uses big-endian Pauli labels, hence the reverse at the end.
    """
    i, j = qubits
    label = ["I"] * num_qubits
    label[i] = pauli[0]
    label[j] = pauli[1]
    return "".join(reversed(label))


def build_onehot_xy_mixer(
    num_qubits: int,
    groups: Sequence[Sequence[int]],
    topology: str = "ring",
    coefficient: float = 1.0,
) -> SparsePauliOp:
    """Build an XY mixer Hamiltonian to preserve one-hot structure inside groups.

    XY terms (XX + YY) preserve Hamming weight within the acted-on subset.
    If each group is initialized with Hamming weight 1 (one-hot), it remains one-hot
    under this mixer.

    This is the core *feasible-subspace* mechanism used here:
      - operation-start one-hot groups (and optionally makespan/tardiness one-hot groups)
        remain valid during QAOA mixing.

    Note:
      This does NOT enforce machine capacity / precedence constraints by itself.
      Those are enforced (softly) via QUBO penalties, and we also decode+repair to a feasible schedule.
    """
    paulis = []
    coeffs = []

    for group in groups:
        g = list(group)
        if len(g) < 2:
            continue

        edges = []
        if topology == "ring":
            for k in range(len(g)):
                a = g[k]
                b = g[(k + 1) % len(g)]
                if a != b:
                    edges.append((a, b))
        elif topology == "complete":
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    edges.append((g[i], g[j]))
        else:
            raise ValueError("topology must be 'ring' or 'complete'")

        for (a, b) in edges:
            paulis.append(_pauli_term(num_qubits, "XX", (a, b)))
            coeffs.append(0.5 * coefficient)
            paulis.append(_pauli_term(num_qubits, "YY", (a, b)))
            coeffs.append(0.5 * coefficient)

    if not paulis:
        # fallback: trivial mixer (should not happen for meaningful groups)
        paulis = ["I" * num_qubits]
        coeffs = [0.0]

    return SparsePauliOp(paulis, coeffs)