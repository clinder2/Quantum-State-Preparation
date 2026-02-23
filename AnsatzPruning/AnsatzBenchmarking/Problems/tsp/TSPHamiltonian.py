import numpy as np
from typing import Tuple, Optional, Dict, Iterable
from qiskit.quantum_info import SparsePauliOp, Pauli

try:
    from scipy.sparse import lil_matrix, issparse  
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    lil_matrix = None
    issparse = lambda x: False

def _qubo_upper_to_ising_pauli(
    Q: np.ndarray,
    offset: float = 0.0,
    tol: float = 1e-12,
) -> Tuple[list, np.ndarray]:
    # Ensure float64 for numerical stability
    Q = np.asarray(Q, dtype=np.float64)
    n = Q.shape[0]

    pauli_dict: Dict[str, float] = {}

    def add_term(z_indices: Iterable[int], coeff: float):
        if abs(coeff) <= tol or not np.isfinite(coeff):
            return
        s = ["I"] * n
        for idx in z_indices:
            # rightmost char = qubit 0
            s[n - 1 - int(idx)] = "Z"
        key = "".join(s)
        pauli_dict[key] = pauli_dict.get(key, 0.0) + float(coeff)

    const = float(offset)

    # Diagonal
    diag = np.diag(Q)
    nz_diag = np.nonzero(np.abs(diag) > tol)[0]
    for i in nz_diag:
        q = float(diag[i])
        const += q * 0.5
        add_term((i,), -q * 0.5)

    # Off-diagonal upper triangle (i < j)
    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu, ju):
        q = float(Q[i, j])
        if abs(q) <= tol:
            continue
        const += q * 0.25
        add_term((i,), -q * 0.25)
        add_term((j,), -q * 0.25)
        add_term((i, j), q * 0.25)

    # Add constant identity (only if large enough)
    add_term(tuple(), const)

    paulis = list(pauli_dict.keys())
    coeffs = np.array([pauli_dict[p] for p in paulis], dtype=np.float64)

    # Remove any tiny leftovers
    keep = np.abs(coeffs) > tol
    paulis = [p for p, k in zip(paulis, keep) if k]
    coeffs = coeffs[keep]

    return paulis, coeffs


def buildTSPHamiltonian(
    distance_matrix: np.ndarray,
    penalty: Optional[float] = None,
    *,
    max_cities: int = 4,
    clamp_penalty: float = 20.0,
    normalize: bool = False,
    max_coeff: float = 10.0,
    tol: float = 1e-12,
    return_scale: bool = False,
) -> Optional[Tuple[SparsePauliOp, float]]:

    D = np.asarray(distance_matrix, dtype=float)
    if D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix must be square")
    N = int(D.shape[0])

    if N > max_cities:
        raise ValueError(f"TSP Hamiltonian capped at {max_cities} cities for this evaluator. Got N={N}.")

    n_qubits = N * N

    def idx(i, t):
        return i * N + t

    # Use sparse Q for stability and incremental updates if scipy available
    if _HAS_SCIPY and lil_matrix is not None:
        Q = lil_matrix((n_qubits, n_qubits), dtype=np.float64)
        def Q_add(a, b, val):
            Q[a, b] = Q[a, b] + val
    else:
        Q = np.zeros((n_qubits, n_qubits), dtype=np.float64)
        def Q_add(a, b, val):
            Q[a, b] += val

    offset = 0.0

    # Auto penalty suggestion based on distances and N, but keep modest
    max_d = float(np.max(np.abs(D))) if D.size > 0 else 1.0
    suggested_A = max(1.0, max_d) * float(N)  # baseline
    A = float(penalty) if penalty is not None else float(min(suggested_A, clamp_penalty))
    # ensure finite positive
    if not np.isfinite(A) or A <= 0.0:
        A = max(1.0, min(suggested_A, clamp_penalty))

    # --- Cost term ---
    # cost = sum_t sum_{i != j} D[i,j] * x_{i,t} * x_{j,t+1}
    for t in range(N):
        t_next = (t + 1) % N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                qi = idx(i, t)
                qj = idx(j, t_next)
                a, b = (qi, qj) if qi < qj else (qj, qi)
                Q_add(a, b, float(D[i, j]))

    # --- Time-slot constraints: (sum_i x_{i,t} - 1)^2
    for t in range(N):
        offset += A * 1.0
        for i in range(N):
            qi = idx(i, t)
            Q_add(qi, qi, -A)
        for i in range(N):
            for j in range(i + 1, N):
                qi = idx(i, t)
                qj = idx(j, t)
                a, b = (qi, qj) if qi < qj else (qj, qi)
                Q_add(a, b, 2.0 * A)

    # --- City constraints: (sum_t x_{i,t} - 1)^2
    for i in range(N):
        offset += A * 1.0
        for t in range(N):
            qi = idx(i, t)
            Q_add(qi, qi, -A)
        for t in range(N):
            for u in range(t + 1, N):
                qi = idx(i, t)
                qj = idx(i, u)
                a, b = (qi, qj) if qi < qj else (qj, qi)
                Q_add(a, b, 2.0 * A)

    # convert sparse Q (if scipy) to dense upper-triangular numpy for converter convenience
    if _HAS_SCIPY and issparse(Q):
        Q = Q.toarray()

    # Now convert to Pauli strings and coeffs
    paulis, coeffs = _qubo_upper_to_ising_pauli(Q, offset=offset, tol=tol)

    # If normalization requested, compute scale
    scale = 1.0
    if normalize and coeffs.size > 0:
        max_abs = float(np.max(np.abs(coeffs)))
        if max_abs > max_coeff and max_abs > 0:
            scale = max_coeff / max_abs
            coeffs = coeffs * scale

    # Build SparsePauliOp using Pauli objects to be explicit
    # keep right-to-left qubit ordering (Pauli accepts string with qubit0 on right)
    pauli_objs = [Pauli(p) for p in paulis] if paulis else [Pauli("I" * n_qubits)]
    # If we had removed all paulis due to tol, ensure at least identity
    if len(pauli_objs) == 0:
        pauli_objs = [Pauli("I" * n_qubits)]
        coeffs = np.array([0.0], dtype=np.float64)

    H = SparsePauliOp(pauli_objs, coeffs).simplify()

    H_max = (-1.0) * H

    if normalize and return_scale:
        return H_max, scale
    if normalize:
        return H_max  # scale applied inside
    return H_max
