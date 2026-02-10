import numpy as np
from qiskit.quantum_info import SparsePauliOp

# Import your function here
from KnapsackHamiltonian import buildKnapsackHamiltonian

def classical_knapsack_cost(bitstring, values, weights, W, P):
    """C(x) = -sum v_i x_i + P( sum w_i x_i - W )^2"""
    x = np.array([int(b) for b in bitstring], dtype=int)
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    return -np.dot(v, x) + P * (np.dot(w, x) - W) ** 2


def pauli_eigenvalue_on_bitstring(pauli_str, bitstring):
    """
    Compute eigenvalue of a Pauli string consisting only of I and Z
    on computational basis state |bitstring>.
    
    Qiskit convention: rightmost char is qubit 0.
    bitstring here we treat as "qubit 0 is leftmost"? We'll avoid ambiguity by
    defining: bitstring[i] corresponds to qubit i.
    """
    # Ensure pauli_str length == n
    n = len(bitstring)
    assert len(pauli_str) == n

    eig = 1.0
    for i in range(n):
        # Qiskit Pauli strings are right-to-left: char index (n-1-i) is qubit i
        ch = pauli_str[n - 1 - i]
        if ch == "I":
            continue
        if ch != "Z":
            raise ValueError("This test only supports I/Z Hamiltonians.")
        # Z eigenvalue: +1 on |0>, -1 on |1>
        eig *= (1.0 if bitstring[i] == "0" else -1.0)
    return eig


def energy_from_sparsepauliop(H: SparsePauliOp, bitstring):
    """
    For a diagonal (I/Z-only) Hamiltonian, energy on |bitstring>
    is sum_k coeff_k * eigenvalue(Pauli_k).
    """
    energy = 0.0
    for pauli, coeff in zip(H.paulis, H.coeffs):
        pstr = pauli.to_label()  # e.g., "IZZI"
        energy += float(np.real(coeff)) * pauli_eigenvalue_on_bitstring(pstr, bitstring)
    return energy


def test_knapsack_hamiltonian_matches_classical():
    # Small instance (n=3) so we can brute force all 2^n states
    values = [6, 10, 12]
    weights = [1, 2, 3]
    W = 3
    P = 20.0

    H = buildKnapsackHamiltonian(values, weights, W, P)

    n = len(values)

    # Structural test: only I and Z should appear
    for pauli in H.paulis:
        label = pauli.to_label()
        assert set(label).issubset({"I", "Z"}), f"Found non I/Z Pauli term: {label}"

    # Energy equality test across all bitstrings
    for x_int in range(2**n):
        bitstring = format(x_int, f"0{n}b")  # qubit order: bitstring[i] is qubit i
        E_quantum = energy_from_sparsepauliop(H, bitstring)
        E_classical = classical_knapsack_cost(bitstring, values, weights, W, P)

        # for debugging
        if bitstring == "000":
            print("\n--- DEBUG FOR |000> ---")
            print("Quantum energy:", E_quantum)
            print("Classical energy:", E_classical)
            print("Hamiltonian:")
            print(H)
            print("----------------------\n")

        # Should match up to tiny numerical tolerance
        assert np.isclose(E_quantum, E_classical, atol=1e-8), (
            f"Mismatch for |{bitstring}>: "
            f"H={E_quantum}, C={E_classical}"
        )

if __name__ == "__main__":
    test_knapsack_hamiltonian_matches_classical()
    print("Knapsack Hamiltonian test PASSED")