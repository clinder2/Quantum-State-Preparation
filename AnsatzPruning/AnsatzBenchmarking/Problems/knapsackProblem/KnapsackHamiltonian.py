from qiskit.quantum_info import SparsePauliOp, Pauli

def buildKnapsackHamiltonian(values, weights, W, P):
    """
    values[i]  = v_i
    weights[i] = w_i
    W          = capacity
    P          = penalty strength
    """
    n = len(values)
    pauli_terms = []

    # ---------- VALUE TERM ----------
    for i, v in enumerate(values):
        # constant part
        pauli_terms.append((Pauli("I" * n), -v / 2))

        # Z_i part
        p = ["I"] * n
        p[n - 1 - i] = "Z"
        pauli_terms.append((Pauli("".join(p)), v / 2))

        # ---------- PENALTY: S^2 ----------
    for i in range(n):
        wi = weights[i]

        # w_i^2 / 4 * I
        pauli_terms.append((Pauli("I" * n), P * wi * wi / 2))

        # - w_i^2 / 2 * Z_i
        p = ["I"] * n
        p[n - 1 - i] = "Z"
        pauli_terms.append((Pauli("".join(p)), -P * wi * wi / 2))

        # ---------- PENALTY: cross terms ----------
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = weights[i], weights[j]

            # I term
            pauli_terms.append((Pauli("I" * n), P * wi * wj / 2))

            # -Z_i
            p = ["I"] * n
            p[n - 1 - i] = "Z"
            pauli_terms.append((Pauli("".join(p)), -P * wi * wj / 2))

            # -Z_j
            p = ["I"] * n
            p[n - 1 - j] = "Z"
            pauli_terms.append((Pauli("".join(p)), -P * wi * wj / 2))

            # +Z_i Z_j
            p = ["I"] * n
            p[n - 1 - i] = "Z"
            p[n - 1 - j] = "Z"
            pauli_terms.append((Pauli("".join(p)), P * wi * wj / 2))

        # ---------- -2 W S ----------
    for i, wi in enumerate(weights):
        pauli_terms.append((Pauli("I" * n), -P * W * wi))
        p = ["I"] * n
        p[n - 1 - i] = "Z"
        pauli_terms.append((Pauli("".join(p)), P * W * wi))

        # ---------- + W^2 ----------
    pauli_terms.append((Pauli("I" * n), P * W * W))

    paulis, coeffs = zip(*pauli_terms)
    return SparsePauliOp(paulis, coeffs).simplify()
