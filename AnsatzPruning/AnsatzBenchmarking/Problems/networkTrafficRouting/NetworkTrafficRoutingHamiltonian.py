import networkx as nx
from qiskit.quantum_info import SparsePauliOp, Pauli


def buildNetworkTrafficRoutingHamiltonian(
    graph: nx.Graph,
    paths: list[list[tuple[int, int]]]
) -> SparsePauliOp:
    """
    Build congestion-aware Network Traffic Routing Hamiltonian.

    Cost:
        C(x) = sum_{(i,j)} c_ij * f_ij(x)^2

    where:
        f_ij(x) = sum of path variables using edge (i,j)

    Binary mapping:
        x = (I - Z)/2
    """

    numQubits = len(paths)

    # Map each edge to list of path indices that use it
    edge_to_paths = {}

    for p_idx, path in enumerate(paths):
        for edge in path:
            edge = tuple(sorted(edge))
            if edge not in edge_to_paths:
                edge_to_paths[edge] = []
            edge_to_paths[edge].append(p_idx)

    pauliTerms = []

    for (u, v), path_indices in edge_to_paths.items():

        weight = graph[u][v].get("weight", 1.0)

        # f_ij(x)^2 expansion
        # = sum x_p + 2 sum_{p<q} x_p x_q

        # Linear terms
        for p in path_indices:

            # x_p = (I - Z_p)/2
            pauliTerms.append((Pauli("I" * numQubits), weight / 2))

            pauliString = ["I"] * numQubits
            pauliString[numQubits - 1 - p] = "Z"
            pauliTerms.append(
                (Pauli("".join(pauliString)), -weight / 2)
            )

        # Quadratic terms
        for i in range(len(path_indices)):
            for j in range(i + 1, len(path_indices)):

                p = path_indices[i]
                q = path_indices[j]

                # 2 * x_p x_q
                # x_p x_q = 1/4 (I - Zp - Zq + ZpZq)
                coeff = weight / 2  # because 2 * 1/4 = 1/2

                # Identity
                pauliTerms.append((Pauli("I" * numQubits), coeff))

                # -Zp
                z_p = ["I"] * numQubits
                z_p[numQubits - 1 - p] = "Z"
                pauliTerms.append((Pauli("".join(z_p)), -coeff))

                # -Zq
                z_q = ["I"] * numQubits
                z_q[numQubits - 1 - q] = "Z"
                pauliTerms.append((Pauli("".join(z_q)), -coeff))

                # +ZpZq
                z_pq = ["I"] * numQubits
                z_pq[numQubits - 1 - p] = "Z"
                z_pq[numQubits - 1 - q] = "Z"
                pauliTerms.append((Pauli("".join(z_pq)), coeff))

    if not pauliTerms:
        return SparsePauliOp(Pauli("I"), 0)

    paulis = [p for p, _ in pauliTerms]
    coeffs = [c for _, c in pauliTerms]

    return SparsePauliOp(paulis, coeffs).simplify()
