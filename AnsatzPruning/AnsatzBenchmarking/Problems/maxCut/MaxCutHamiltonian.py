import networkx as nx 
from qiskit.quantum_info import SparsePauliOp, Pauli



def buildMaxCutHamiltonian(graph:nx.graph) -> SparsePauliOp:
    '''
    Build graph hamiltonian based on the following equation \n
    H = 1/2 ( Sum( w_ij (I - ZiZj) ) )
    '''

    numQubits = graph.number_of_nodes()

    #reverse mapping from node to qubit index
    if not all(isinstance(n, int) and 0 <= n < numQubits for n in graph.nodes):
        nodeToQubit = {node: i for i, node in enumerate(graph.nodes)}
    else:
        nodeToQubit = {i: i for i in range(numQubits)}

    pauliTerms = [] 

    for u, v, data in graph.edges(data=True): 
        weight = data.get('weight', 1.0) 
        i = nodeToQubit[u]
        j = nodeToQubit[v]

        pauliTerms.append((Pauli('I' * numQubits), weight / 2))

        pauliStrings = ['I'] * numQubits

        pauliStrings[numQubits - 1 - i] = 'Z'
        pauliStrings[numQubits - 1 - j] = 'Z'
        pauliString = "".join(pauliStrings)
        pauliTerms.append((Pauli(pauliString) , -weight/2))

    # empty graph 
    if not pauliTerms:
        return SparsePauliOp(Pauli('I' * numQubits), 0)

    paulis = [p for p, _ in pauliTerms]
    coeffs = [c for _, c in pauliTerms]
    H = SparsePauliOp(paulis, coeffs).simplify()
    return H


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 0, weight=1.0)
    G.add_edge(0, 2, weight=2.0) 

    print(f"--- Graph: {G.nodes} nodes, {G.edges} edges ---")
    
    hamiltonian = buildMaxCutHamiltonian(G)

    print("\n--- Max-Cut Hamiltonian (SparsePauliOp) ---")
    print("Format: Coefficient * Pauli String (right-to-left order, Qubit 0 is rightmost)")
    print(hamiltonian)

    '''
    Expected Output: 
    Edges , Weight | Contributed Term in H | Pauli String (Excluding I) 
    (0, 1), 1.0	    , 0.5I-0.5Z0Z1         , IIZZ
    (1, 2), 1.0     , 0.5I-0.5Z1Z2         , IZZI 
    (2, 3),	1.0	    , 0.5I-0.5Z2Z3         , ZZII
    (3, 0),	1.0	    , 0.5I-0.5Z3Z0         , ZIIZ
    (0, 2),	2.0	    , 1.0I-1.0Z0Z2         , IZIZ

    Coefficient of all pauli string except following is - 0.5 
    IIII = 3 
    IZIZ = -1 
    '''
    