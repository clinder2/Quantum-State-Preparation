import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AnsatzPruning'))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_aer import AerSimulator
from SetCoverHamiltonian import get_subset_Hamiltonian
from AnsatzPruning import MomentumMonteCarlo

def test_set_cover_hard():
    """
    Test the two-phased approach on a harder 8-qubit Set Cover instance.
    """
    print("=" * 60)
    print("Testing Harder 8-Qubit Set Cover Instance")
    print("=" * 60)
    
    # Universe: {1, 2, 3, 4, 5, 6}
    universe = [1, 2, 3, 4, 5, 6]
    
    # Subsets (8 subsets total)
    subsets = [
        {1, 2},       # S0
        {3, 4},       # S1
        {5, 6},       # S2
        {1, 3, 5},    # S3
        {2, 4, 6},    # S4
        {1, 4},       # S5
        {2, 5},       # S6
        {3, 6}        # S7
    ]
    
    # Generate Hamiltonian
    H = get_subset_Hamiltonian(universe, subsets)
    num_qubits = len(subsets)
    print(f"\nSet Cover Hamiltonian:")
    print(f"  Number of qubits: {num_qubits}")
    
    # Brute force classical ground state energy
    matrix = H.to_matrix()
    diagonal = np.real(np.diag(matrix))
    min_energy = np.min(diagonal)
    print(f"  Ground state energy (classical): {min_energy:.6f}")
    
    valid_solutions = np.where(np.isclose(diagonal, min_energy))[0]
    print(f"  Number of classical solutions: {len(valid_solutions)}")
    for sol in valid_solutions:
        bitstring = bin(sol)[2:].zfill(num_qubits)
        print(f"    - Solution |{bitstring}> (int {sol})")

    # Setup initial ansatz
    # Create 8 parameters for 8 qubits
    params_symbols = [Parameter(f"a{i}") for i in range(num_qubits)]
    
    circuit = QuantumCircuit(num_qubits)
    ansatz = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        ansatz.rx(params_symbols[i], i)
    
    initial_params_values = [1.0] * num_qubits
    indices = list(range(num_qubits))
    estimator = Estimator()
    
    print("\n" + "=" * 60)
    print("Running Two-Phased Optimization (Hard Instance)")
    print("=" * 60)
    
    # Increased optimization runs for the harder problem
    optimized_ansatz, optimized_params = MomentumMonteCarlo.momentum_monte_carlo(
        initial_params_values, indices, ansatz, circuit, H, estimator,
        beta1=0.9, beta2=0.99, iters=3, optimization_runs=500, method='simulated_annealing'
    )
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    simulator = AerSimulator(method='statevector')
    bound_circuit = optimized_ansatz.assign_parameters(optimized_params)
    bound_circuit.save_statevector()
    
    from qiskit import transpile
    transpiled = transpile(bound_circuit, simulator)
    result = simulator.run(transpiled).result()
    statevector = result.get_statevector()
    
    final_energy = np.real(statevector.expectation_value(H))
    print(f"\nFinal energy: {final_energy:.6f}")
    print(f"Ground state energy: {min_energy:.6f}")
    print(f"Energy gap: {final_energy - min_energy:.6f}")
    
    # State Analysis
    probabilities = np.abs(statevector.data) ** 2
    top_states = np.argsort(probabilities)[::-1][:10]
    
    print("\nTop 10 most probable states:")
    for state in top_states:
        prob = probabilities[state]
        if prob < 0.001: continue
        
        bitstring = bin(state)[2:].zfill(num_qubits)
        is_valid = state in valid_solutions
        valid_marker = "✓ VALID EXACT COVER" if is_valid else "x"
        
        print(f"  |{bitstring}> : {prob:.4f} {valid_marker}")
        if is_valid:
            chosen = [i for i in range(num_qubits) if bitstring[num_qubits-1-i] == '1']
            print(f"    Subsets: {chosen}")

if __name__ == "__main__":
    test_set_cover_hard()
