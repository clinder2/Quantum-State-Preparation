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

def test_set_cover_two_phased():
    """
    Test the two-phased approach (MomentumBuilder + Monte Carlo) on Set Cover.
    """
    print("=" * 60)
    print("Testing Two-Phased Approach (Momentum + MC) on Set Cover")
    print("=" * 60)
    
    # Define Set Cover instance (same as previous test)
    universe = ['A', 'B', 'C']
    subsets = [{'A', 'B'}, {'C'}, {'A'}, {'B', 'C'}]
    
    # Generate Hamiltonian
    H = get_subset_Hamiltonian(universe, subsets)
    print(f"\nSet Cover Hamiltonian:")
    print(f"  Number of qubits: {H.num_qubits}")
    
    matrix = H.to_matrix()
    min_energy = np.real(np.linalg.eigvalsh(matrix)[0])
    print(f"  Ground state energy (classical): {min_energy:.6f}")
    
    # Setup initial ansatz
    num_qubits = 4
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    
    circuit = QuantumCircuit(num_qubits)
    ansatz = QuantumCircuit(num_qubits)
    ansatz.rx(angle1, 0)
    ansatz.rx(angle2, 1)
    ansatz.rx(angle3, 2)
    ansatz.rx(angle4, 3)
    
    params = [1, 1, 1, 1]
    inds = [0, 1, 2, 3]
    estimator = Estimator()
    
    print("\n" + "=" * 60)
    print("Running Two-Phased Optimization")
    print("=" * 60)
    
    # optimization_runs=200 for better convergence
    optimized_ansatz, optimized_params = MomentumMonteCarlo.momentum_monte_carlo(
        params, inds, ansatz, circuit, H, estimator,
        beta1=0.9, beta2=0.99, iters=3, optimization_runs=200, method='simulated_annealing'
    )
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    # Evaluate final energy using statevector
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
    top_states = np.argsort(probabilities)[::-1][:3]
    
    print("\nTop 3 most probable states:")
    for state in top_states:
        prob = probabilities[state]
        bitstring = bin(state)[2:].zfill(num_qubits)
        print(f"  |{bitstring}> : {prob:.4f}")
        
        chosen_subsets = [i for i in range(num_qubits) if bitstring[num_qubits-1-i] == '1']
        covered_elements = set()
        for idx in chosen_subsets:
            covered_elements.update(subsets[idx])
        
        if covered_elements == set(universe):
            print(f"    ✓ Valid Exact Cover! (Subsets: {chosen_subsets})")

if __name__ == "__main__":
    test_set_cover_two_phased()
