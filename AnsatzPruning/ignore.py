from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

# Define a simple Hamiltonian, e.g. Z ⊗ Z
H = SparsePauliOp.from_list([("ZZ", 1.0)])

# Ansatz circuit
n_qubits = 2
ansatz = EfficientSU2(n_qubits, reps=2)

# Estimator + Optimizer
estimator = StatevectorEstimator()
optimizer = SLSQP(maxiter=100)

# VQE setup
vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

# Run the algorithm
result = vqe.compute_minimum_eigenvalue(operator=H)
print("Computed ground state energy:", result.eigenvalue.real)
