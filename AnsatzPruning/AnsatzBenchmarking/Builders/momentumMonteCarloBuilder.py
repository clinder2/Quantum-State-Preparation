from .base import AnsatzBuilder
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator

from MomentumMonteCarlo import momentum_monte_carlo as mb

class MonteCarloMomentumBuilder(AnsatzBuilder):
    def build(self):
        H = self.hamiltonian
        n_qubits = H.num_qubits
        
        circuit = QuantumCircuit(n_qubits)
        ansatz = QuantumCircuit(n_qubits)

        for i in range(n_qubits):
            pName = f"angle{i}"
            p = Parameter(pName)
            ansatz.rx(p, i)

        paramsList = [1] * n_qubits
        paramsIndex = [i for i in range(n_qubits)]

        # Call the Monte Carlo pipeline
        optimized_ansatz, optimized_params = mb(
            params=paramsList,
            inds=paramsIndex,
            ansatz=ansatz,
            circuit=circuit,
            hamiltonian=H,
            estimator=StatevectorEstimator(),
            beta1=0.9,
            beta2=0.99,
            iters=3,                
            optimization_runs=500
        )

        self.circuit = optimized_ansatz
        self.optimized_params = np.array(optimized_params, dtype=float).tolist()

        return self.circuit
