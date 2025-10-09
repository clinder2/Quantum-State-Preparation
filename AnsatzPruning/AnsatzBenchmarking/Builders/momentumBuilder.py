from .base import AnsatzBuilder 
from  MomentumBuilder import MomentumBuilder as mb
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator

class MomentumBuilder(AnsatzBuilder):
    # Ansatz builder with SU2 

    def build(self):
        H = self.hamiltonian
        n_qubits = self.hamiltonian.num_qubits
        observables = [
            *H.paulis, H
        ]
        circuit = QuantumCircuit(n_qubits)
        ansatz = QuantumCircuit(n_qubits)

        for i in range(n_qubits): 
            pName = f"angle{i}" 
            p = Parameter(pName)
            ansatz.rx(p, i)

        paramsList = [1]*n_qubits        
        paramsIndex = [i for i in range(n_qubits)]
        
        self.circuit = mb(paramsList, paramsIndex, ansatz, circuit, observables, StatevectorEstimator(), .9, .99)

        return self.circuit
