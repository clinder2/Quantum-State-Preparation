from .base import AnsatzBuilder 
from qiskit.circuit.library import EfficientSU2

class FixedSU2Builder(AnsatzBuilder):
    # Ansatz builder with SU2 

    def build(self):
        n_qubits = self.hamiltonian.num_qubits
        self.circuit = EfficientSU2(n_qubits)
        return self.circuit
