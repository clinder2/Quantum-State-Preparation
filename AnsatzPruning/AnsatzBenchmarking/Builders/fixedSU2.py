from .base import AnsatzBuilder 
from qiskit.circuit.library import efficient_su2

class FixedSU2Builder(AnsatzBuilder):
    # Ansatz builder with SU2 

    def build(self):
        n_qubits = self.hamiltonian.num_qubits
        self.circuit = efficient_su2(n_qubits, reps=2)
        return self.circuit
