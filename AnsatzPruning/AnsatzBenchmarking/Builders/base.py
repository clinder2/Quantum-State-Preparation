from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class AnsatzBuilder(ABC):
    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        self.circuit = None

    @abstractmethod
    def build(self):
        """Construct and return the ansatz QuantumCircuit."""
        pass

    def get_circuit(self) -> QuantumCircuit:
        if self.circuit is None:
            self.circuit = self.build()
        return self.circuit