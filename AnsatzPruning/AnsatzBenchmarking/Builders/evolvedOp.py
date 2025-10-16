import numpy as np
from .base import AnsatzBuilder 
from qiskit.circuit.library import EvolvedOperatorAnsatz as EOA 
from qiskit.quantum_info import SparsePauliOp

class EvolvedOperatorBuilder(AnsatzBuilder):
    # Ansatz builder with SU2 

    def build(self):
        pauli_list = self.hamiltonian.paulis
        coeffs = np.real(self.hamiltonian.coeffs)

        individual_operators = []
        for i in range(len(pauli_list)):
            pauli_term = pauli_list[i]
            coeff_term = coeffs[i]
            individual_operators.append(SparsePauliOp(pauli_term, coeff_term))
    
        self.circuit = EOA(individual_operators)

        return self.circuit
