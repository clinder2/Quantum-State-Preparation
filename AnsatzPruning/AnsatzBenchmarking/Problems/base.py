from abc import ABC, abstractmethod
from qiskit.quantum_info import SparsePauliOp

class ProblemSet(ABC): 
    def __init__(self):
        self.problemSets = None
        pass 

    @abstractmethod
    def createProblemSets(self) -> list[ tuple[SparsePauliOp, float] ]: 
        '''Return a list of problems with Hamiltonian Expected Answer'''
        pass 

    def getProblemSet(self): 
        if self.problemSets is None: 
            self.problemSets = self.createProblemSets() 

        return self.problemSets
        
