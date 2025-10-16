from qiskit_algorithms import VQE
from .Builders.base import AnsatzBuilder
from .Problems.base import ProblemSet
from qiskit_algorithms.optimizers import COBYLA
import time
from qiskit.primitives import StatevectorEstimator
import numpy as np

def evaluateBuilder(builder_class:AnsatzBuilder, problems:ProblemSet ): 
    '''
    Evaluator using StateVectorEstimator and VQE to find ground states 
    '''

    problemSet = problems.getProblemSet()
    results = []
    estimator = StatevectorEstimator() 
    optimizer = COBYLA(maxiter=100)
    for i, (h, exact) in enumerate(problemSet): 
        hamiltonian = -h
        builder = builder_class(hamiltonian)

        start = time.time()
        circuit = builder.getCircuit() 
        build_time = time.time() - start

        depth = circuit.depth()
        gates = circuit.num_parameters
        # Correct VQE initialization and call
        vqe = VQE(estimator=estimator, ansatz=circuit, optimizer=optimizer)
        vqe_result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        vqe_energy = -1 * vqe_result.eigenvalue.real
        error = abs(vqe_energy - exact)

        eigenvalues, _ = np.linalg.eig(h.to_matrix())
        groundState = np.real(np.max(eigenvalues))

        result = {
            "builder": builder_class.__name__,
            "problem_index": i,
            "depth": depth,
            "gates": gates,
            "build_time": build_time,
            "energy_error": error,
            "vqe_energy": vqe_energy,
            "solved_eigenvalue": groundState, 
            "exact_energy": exact
        }

        results.append(result)
    
    return results