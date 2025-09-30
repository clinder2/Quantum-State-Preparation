from qiskit_algorithms import VQE
from .Builders.base import AnsatzBuilder
from .Problems.base import ProblemSet
from qiskit_algorithms.optimizers import SLSQP
import time
from qiskit.primitives import Estimator

def evaluateBuilder(builder_class:AnsatzBuilder, problems:ProblemSet ): 

    problemSet = problems.getProblemSet()
    results = []
    estimator = Estimator(options={"shots":None}) 
    optimizer = SLSQP(maxiter=100)

    for i, (hamiltonian, exact) in enumerate(problemSet): 
        builder = builder_class(hamiltonian)

        start = time.time()
        circuit = builder.getCircuit() 
        build_time = time.time() - start

        depth = circuit.depth()
        gates = circuit.size()

        vqe = VQE(estimator, circuit, optimizer)
        vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
        vqe_energy = vqe_result.eigenvalue.real
        error = abs(vqe_energy - exact)

        result = {
            "builder": builder_class.__name__,
            "problem_index": i,
            "depth": depth,
            "gates": gates,
            "build_time": build_time,
            "energy_error": error,
            "vqe_energy": vqe_energy,
            "exact_energy": exact
        }

        results.append(result)
    
    return results

