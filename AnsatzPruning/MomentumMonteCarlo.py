import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_aer.aerprovider import AerSimulator
from Optimization import MonteCarlo
from AnsatzPruning import MomentumBuilder
from AnsatzPruning.Utilities import cost_func
from qiskit.circuit import Parameter


def momentum_monte_carlo(params:list, inds:list, ansatz:QuantumCircuit,
                         circuit:QuantumCircuit, hamiltonian:SparsePauliOp,
                         estimator:Estimator, beta1:float, beta2:float,
                         iters:int=2, optimization_runs:int=100, method:str='hill_climbing'):
    """
    Ansatz optimization pipeline that first runs MomentumBuilder and then optimizes 
    the parameters using Monte Carlo optimization method.
    """
    observables = [*hamiltonian.paulis, hamiltonian]
    
    # Run MomentumBuilder
    # print("Running MomentumBuilder")
    optimized_ansatz = MomentumBuilder.MomentumBuilder(
        params, inds, ansatz, circuit, observables, estimator,
        beta1, beta2, iters
    )
    
    # Extract parameters from ansatz
    num_params = len(optimized_ansatz.parameters)
    initial_params = np.ones(num_params)
    print("Cost after MomentumBuilder: ", cost_func(initial_params, optimized_ansatz, observables, estimator))
    
    # Run Monte Carlo optimization
    # print(f"Running Monte Carlo (stochastic hill climbing)")
    simulator = AerSimulator(method='statevector')
    initial_params = initial_params.copy()

    # Stochastic hill climbing
    # optimized_params = MonteCarlo.stochastic_hill_climbing(
    #     optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    # )

    # Differential Evolution
    # optimized_params = MonteCarlo.diff_evolution(
    #     optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    # )

    # Global best PSO
    # optimized_params = MonteCarlo.gbest_pso(
    #     optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    # )

    # Simulated Annealing
    optimized_params = MonteCarlo.simulated_annealing(
        optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    )

    print("Cost after MomentumBuilder and Monte Carlo: ", cost_func(optimized_params, optimized_ansatz, observables, estimator))
    
    return optimized_ansatz, optimized_params


if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 1), ("ZZII", 3), ("IZZI", 1), ("IIZZ", 1)])
    
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    
    circuit = QuantumCircuit(4)
    ansatz = QuantumCircuit(4)

    ansatz.rx(angle1, 0)
    ansatz.rx(angle2, 1)
    ansatz.rx(angle3, 2)
    ansatz.rx(angle4, 3)

    # ansatz.draw(output="mpl")

    # Run MomentumBuilder for comparison
    # observables = [*H.paulis,H]
    # final_circuit_MB = MomentumBuilder.MomentumBuilder([1,1,1,1], [0,1,2,3], ansatz, circuit, observables, Estimator(), 0.9, 0.99)
    # final_circuit_MB.draw(output="mpl")

    final_circuit_MMC, final_params = momentum_monte_carlo([1,1,1,1], [0,1,2,3], ansatz, circuit, H, Estimator(),
        beta1=0.9, beta2=0.99, iters=2, optimization_runs=100, method='hill_climbing'
    )
    final_circuit_MMC.draw(output="mpl")
    
    # print(f"Optimization complete. Final parameters: {final_params}")
    plt.show()

