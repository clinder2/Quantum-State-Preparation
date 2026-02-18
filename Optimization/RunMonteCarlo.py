import numpy as np
import math
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from MonteCarlo import simulated_annealing, E, cost_func

def find_parameter(y, optimization_method):
    print(f"Target State: {y}")
    n = int(math.log2(len(y)))
    ansatz = RealAmplitudes(n, reps=1)
    parameters = np.ones(ansatz.num_parameters)
    simulator = AerSimulator(method='statevector')
    meas = ClassicalRegister(1, "meas")
    qreg = QuantumRegister(2 * n + 1)
    c = QuantumCircuit(qreg, meas)

    # Prepare the circuit for the swap test
    a = [1]
    b = [n + 1]
    for i in range(2, n + 1):
        a.append(i)
        b.append(n + i)

    final = c.compose(ansatz, a)
    final.initialize(y, b)
    final.save_statevector(label="ans")
    final.h(0)

    # Perform the swap test
    for i in range(1, n + 1):
        final.cswap(0, i, n + i)
    final.h(0)
    final.measure([0], meas)

    # Transpile
    circ = transpile(final, simulator)

    print(f"Running {optimization_method.__name__}...")
    start_energy = E(parameters, circ, simulator)
    print(f"Initial Energy: {start_energy}")

    optimized_params = optimization_method(100, parameters, circ, simulator)

    final_energy = E(optimized_params, circ, simulator)
    print(f"Final Energy: {final_energy}")
    return optimized_params

if __name__ == "__main__":
    from random import randint
    # Generate random state for 2 qubits (length 4)
    randvect = [randint(0, 100) for _ in range(4)]
    norm = np.linalg.norm(randvect)
    randvect = randvect / norm
    
    find_parameter(randvect, simulated_annealing)
