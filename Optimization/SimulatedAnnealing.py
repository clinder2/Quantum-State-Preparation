import numpy as np
import math
from random import randint
import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
from scipy.optimize import minimize
from qiskit_aer.aerprovider import AerSimulator
from qiskit_aer import Aer, aerprovider
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import HGate, ZGate
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import *

'''
Problem 3. Reversely find a parameter.

Input y is desired prepared quantum state. (L2 Normalized |y|^2 = 1)
Find parameters of variational quantum circuit "RealAmplitude" that can prepare y.
'''

def find_parameter(y):
    n = int(math.log2(len(y)))
    #print(n)
    ansatz = RealAmplitudes(n,reps=1)
    parameters = np.ones(ansatz.num_parameters) # Initial guess
    simulator = AerSimulator(method='statevector')
    meas = ClassicalRegister(1, "meas")
    qreg = QuantumRegister(2*n + 1)
    c = QuantumCircuit(qreg, meas) # Assemble circuit
    a = [1]
    b = [n+1]
    for i in range(2, n+1):
        a.append(i)
        b.append(n+i)
    #print(a)
    #print(b)
    final = c.compose(ansatz, a)
    final.initialize(y, b) # initialize ancillas to input y
    final.save_statevector(label="ans")
    final.h(0) # swap test
    for i in range(1, n+1):
        final.cswap(0,i,n+i) # swap test
    final.h(0) # swap test
    final.measure([0],meas)
    circ = transpile(final, simulator) # transpile so compatible with simulator
    circfinal = circ.assign_parameters(parameters)
    results = simulator.run(circfinal).result()
    print("Before optimization: " + str(results.get_counts()))
    Energy = E(parameters, circ, simulator) # Optimization loop
    anneal = SA(100, parameters, circ, simulator)
    print(Energy)
    #print(parameters)
    circfinal = circ.assign_parameters(anneal)
    results = simulator.run(circfinal).result()
    counts = results.get_counts()
    if '1' in counts:
        b = counts['1']
    else:
        b = 0
    s = (1 - (2/1024)*b) # P0 - P1 (|<q1|q3>|^2|<q2|q4>|^2)
    #print(circfinal.decompose())
    circfinal.draw(output="mpl")
    matplotlib.pyplot.show()
    print("After optimization: " + str(results.get_counts()))
    print(s)
    temp = partial_trace(results.data(0)['ans'], [0,3,4])
    partial = np.diagonal(temp)
    print(partial)
    temp = partial_trace(results.data(0)['ans'], [0,1,2])
    partial2 = np.diagonal(temp)
    print(partial2)
    norm = np.linalg.norm(partial-partial2)
    print(norm)
    return parameters

def cost_func(params, ansatz, simulator):
    circfinal = ansatz.assign_parameters(params)
    results = simulator.run(circfinal, shots=1024).result()
    counts = results.get_counts()
    if '1' in counts:
        b = counts['1']
    else:
        b = 0
    s = -1*(1-((2/1024)*b))
    return s

def E(params, ansatz, simulator):
    circfinal = ansatz.assign_parameters(params)
    results = simulator.run(circfinal, shots=20).result()
    temp = partial_trace(results.data(0)['ans'], [0,3,4])
    partial = np.diagonal(temp)
    #print(partial)
    temp = partial_trace(results.data(0)['ans'], [0,1,2])
    partial2 = np.diagonal(temp)
    #print(partial2)
    norm = np.linalg.norm(partial-partial2)
    #print("energy: " + str(norm))
    return norm

def SA(runs, params, ansatz, simulator):
    #deltaETest = 0
    # average absolute value delta_E: 0.01-0.03
    # acceptance at beginning: 0.99%
    # acceptance at end: 0.00001%
    # temperature: 1.9-0.0017
    B = E(params, ansatz, simulator)
    theta_vect = params
    prev_E = B
    def T(t):
        c = 0.02
        a = 0.01
        temperature = c/(a + math.log(t))
        return temperature
    for t in range(1, runs):    
        delta = np.random.normal(0, .1, 4)
        params_new = params + delta
        E_new = E(params_new, ansatz, simulator)
        delta_E = E_new - prev_E
        #deltaETest += delta_E
        print("prev_E, " + str(prev_E) + ", E_new, " + str(E_new) + ", delta_E: " + str(delta_E))
        if delta_E <= 0:
            params = params_new
            prev_E = E_new
        else:
            h = math.pow(math.e, -1 * delta_E/T(t))
            U = np.random.normal(.5, .5, 1)
            print("h: " + str(h) + ", U: " + str(U) + ", t: " + str(t) + ", T(): " + str(T(t)))
            if U < h:
                params = params_new
                prev_E = E_new
    #print("delta: " + str(deltaETest/runs))
    return params

if __name__ == "__main__":
    randvect = [randint(0,100) for p in range(0,4)]
    norm = np.linalg.norm(randvect)
    randvect = randvect/norm
    print(randvect)
    find_parameter(randvect)
