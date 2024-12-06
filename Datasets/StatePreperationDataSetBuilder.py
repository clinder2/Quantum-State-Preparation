import numpy as np
import math
from random import randint
import time
import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
import csv
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
DataSet builder for 2 qubit input, 4 optimization loops
Average time for 100 inputs: 0.40077924489974975
'''

def find_parameter(y):
    output = {
        "y1": y[0],
        "y2": y[1],
        "y3": y[2],
        "y4": y[3],
        "fidelity": 0.0,
        "mse": 0.0,
        "parameters": None
    }
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
    #print("Before optimization: " + str(results.get_counts()))
    for i in range(1,5): # Optimization loop
        #print(parameters)
        min = minimize(cost_func, parameters, args=(circ, simulator), method="COBYLA")
        parameters = min.x
    #print(min)
    parameters = min.x
    #print(parameters)
    circfinal = circ.assign_parameters(parameters)
    results = simulator.run(circfinal).result()
    counts = results.get_counts()
    if '1' in counts:
        b = counts['1']
    else:
        b = 0
    s = (1 - (2/1024)*b) # P0 - P1 (|<q1|q3>|^2|<q2|q4>|^2)
    #circfinal.draw(output="mpl")
    #matplotlib.pyplot.show()
    #print("After optimization: " + str(results.get_counts()))
    #print(s)
    temp = partial_trace(results.data(0)['ans'], [0,3,4])
    partial = np.diagonal(temp)
    temp = partial_trace(results.data(0)['ans'], [0,1,2])
    partial2 = np.diagonal(temp)
    norm = np.linalg.norm(partial-partial2)
    #print(norm)
    output["fidelity"] = s
    output["parameters"] = parameters
    output["mse"] = norm
    return output

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

if __name__ == "__main__":
    fields = ['y1', 'y2', 'y3', 'y4', 'fidelity', 'mse', 'parameters']
    ans = []
    avertime = 0
    for i in range(1, 101):
        start = time.time()
        randvect = [randint(0,100) for p in range(0,4)]
        norm = np.linalg.norm(randvect)
        randvect = randvect/norm
        a = find_parameter(randvect)
        ans.append(a)
        end = time.time()
        total = end - start
        avertime = avertime + total
    print(avertime/100)
    with open("Testbuild.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ans)

