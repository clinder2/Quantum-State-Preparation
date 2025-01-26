import numpy as np
import math
import time
import pandas as pd
from random import randint
import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
import csv
from scipy.optimize import minimize
from qiskit_aer.aerprovider import AerSimulator
#from qiskit_aer import Aer, aerprovider
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import *
from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import *

def find_parameter(y):
    start = time.time()
    output = {
        "y": y,
        "fidelity": 0.0,
        "mse": 0.0,
        "parameters": None,
        'time': None
    }
    estimator = Estimator() # Statevector estimator
    simulator = AerSimulator(method='statevector')
    H = SparsePauliOp.from_list([("ZIIZ", 1),("ZZII", 3),("IZZI", 1),("IIZZ", 1)])
    observables = [
        *H.paulis,H
    ]
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    final = QuantumCircuit(4)
    #final.h(0)
    #final.rx(angle1, 0)
    final.rx(angle2, 1)
    layer = QuantumCircuit(4)
    layer.cx(0,1)
    layer.cx(1,2)
    layer.cx(0,3)
    #layer.rx(angle3, 2)
    layer.rx(angle4, 3)
    final = final.compose(layer)
    parameters = np.ones(final.num_parameters) # Initial guess
    delta = np.ones(len(parameters))
    delta[0] = math.pi/2
    #ansatz = QAOAAnsatz(H, reps=2)
    #circ = transpile(final, estimator) # transpile so compatible with simulator
    #final = final.assign_parameters(parameters)
    #job = estimator.run([(final, observables)])
    print(final)
    #print(final.num_parameters)
    #final = final.assign_parameters(parameters)
    #print(simulator.run(final).result())
    t = cost_func(parameters, final, observables, estimator)
    for (obs,val) in zip(observables,t):
        if obs is H:
            print("Total: " + str(val))
        else:
            print(str(obs) + ": " + str(val))
    print(cost_func(parameters, final, observables, estimator))
    for i in range(0, len(parameters)):
        print(gradi(i, parameters, final, observables, estimator))
    min = minimize(cost_func, parameters, args=(final, H, estimator), method="COBYLA")
    print(min.x)
    print(cost_func(min.x, final, observables, estimator))
    return output

def cost_func(params, circuit, hamiltonian, estimator):
    pub = (circuit, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    #cost = cost[len(cost)-1]
    return cost

def gradi(i, params, circuit, hamiltonian, estimator):
    delta = np.zeros(len(params))
    delta[i] = math.pi/2
    costp = cost_func(params+delta, circuit, hamiltonian, estimator)
    costm = cost_func(params-delta, circuit, hamiltonian, estimator)
    #print(params+delta)
    return (costp-costm)/2

if __name__ == "__main__":
    fields = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'fidelity', 'mse', 'parameters', 'time']
    ans = []
    times = 0
    data = pd.read_csv("Datasets/Ansatze/3Q/3QAnsatz-RRCXCXCXHHHRR.csv", usecols=['y1','y2','y3','y4','y5','y6','y7','y8'])
    data = pd.DataFrame(data)
    data.dropna(how='all', inplace=True)
    for i in range(0, 1):
        print(i)
        randvect = [randint(-10000,10000) for p in range(0,16)]
        #randvect = [data.at[i, 'y1'], data.at[i, 'y2'], data.at[i, 'y3'], data.at[i, 'y4']]
        norm = np.linalg.norm(randvect)
        randvect = randvect/norm
        ans1 = find_parameter(randvect)
        ans.append(ans1)
        #times.append(ans1['time'])
        #times = times + ans1['time']
        #print(ans1)
