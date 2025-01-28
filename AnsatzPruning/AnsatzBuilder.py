import numpy as np
import math
import pandas as pd
from random import randint
import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
from scipy.optimize import minimize
from qiskit_aer.aerprovider import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import *
from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter, ParameterVector
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import *
from Utilities import *
import heapq

"""
Function to naively add fully parametrized, maximally entangled layer layers times
Calculates gradients for each layer, removes smallest magnitude RY gates with given pruning rate
"""
def NaiveBuilder(params:list, ansatz:QuantumCircuit, layers:int, 
                 circuit:QuantumCircuit, hamiltonian:SparsePauliOp, estimator:Estimator):
    n = circuit.num_qubits
    newParams = ParameterVector('new', layers*n)
    for l in range(0,layers):
        ### Naive layer construction
        naiveLayer = QuantumCircuit(n)
        for i in range(0,n):
            naiveLayer.ry(newParams[l*n + i], i)
            params.append(1)
        for i in range(1,n):
            naiveLayer.cx(0,i)
        tempAnsatz = ansatz.compose(naiveLayer)
        tempCircuit = circuit.compose(tempAnsatz)
        accumulator = []
        for i in range(len(params)-n,len(params)):
            #print(gradi(i,params,tempCircuit,hamiltonian,estimator))
            heapq.heappush(accumulator, (abs(gradi(i,params,tempCircuit,hamiltonian,estimator)[len(hamiltonian)-1]).item(),i))
        ### 50% pruning rate
        rate=0.5
        bound = math.floor(rate*n)
        remove = []
        for i in range(0,bound):
            index = heapq.heappop(accumulator)[1]
            heapq.heappush(remove,index%n)
        i = 0
        while len(remove) > 0:
            del naiveLayer.data[heapq.heappop(remove)-i]
            i = i + 1
            del params[len(params)-1]
        ansatz = ansatz.compose(naiveLayer)
        print(circuit.compose(ansatz))
        #print(cost_func(params,circuit,hamiltonian,estimator))
    circuit = circuit.compose(ansatz)
    x = minimize(cost_func, params, args=(circuit, H, estimator), method="COBYLA")
    print(x)

if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 1),("ZZII", 3),("IZZI", 1),("IIZZ", 1)]) # Toy hamiltonian
    observables = [
        *H.paulis,H
    ]
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    circuit = QuantumCircuit(4)
    final = QuantumCircuit(4)
    final.rx(angle1, 0)
    final.rx(angle2, 1)
    final.rx(angle3, 2)
    final.rx(angle4, 3)
    NaiveBuilder([1,1,1,1], circuit,3,final,observables,Estimator())