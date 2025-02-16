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
def LayerOptimizer(params:list, ansatz:QuantumCircuit, layers:int, 
                 circuit:QuantumCircuit, hamiltonian:SparsePauliOp, estimator:Estimator):
    simulator = AerSimulator(method='statevector')
    n = circuit.num_qubits
    newParams = ParameterVector('new', layers*n)
    prev = np.ones(int(math.pow(2,n)))
    prev/=np.linalg.norm(prev)
    prev2=[]
    for p in prev:
        prev2.append(p)
    prev=prev2
    for l in range(0,layers):
        ### Naive layer construction
        naiveLayer = QuantumCircuit(n)
        currParams = []
        for i in range(0,n):
            naiveLayer.ry(newParams[l*n + i], i)
            params.append(1)
            currParams.append(i)
        for i in range(1,n):
            naiveLayer.cx(0,i)
        tempAnsatz = ansatz.compose(naiveLayer)
        tempCircuit = circuit.compose(tempAnsatz)
        accumulator = []
        for i in range(len(params)-n,len(params)):
            heapq.heappush(accumulator, (abs(gradi(i,params,tempCircuit,hamiltonian,estimator)[len(hamiltonian)-1]).item(),i))
        ### 50% pruning rate
        rate=0.7
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
            del currParams[len(currParams)-1]
        ansatz = ansatz.compose(naiveLayer)
        #print(circuit.compose(ansatz))
        c = circuit.compose(ansatz)
        #c.draw(output='mpl')
        #matplotlib.pyplot.show()
        ### temp circuit for optimizing current layer's params
        tempC = QuantumCircuit(n)
        indicies = np.arange(n)
        ind = []
        for x in indicies:
            ind.append(x)
        indicies = ind
        tempC.initialize(prev, indicies) # init to prev layer's output
        tempC = tempC.compose(naiveLayer) # add new layer
        #tempC = tempC.assign_parameters(currParams)
        #tempC.save_statevector(str(l))
        x = minimize(cost_func, currParams, args=(tempC, H, estimator), method="COBYLA")
        index=0
        for p in range(len(currParams)-1, -1, -1):
            params[len(params)-p-1]=x.x[index].item() # add params to main param vector
            index=index+1
        #ansatz.save_statevector(str(l))
        circuit2 = circuit.compose(ansatz)
        circuit2.save_statevector(str(l))
        circfinal = circuit2.assign_parameters(params)
        results = simulator.run(circfinal, shots=1024).result()
        temp = partial_trace(results.data(0)[str(l)], [])
        prev = np.diagonal(temp)
        norm = np.linalg.norm(prev-np.zeros(int(math.pow(2, n))))
        prev=prev/norm # update prev to latest layer's output
        del circuit2.data[len(circuit2.data)-1]
        #print(cost_func(params,circuit,hamiltonian,estimator))
        #x = minimize(cost_func, currParams, args=(circuit, H, estimator), method="COBYLA")
    print(params)
    circuit = circuit.compose(ansatz)
    print(circuit)
    x = minimize(cost_func, params, args=(circuit, H, estimator), method="COBYLA")
    print(x)

if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 7),("ZZII", 1),("IZZI", 5),("IIZZ", 1)]) # Toy hamiltonian
    observables = [
        *H.paulis,H
    ]
    n=4
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
    guess = [1 for i in range(0, n)]
    #final = final.assign_parameters(guess)
    LayerOptimizer(guess, circuit,3,final,observables,Estimator())