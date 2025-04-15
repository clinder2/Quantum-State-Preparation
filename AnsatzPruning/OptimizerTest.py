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
from Hgenerator import *
import heapq
import time

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
    badcircuit = circuit.copy()
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
        badcircuit=badcircuit.compose(tempAnsatz)
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
        #print(x.fun)
        #print(x.x)
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
        #print("params: " + str(params))
        #print(cost_func(params,circuit,hamiltonian,estimator))
        #x = minimize(cost_func, currParams, args=(circuit, H, estimator), method="COBYLA")
    #print(params)
    circuit = circuit.compose(ansatz)
    """ index=0
    for p in circuit.parameters:
        p.assign(p, params[index])
        index=index+1
    print(circuit.parameters) """
    #print(circuit)
    #print(badcircuit)
    x = minimize(cost_func, params, args=(circuit, H, estimator), method="COBYLA")
    #print('final params: ' + str(x.x))
    #print('Cost: ' + str(cost_func(params, circuit, hamiltonian, Estimator())))
    return x, circuit, badcircuit

def naive(circuit):
    print('a')

if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZXZZ", 7),("ZZXX", 1),("IZZI", 5),("XIZZ", 1)]) # Toy hamiltonian
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
    """ final.rx(angle1, 0)
    final.rx(angle2, 1)
    final.rx(angle3, 2)
    final.rx(angle4, 3) """
    #guess = [1 for i in range(0, n)]
    guess=[]
    ans = []
    fields = ['H', 'goal', 'time', 'min', 'diff']
    total_speedup = 0
    total_err = 0
    #final = final.assign_parameters(guess)
    for t in range(0, 1):
        H=makeH(4, 5)
        observables = [
            *H.paulis,H
        ]
        print(H)
        guess=[]
        s = time.time()
        c = LayerOptimizer(guess, circuit,3,final,observables,Estimator())
        e = time.time()
        topt = e-s
        print(e-s)
        s = time.time()
        x = minimize(cost_func, np.ones(len(c[2].parameters)), args=(c[2], H, Estimator()), method="COBYLA")
        e = time.time()
        tnon = e-s
        print(e-s)
        print(c[0].fun)
        #print(c[1])
        print(x.fun)
        a = {}
        a['H'] = H
        a['goal'] = np.sum(H.coeffs)
        a['time'] = (tnon-topt)/tnon
        a['min'] = c[0].fun
        a['diff'] = x.fun - c[0].fun
        total_speedup += (tnon-topt)/tnon
        total_err+=x.fun - c[0].fun
        ans.append(a)
    print(total_speedup/20)
    print(total_err/20)
    """ with open("Pruning-IterativeOpt.csv", "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        #writer.writeheader()
        writer.writerows(ans) """
    """ c[1].draw(output='mpl')
    matplotlib.pyplot.show()
    c[2].draw(output='mpl')
    matplotlib.pyplot.show() """
    #print(cost_func(np.ones(len(c.parameters)), c, observables, Estimator()))

"""
Pruning-Cumalative optimization
H = SparsePauliOp.from_list([("ZIZZ", 7),("ZZII", 1),("IZZI", 5),("IIZZ", 1)])
pruning: time=0.0871131420135498, ans=-11.99999996574813
no pruning: time=0.781792163848877, ans=-13.999993902707512
H = SparsePauliOp.from_list([("ZXZZ", 7),("ZZXX", 1),("IZZI", 5),("XIZZ", 1)]) # Toy hamiltonian
pruning: time=0.1971120834350586, ans=-8.660253570659968
no pruning: time=0.8031327724456787, ans=-8.695101719893552
ave_speedup: 0.7291977804367562
ave_error: -3.966025361768929

Pruning-Iterative optimization
ave_speedup: 0.48609148253809387
ave_error: -2.8509075198672567
"""