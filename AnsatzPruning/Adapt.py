#from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
#from qiskit_algorithms.optimizers import SLSQP
import numpy as np
import math
import pandas as pd
from random import randint
import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QGA.LayerGA import *
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
#from qiskit_nature import *
#from qiskit_nature.second_q.operators.commutators import commutator
from Hgenerator import *
#import qiskit.opflow as OP
import heapq
import time
#from qiskit.aqua.operators import PrimitiveOp
#from qiskit.opflow import PrimitiveOp

"""
Function to build ansatz from operator library
Calculates gradients of each layer, adds layers with largest gradients, removes smallest magnitude RY gates with given pruning rate
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
    L=[] #library
    for i in range(5*layers):
        lay=randomLayer(n)
        L.append(lay)
    Lheap = []
    i=0
    for term in L:
        curr=0
        s=term
        term=buildLayer(term, n)
        if term.num_parameters>0:
            for theta in range(term.num_parameters):
                grad=gradi(theta, np.ones(term.num_parameters), term, hamiltonian, Estimator())[-1]
                curr+=abs(grad)
            heapq.heappush(Lheap, (-curr/term.num_parameters, i))
        i+=1
    id=0
    badcircuit = circuit.copy()
    while len(Lheap)>0 and id<layers:
        nextLayerStr=L[heapq.heappop(Lheap)[1]]
        temp=buildLayer(nextLayerStr, n)
        id+=1
        naiveLayer = QuantumCircuit(n)
        Params = ParameterVector(str(nextLayerStr)+str(id), temp.num_parameters)
        currParams = []
        t=0
        for i in range(0,n):
            if nextLayerStr[i]=='R':
                naiveLayer.ry(Params[t], i)
                currParams.append(i)
                t+=1
        tempAnsatz = ansatz.compose(naiveLayer)
        tempCircuit = circuit.compose(tempAnsatz)
        badcircuit=badcircuit.compose(tempAnsatz)
        accumulator = []
        for i in range(naiveLayer.num_parameters):
            heapq.heappush(accumulator, (abs(gradi(i,np.ones(tempCircuit.num_parameters),tempCircuit,hamiltonian,estimator)[-1]).item(),i))
        ### 50% pruning rate
        rate=0.5
        remove = []
        r=int(math.ceil(rate*len(accumulator)))
        for i in range(0,r):
            index = heapq.heappop(accumulator)[1]
            heapq.heappush(remove,index)
        t=0
        for i in range(r):
            curr=heapq.heappop(remove)
            del currParams[curr-t]
            del naiveLayer.data[curr-t]
            t+=1
        ansatz = ansatz.compose(naiveLayer)
        c = circuit.compose(ansatz)
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

        # Skip optimization if no parameters after pruning
        if len(currParams) == 0:
            # A dummy result object to handle no parameters
            class DummyResult:
                def __init__(self):
                    cost = cost_func([], tempC, H, estimator)
                    if isinstance(cost, np.ndarray):
                        self.fun = float(cost.item() if cost.size == 1 else cost[0])
                    elif isinstance(cost, list):
                        self.fun = float(cost[0] if len(cost) > 0 else 0)
                    else:
                        self.fun = float(cost)
                    self.x = np.array([])
                    self.success = True
            x = DummyResult()
        else:
            x = minimize(cost_func, np.ones(len(currParams)), args=(tempC, H, estimator), method="COBYLA")
        #print(x.fun)
        #print(x.x)
        index=0
        """ for p in range(len(currParams)-1, -1, -1):
            params[len(params)-p-1]=x.x[index].item() # add params to main param vector
            index=index+1 """
        #ansatz.save_statevector(str(l))
        circuit2 = circuit.compose(ansatz)
        circuit2.save_statevector(str(len(Lheap)))
        params=np.ones(circuit2.num_parameters)
        circfinal = circuit2.assign_parameters(params)
        results = simulator.run(circfinal, shots=1024).result()
        temp = partial_trace(results.data(0)[str(len(Lheap))], [])
        prev = np.diagonal(temp)
        norm = np.linalg.norm(prev-np.zeros(int(math.pow(2, n))))
        prev=prev/norm # update prev to latest layer's output
        del circuit2.data[len(circuit2.data)-1]
    #print(params)
    circuit = circuit.compose(ansatz)
    """ index=0
    for p in circuit.parameters:
        p.assign(p, params[index])
        index=index+1
    print(circuit.parameters) """
    #print(circuit)
    # Skip optimization if no parameters
    if len(params) == 0 or circuit.num_parameters == 0:
        # A dummy result object
        class DummyResult:
            def __init__(self):
                cost = cost_func([], circuit, H, estimator)
                if isinstance(cost, np.ndarray):
                    self.fun = float(cost.item() if cost.size == 1 else cost[0])
                elif isinstance(cost, list):
                    self.fun = float(cost[0] if len(cost) > 0 else 0)
                else:
                    self.fun = float(cost)
                self.x = np.array([])
                self.success = True
        x = DummyResult()
    else:
        x = minimize(cost_func, params, args=(circuit, H, estimator), method="COBYLA")
    #print(x)
    #print('Cost: ' + str(cost_func(params, circuit, hamiltonian, Estimator())))
    return x, circuit, badcircuit

def layer_Grad(H, A):
    #H=SparsePauliOp.from_list([("ZIZZ", 2)])
    #A=SparsePauliOp.from_list([("IXZZ", 3)])
    c=commutator(H,A)
    #c=OP.commutator(H,H2)
    print('c'+str(c))
    a=cost_func([], QuantumCircuit(4), c, Estimator())
    print('a: ' + str(a))
    return c

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
    guess=[]
    c,x_v,_ = LayerOptimizer(guess, circuit,5,final,observables,Estimator())
    print(x_v)
    #matplotlib.pyplot.show()
    x_v.draw(output='mpl')
    matplotlib.pyplot.show() 

    guess=[]
    ans = []
    fields = ['H', 'goal', 'time', 'min', 'diff']
    total_speedup = 0
    total_err = 0
    #final = final.assign_parameters(guess)
    for t in range(0, 0):
        H=makeH(4, 5)
        observables = [
            *H.paulis,H
        ]
        #print(H)
        guess=[]
        s = time.time()
        c = LayerOptimizer(guess, circuit,7,final,observables,Estimator())
        #e = time.time()
        #s = time.time()
        x = minimize(cost_func, c[0].x, args=(c[1], H, Estimator()), method="COBYLA")
        cn=x
        e = time.time()
        topt = e-s
        #print(e-s)
        s = time.time()
        x = minimize(cost_func, np.ones(len(c[2].parameters)), args=(c[2], H, Estimator()), method="COBYLA")
        e = time.time()
        tnon = e-s
        #print(e-s)
        print('L: ' + str(cn.fun))
        #print(c[1])
        print(x.fun)
        a = {}
        a['H'] = H
        a['goal'] = np.sum(H.coeffs)
        a['time'] = (tnon-topt)/tnon
        a['min'] = cn.fun
        a['diff'] = x.fun - cn.fun
        total_speedup += (tnon-topt)/tnon
        total_err+=x.fun - cn.fun
        ans.append(a)
        if cn.fun>x.fun:
            print(c[1].num_parameters)
            print(c[2].num_parameters)
    #print(ans)
    print(total_speedup/50)
    print(total_err/50)
    """ with open("LayerGrad+min.csv", "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ans) """
"""
speedup: -1.2546402998485429
error: -1.5743792010096762

min
speedup: -1.5370859664085887
error: -1.6177806537058543
"""