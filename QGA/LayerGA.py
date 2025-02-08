import numpy as np
import math
import pandas as pd
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
import random

"""
Chromosome format: 'RRIR|0,1|2,3|1,3'
"""

def buildLayer(chrom: str, qbits: int):
    layer = QuantumCircuit(qbits)
    params = []
    for i in range(0, qbits):
        if chrom[i]=='R':
            p=Parameter(str(i))
            params.append(p)
            layer.ry(p, i)
    if len(chrom) > qbits:
        cx = chrom.split('|')[1:]
        for cxGate in cx:
            cxGate=cxGate.split(',')
            q0=int(cxGate[0])
            q1=int(cxGate[1])
            layer.cx(q0, q1)
    return layer

def randomLayer(qubits: int):
    MAXCX = 3
    r = np.random.normal(.5, .5, qubits + MAXCX)
    l = ''
    for i in range(0, qubits):
        if r[i] <= .5:
            l += 'R'
        else:
            l+='I'
    for i in range(qubits, len(r)):
        if r[i] <= .5:
            l+='|'
            i1 = randint(0, qubits-1)
            i2 = randint(0, qubits-1)
            while i2 == i1:
                i2 = randint(0, qubits-1)
            l+=str(i1) + ',' + str(i2)
    print(l)
    return l

def fitness(qubits, layer: QuantumCircuit, estimator: Estimator, H):
    layer = buildLayer(layer, qubits)
    params = np.ones(len(layer.parameters))
    return cost_func(params, layer, H, estimator)[len(H)-1]

def mutate(layer: QuantumCircuit):
    return layer

def QGA(popSize: int, qubits: int, generations: int, estimator: Estimator, H):
    chromPop = [randomLayer(qubits) for i in range(0, popSize)]
    fitnessVals = [fitness(qubits, chromPop[i], estimator, H).item() for i in range(0, popSize)]
    # GA PARAMETERS
    elitism = 0.1
    pm = 0.1 # prob. mutation
    orelax = 1.4 # over relaxation factor for mating
    epsilon = 0.07 # mutation range
    bestFitness = max(fitnessVals)
    for i in range(0, generations):
        fitnessVals = [fitness(qubits, chromPop[i], estimator, H) for i in range(0, popSize)]
        sorted=np.argsort(fitnessVals)[::-1]
        ordered = []
        for s in sorted:
            ordered.append(chromPop[s])
        if fitnessVals[sorted[0]] > bestFitness:
            bestFitness = fitnessVals[sorted[0]]
        # Preserve elites
        elite = int(elitism*popSize)
        newPop = []
        for j in range(0, elite):
            newPop.append(ordered[j])
        ## MUTATION
        r = np.random.normal(.5, .5, popSize)
        for j in range(0, len(r)):
            if r[j] <= pm:
                ordered[j] = mutate(ordered[j])
        ## MATING TODO
        # Randomly select from modified population
        indicies = []
        for t in range(0, popSize-elite):
            indicies.append(t)
        r = random.sample(indicies, popSize - elite)
        for t in range(0, popSize-elite):
            newPop.append(ordered[r[t]])
        chromPop = newPop
        print(fitnessVals)


if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 1),("ZZII", 3),("IZZI", 1),("IIZZ", 1)]) # Toy hamiltonian
    observables = [
        *H.paulis,H
    ]
    buildLayer('RRIR|0,1|2,3|1,3', 4)
    QGA(10, 4, 1, Estimator(), observables)
