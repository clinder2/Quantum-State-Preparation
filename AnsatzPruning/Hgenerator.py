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
import time

def makeH(q: int, terms: int):
    Ops = ['I', 'X', 'Y', 'Z'] # Paulis
    list = []
    for i in range(0, terms):
        newterm = ""
        for j in range(0, q):
            n = np.random.randint(0, 4)
            newterm+=Ops[n]
        coeff = np.random.randint(1, 11)
        list.append((newterm, coeff))
    H = SparsePauliOp.from_list(list)
    return H

if __name__ == "__main__":
    print(makeH(4, 2))