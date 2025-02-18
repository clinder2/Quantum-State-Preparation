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

# Generic cost function for hamiltonian
def cost_func(params, circuit, hamiltonian, estimator):
    pub = (circuit, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    return cost

# Parameter-shift rule gradiant calculator for ith parameter
def gradi(i, params, circuit, hamiltonian, estimator):
    delta = np.zeros(len(params))
    delta[i] = math.pi/2
    costp = cost_func(params+delta, circuit, hamiltonian, estimator)
    costm = cost_func(params-delta, circuit, hamiltonian, estimator)
    return (costp-costm)/2