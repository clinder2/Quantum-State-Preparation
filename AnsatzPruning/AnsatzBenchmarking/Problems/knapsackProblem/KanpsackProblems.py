from AnsatzPruning.AnsatzBenchmarking.Problems.base import ProblemSet
from KnapsackHamiltonian import buildKnapsackHamiltonian
from qiskit.quantum_info import SparsePauliOp
import itertools
import numpy as np

def knapsack_problem1():
    values = [10]
    weights = [5]
    W = 5
    P = 20
    expectedAns = -10
    return values, weights, W, P, expectedAns

def knapsack_problem2():
    values = [10]
    weights = [6]
    W = 5
    P = 20
    expectedAns = 0 + P * (0 - 5)**2
    return values, weights, W, P, expectedAns

def knapsack_problem3():
    values = [6, 10]
    weights = [2, 3]
    W = 3
    P = 20
    expectedAns = -10
    return values, weights, W, P, expectedAns

def knapsack_problem4():
    values = [6, 7]
    weights = [2, 1]
    W = 3
    P = 20
    expectedAns = -13
    return values, weights, W, P, expectedAns

def knapsack_problem5():
    values = [20, 11, 11]
    weights = [4, 2, 2]
    W = 4
    P = 20
    expectedAns = -22  # pick items 1 & 2
    return values, weights, W, P, expectedAns

def knapsack_problem6():
    values = [3, 4, 5, 6]
    weights = [1, 2, 3, 4]
    W = 5
    P = 20
    expectedAns = -7  # items 0 + 2
    return values, weights, W, P, expectedAns

def knapsack_problem7():
    values = [25, 6, 6, 6, 6]
    weights = [5, 1, 1, 1, 1]
    W = 4
    P = 20
    expectedAns = -24  # four small items
    return values, weights, W, P, expectedAns

def knapsack_problem8():
    values = [5, 5, 5, 5]
    weights = [2, 2, 2, 2]
    W = 4
    P = 20
    expectedAns = -10
    return values, weights, W, P, expectedAns

def knapsack_problem9():
    values = [8, 8, 2, 2]
    weights = [3, 3, 1, 1]
    W = 4
    P = 20
    expectedAns = -10
    return values, weights, W, P, expectedAns

def knapsack_problem10():
    values = [5, 5, 10, 10]
    weights = [1, 1, 3, 3]
    W = 4
    P = 20
    expectedAns = -15
    return values, weights, W, P, expectedAns

def knapsack_problem11():
    values = [3]*8
    weights = [1]*8
    W = 4
    P = 20
    expectedAns = -12
    return values, weights, W, P, expectedAns

def knapsack_problem12():
    values = [4, 4, 4, 4, 10]
    weights = [1, 1, 1, 1, 3]
    W = 4
    P = 20
    expectedAns = -10
    return values, weights, W, P, expectedAns

class KnapsackProblemSet(ProblemSet):
    def createProblemSets(self):
        problems_raw = [
            knapsack_problem1(),
            knapsack_problem2(),
            knapsack_problem3(),
            knapsack_problem4(),
            knapsack_problem5(),
            knapsack_problem6(),
            knapsack_problem7(),
            knapsack_problem8(),
            knapsack_problem9(),
            knapsack_problem10(),
            knapsack_problem11(),
            knapsack_problem12(),
        ]

        problems = []
        for values, weights, W, P, ans in problems_raw:
            H = buildKnapsackHamiltonian(values, weights, W, P)
            problems.append((H, ans))

        return problems
