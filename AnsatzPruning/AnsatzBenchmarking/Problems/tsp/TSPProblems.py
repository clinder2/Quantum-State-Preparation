import numpy as np
import itertools
from typing import List, Tuple

from qiskit.quantum_info import SparsePauliOp
from ..base import ProblemSet
from .TSPHamiltonian import buildTSPHamiltonian


def brute_force_tsp_min_cost(distance_matrix: np.ndarray) -> float:
    D = np.array(distance_matrix, dtype=float)
    N = D.shape[0]
    if N == 0:
        return 0.0
    cities = list(range(N))
    best = float("inf")
    for perm in itertools.permutations(cities[1:]):
        tour = (0,) + perm
        cost = 0.0
        for t in range(N):
            i = tour[t]
            j = tour[(t + 1) % N]
            cost += D[i, j]
        best = min(best, cost)
    return float(best)


def problem1() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem2() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 2, 2],
        [2, 0, 1],
        [2, 1, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem3() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 1, 1],
        [1, 0, 2],
        [1, 2, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem4() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 2, 2],
        [2, 0, 2],
        [2, 2, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem5() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 1, 4],
        [1, 0, 7],
        [4, 7, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem6() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 2, 10],
        [2, 0, 3],
        [10, 3, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem7() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 5, 5],
        [5, 0, 4],
        [5, 4, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem8() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem9() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 8, 3],
        [8, 0, 6],
        [3, 6, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem10() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 2, 9],
        [2, 0, 2],
        [9, 2, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem11() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 1, 6],
        [1, 0, 1],
        [6, 1, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


def problem12() -> Tuple[np.ndarray, float]:
    D = np.array([
        [0, 3, 9],
        [3, 0, 4],
        [9, 4, 0]
    ], dtype=float)
    return D, brute_force_tsp_min_cost(D)


class TSPProblemSet(ProblemSet):

    def createProblemSets(self) -> List[Tuple[SparsePauliOp, float]]:
        instances = [
            problem1(),
            problem2(),
            problem3(),
            problem4(),
            problem5(),
            problem6(),
            problem7(),
            problem8(),
            problem9(),
            problem10(),
            problem11(),
            problem12(),
        ]

        problems: List[Tuple[SparsePauliOp, float]] = []

        for i, (D, min_cost) in enumerate(instances):
            H = buildTSPHamiltonian(D, penalty=20.0)

            expectedAns = -float(min_cost)

            problems.append((H, expectedAns))

        return problems
