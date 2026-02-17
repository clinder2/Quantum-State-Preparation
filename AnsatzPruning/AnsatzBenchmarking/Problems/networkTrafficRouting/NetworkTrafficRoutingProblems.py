from ..base import ProblemSet
from .NetworkTrafficRoutingHamiltonian import buildNetworkTrafficRoutingHamiltonian
import networkx as nx
from qiskit.quantum_info import SparsePauliOp


# Each path is a list of edges
# Example path: [(0,1), (1,2)]


def problem1():
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(1, 2, weight=5.0)

    paths = [
        [(0, 1), (1, 2)],   # Path A
        [(0, 1)],           # Path B
    ]

    # Best: choose only path B -> cost = 5
    expectedAns = 5.0
    return (G, paths, expectedAns)


def problem2():
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0)

    paths = [
        [(0, 1)],
        [(0, 1)]
    ]

    # Choosing both causes congestion:
    # f = 2 → cost = 3 * 4 = 12
    # Best: choose one -> 3
    expectedAns = 3.0
    return (G, paths, expectedAns)


def problem3():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=2.0)

    paths = [
        [(0, 1)],
        [(0, 1), (1, 2)],
        [(1, 2)]
    ]

    expectedAns = 2.0
    return (G, paths, expectedAns)


def problem4():
    G = nx.Graph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 2, weight=4.0)
    G.add_edge(0, 2, weight=1.0)

    paths = [
        [(0, 1), (1, 2)],  # heavy route
        [(0, 2)]           # light direct route
    ]

    expectedAns = 1.0
    return (G, paths, expectedAns)


def problem5():
    G = nx.Graph()
    G.add_edge(0, 1, weight=10.0)

    paths = [
        [(0, 1)],
        [(0, 1)],
        [(0, 1)]
    ]

    # Best: choose one only -> 10
    expectedAns = 10.0
    return (G, paths, expectedAns)


class NetworkTrafficRoutingProblemSet(ProblemSet):

    def createProblemSets(self) -> list[tuple[SparsePauliOp, float]]:

        graphs = [
            problem1(),
            problem2(),
            problem3(),
            problem4(),
            problem5()
        ]

        problems = [(None, None)] * len(graphs)

        for i, (graph, paths, ans) in enumerate(graphs):
            problems[i] = (
                buildNetworkTrafficRoutingHamiltonian(graph, paths),
                ans
            )

        return problems
