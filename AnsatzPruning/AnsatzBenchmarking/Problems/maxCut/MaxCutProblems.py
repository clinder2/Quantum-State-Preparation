from ..base import ProblemSet
from .MaxCutHamiltonian import buildMaxCutHamiltonian
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

def problem1(): 
    # Square graph with heavy diagonal 
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 0, weight=1.0)
    G.add_edge(0, 2, weight=2.0) 
    expectedAns = 4.0 
    # 0110 1001 
    return (G, expectedAns)

def problem2(): 
    # Triangel with a heavy edge
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0) # Heavy edge
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    expectedAns = 4.0
    # 011 101 110 
    return (G, expectedAns)


def problem3(): 
    #line graph with one heavy edge 
    G = nx.Graph()
    G.add_edge(0, 1, weight=10.0)
    G.add_edge(1, 2, weight=1.0)
    expectedAns = 11.0
    return (G, expectedAns)


class MaxCutProblemSet(ProblemSet): 
    def createProblemSets(self) -> list[ tuple[SparsePauliOp, float]]:
        graphs: list[tuple[nx.Graph, float]] = [
            problem1(), 
            problem2(), 
            problem3()
        ] 
        problems:list[ tuple[SparsePauliOp, float]]= [(None, None)]*len(graphs)

        for i, (graph, ans) in enumerate(graphs): 
            problems[i] = (buildMaxCutHamiltonian(graph), ans)

        return problems 
