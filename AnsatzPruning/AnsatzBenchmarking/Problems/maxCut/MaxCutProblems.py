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

def problem4(): 
    # K5
    G = nx.complete_graph(5)
    exceptedAns = 6 
    return (G, exceptedAns)

def problem5(): 
    # line graph 
    G = nx.path_graph(5)
    expectedAns = 4 
    return (G, expectedAns)

def problem6():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)], weight=1.0)
    expectedAns = 4.0
    return (G, expectedAns)

def problem7():
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(1, 2, weight=5.0)
    G.add_edge(0, 2, weight=5.0)
    G.add_edge(3, 4, weight=5.0)
    G.add_edge(4, 5, weight=5.0)
    G.add_edge(3, 5, weight=5.0)
    G.add_edge(2, 3, weight=1.0)
    expectedAns = 21.0
    return (G, expectedAns)

def problem8(): 
    # 4-node cycle with a very heavy anti-diagonal edge (0-3). 
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 0, weight=1.0)
    G.add_edge(1, 3, weight=10.0) 
    expectedAns = 12.0
    return (G, expectedAns)

def problem9(): 
    G = nx.complete_bipartite_graph(3, 3)
    expectedAns = 9.0 
    return (G, expectedAns)

def problem10(): 
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0) 
    G.add_edge(2, 3, weight=1.0) 
    expectedAns = 6.0
    return (G, expectedAns)

def problem11(): 
    G = nx.hypercube_graph(3)
    expectedAns = 12
    return (G, expectedAns)

def problem12(): 
    G = nx.cycle_graph(5)
    G.add_edge(1, 4, weight=5.0) 
    expectedAns = 9.0
    return (G, expectedAns)




class MaxCutProblemSet(ProblemSet): 
    def createProblemSets(self) -> list[ tuple[SparsePauliOp, float]]:
        graphs: list[tuple[nx.Graph, float]] = [
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
            problem12()
        ] 
        problems:list[ tuple[SparsePauliOp, float]]= [(None, None)]*len(graphs)

        for i, (graph, ans) in enumerate(graphs): 
            problems[i] = (buildMaxCutHamiltonian(graph), ans)

        return problems 
