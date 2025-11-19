import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
from qiskit.circuit import ParameterVector
from Utilities import *
import heapq
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator

from rotosolve import rotosolve
"""
Function to add momentum aware layer layers times
Calculates gradients for each layer, removes smallest magnitude RY gates with given pruning rate
"""
def momen_layer(it:int, n:int, momentum:list, radius:int=1, keep:int=2):
    from QGA.LayerGA import buildLayer, randomLayer
    # Use a chromosome string to encode RX/RY/RZ gates for each qubit
    if n == 4:
        chrom = 'XRYZ'
    else:
        chrom = randomLayer(n)
    lay = buildLayer(chrom, n)
    params = [1] * lay.num_parameters
    inds = list(range(n))
    return lay, params, inds

def MomentumBuilder(params:list, inds:list, ansatz:QuantumCircuit,
                 circuit:QuantumCircuit, hamiltonian:SparsePauliOp, 
                 estimator:Estimator, beta1:float, beta2:float, iters:int=2):
    n = circuit.num_qubits
    M=np.zeros((len(params))) ###Momentum
    currCirc=QuantumCircuit(n)
    currCirc=currCirc.compose(ansatz)
    for it in range(iters):
        ### Momentum layer construction
        # naiveLayer = momen_layer()
        # tempAnsatz = ansatz.compose(naiveLayer)
        # tempCircuit = circuit.compose(tempAnsatz)
        accumulator = []
        for i in range(len(params)):
            #print(gradi(i,params,circuit,hamiltonian,estimator))
            grad_i=abs(gradi(i,params,currCirc,hamiltonian,estimator)[len(hamiltonian)-1]).item()
            M[i]=beta1*M[i]+(1-beta1)*grad_i
            heapq.heappush(accumulator, (M[i],inds[i]))
        ### Momentum layer construction
        # print(accumulator)
        mLayer,nparams,ninds=momen_layer(it,n, accumulator)
        params=params+nparams
        inds=inds+ninds
        M=np.concatenate((M,len(nparams)*[0]))
        ansatz = ansatz.compose(mLayer)
        currCirc = circuit.compose(ansatz)
        # rate=0.5
        # bound = math.floor(rate*n)
        # remove = []
        # for i in range(0,bound):
        #     index = heapq.heappop(accumulator)[1]
        #     heapq.heappush(remove,index%n)
        # i = 0
        # while len(remove) > 0:
        #     del naiveLayer.data[heapq.heappop(remove)-i]
        #     i = i + 1
        #     del params[len(params)-1]
        # tempParams = []
        # for i in range(n):
        #     tempParams.append(1)
        # lay = minimize(cost_func, tempParams, args=(circuit, H, estimator), method="COBYLA")
        # print("layer by layer", lay)

    circuit=circuit.compose(ansatz)
    # print(circuit)
    #     #print(ansatz.data)
    #     #print(cost_func(params,circuit,hamiltonian,estimator))
    # # circuit = circuit.compose(ansatz)
    # x = minimize(cost_func, params, args=(circuit, H, estimator), method="COBYLA")
    # print(x)

    return circuit

if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 1),("ZZII", 3),("IZZI", 1),("IIZZ", 1)]) # Toy hamiltonian
    circuit = QuantumCircuit(4)
    ansatz = QuantumCircuit(4)
    # Use MomentumBuilder to build the circuit
    final_circuit = MomentumBuilder([1,1,1,1],[0,1,2,3],ansatz,circuit,H,None,.9,.99)
    # Use dummy params for demonstration
    params = [1.0] * final_circuit.num_parameters
    from qiskit.quantum_info import Statevector
    def statevector_cost(params, circuit, hamiltonian):
        qc_bound = circuit.assign_parameters(params)
        sv = Statevector.from_instruction(qc_bound)
        mat = hamiltonian.to_matrix()
        return float((sv.data.conj().T @ mat @ sv.data).real)
    cost = statevector_cost(params, final_circuit, H)
    print("Expectation value <psi|H|psi>:", cost)