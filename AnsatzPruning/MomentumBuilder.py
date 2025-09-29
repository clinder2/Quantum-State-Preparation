import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True
from qiskit.circuit import ParameterVector
from Utilities import *
import heapq

from rotosolve import rotosolve
"""
Function to add momentum aware layer layers times
Calculates gradients for each layer, removes smallest magnitude RY gates with given pruning rate
"""
def momen_layer(it:int, n:int, momentum:list, radius:int=1, keep:int=2):
    lay=QuantumCircuit(n)
    params=[]
    inds=[]
    for i in range(keep):
        angle = Parameter("it-"+str(it)+", "+str(i))
        ind=momentum[i][1]
        params.append(1)
        inds.append(ind)
        lay.rx(angle, ind)
        for r in range(1,radius+1):
            if ind +r<n:
                lay.cx(ind,ind+r)
            if ind-r>=0:
                lay.cx(ind,ind-r)
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
        print(accumulator)
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
    print(circuit)
        #print(ansatz.data)
        #print(cost_func(params,circuit,hamiltonian,estimator))
    # circuit = circuit.compose(ansatz)
    x = minimize(cost_func, params, args=(circuit, H, estimator), method="COBYLA")
    print(x)

if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 1),("ZZII", 3),("IZZI", 1),("IIZZ", 1)]) # Toy hamiltonian
    observables = [
        *H.paulis,H
    ]
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    circuit = QuantumCircuit(4)
    ansatz = QuantumCircuit(4)
    ansatz.rx(angle1, 0)
    ansatz.rx(angle2, 1)
    ansatz.rx(angle3, 2)
    ansatz.rx(angle4, 3)
    #circuit = circuit.compose(ansatz)
    MomentumBuilder([1,1,1,1],[0,1,2,3],ansatz,circuit,observables,Estimator(),.9,.99)