"""Evaluate layer builder circuits on simple Hamiltonians and print costs.

Run from repository root (this script will use the current Python environment):
    python3 scripts/eval_layers.py

Requires qiskit and numpy installed.
"""
import sys
import os
import numpy as np
from qiskit.quantum_info import SparsePauliOp

# Make project root importable so we can import QGA package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from QGA.LayerGA import buildLayer, randomLayer


def statevector_cost(params, layer, hamiltonian):
    """Evaluate expectation value <psi|H|psi> by simulating statevector from circuit."""
    # bind parameters and simulate statevector using qiskit.quantum_info.Statevector
    qc_bound = layer.assign_parameters(params)
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc_bound)
    mat = hamiltonian.to_matrix()
    exp = np.vdot(sv.data, mat.dot(sv.data)).real
    return exp


def simple_hamiltonians(n_qubits):
    # single-Z on qubit 0, pair ZZ on qubits 0-1, alternating ZI pattern
    hz0 = SparsePauliOp.from_list([('Z' + 'I'*(n_qubits-1), 1)])
    hzz = SparsePauliOp.from_list([('Z'*2 + 'I'*(n_qubits-2), 1)])
    hall = SparsePauliOp.from_list([( ''.join(['Z' if i%2==0 else 'I' for i in range(n_qubits)]), 1)])
    return {'Z0': hz0, 'ZZ': hzz, 'ALT': hall}


def evaluate_chromosome(chrom, n_qubits):
    layer = buildLayer(chrom, n_qubits)
    num_params = layer.num_parameters
    params = np.ones(num_params)
    results = {}
    for name, H in simple_hamiltonians(n_qubits).items():
        try:
            cost = statevector_cost(params, layer, H)
            results[name] = float(cost)
        except Exception as e:
            results[name] = f'ERROR: {e}'
    return results


def main():
    n_qubits = 4
    test_chroms = ['RRRR', 'XXXX', 'ZZZZ', 'XRYI', randomLayer(n_qubits)]
    print(f'Evaluating {len(test_chroms)} chromosomes on {n_qubits} qubits')
    for chrom in test_chroms:
        print('\nChromosome:', chrom)
        res = evaluate_chromosome(chrom, n_qubits)
        for hname, val in res.items():
            print(f'  {hname}: {val}')


if __name__ == '__main__':
    main()
