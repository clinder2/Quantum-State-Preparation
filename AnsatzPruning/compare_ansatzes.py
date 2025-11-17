"""Custom ansatz vs EfficientSU2 using QGA Hamiltonian."""
import time
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from Adapt import LayerOptimizer
from Utilities import cost_func, Estimator
from SLSQP import slsqp


def load_qga_hamiltonian():
    """Load Hamiltonian from QGA/hamiltonianOutput.txt"""
    h_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'QGA', 'hamiltonianOutput.txt')
    terms = []
    with open(h_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: PAULISTRING|coeff1,coeff2|...
            parts = line.split('|')
            pauli_str = parts[0]
            # Use first coefficient as weight (or default to 1.0)
            coeff = float(parts[1].split(',')[0]) if len(parts) > 1 else 1.0
            terms.append((pauli_str, coeff))
    return SparsePauliOp.from_list(terms)


def wrap_cost(cost):
    """Ensure cost_func returns a scalar."""
    if isinstance(cost, np.ndarray):
        return float(cost.item() if cost.size == 1 else cost[0])
    if isinstance(cost, list):
        return float(cost[0] if len(cost) > 0 else 0)
    return float(cost)


def benchmark_custom(hamiltonian, n_qubits, layers, estimator):
    """Benchmark custom ansatz."""
    circuit = QuantumCircuit(n_qubits)
    final = QuantumCircuit(n_qubits)
    
    start = time.time()
    result = LayerOptimizer([], circuit, layers, final, hamiltonian, estimator)
    build_time = time.time() - start
    
    ansatz = result[1]
    params = result[0].x
    
    start = time.time()
    opt_result = minimize(
        lambda p: wrap_cost(cost_func(p, ansatz, hamiltonian, estimator)),
        params,
        method="COBYLA"
    )
    opt_time = time.time() - start
    
    return {
        'ansatz_type': 'Custom',
        'energy': opt_result.fun,
        'time': build_time + opt_time,
        'params': len(ansatz.parameters),
        'converged': opt_result.success,
    }


def benchmark_su2(hamiltonian, n_qubits, reps, estimator):
    """Benchmark EfficientSU2."""
    ansatz = efficient_su2(n_qubits, reps=reps)
    params = np.random.random(len(ansatz.parameters))
    
    start = time.time()
    result = slsqp(
        func=lambda p: wrap_cost(cost_func(p, ansatz, hamiltonian, estimator)),
        x0=params,
        maxiter=200,
    )
    opt_time = time.time() - start
    
    return {
        'ansatz_type': 'EfficientSU2',
        'energy': result.fun,
        'time': opt_time,
        'params': len(ansatz.parameters),
        'converged': result.success,
    }


def run_comparison(hamiltonian, n_qubits, layers=5, reps=2, trials=3):
    """Run compact comparison."""
    estimator = Estimator()
    results = []
    
    print(f"\n{'='*50}")
    print(f"Benchmark: {n_qubits} qubits, {len(hamiltonian)} terms")
    print(f"{'='*50}\n")
    
    # Custom ansatz
    print("Custom Ansatz:", end=" ")
    for i in range(trials):
        try:
            r = benchmark_custom(hamiltonian, n_qubits, layers, estimator)
            r['trial'] = i + 1
            results.append(r)
            print(f"✓", end=" ")
        except Exception as e:
            print(f"✗", end=" ")
    print()
    
    # EfficientSU2
    print("EfficientSU2:  ", end=" ")
    for i in range(trials):
        try:
            r = benchmark_su2(hamiltonian, n_qubits, reps, estimator)
            r['trial'] = i + 1
            results.append(r)
            print(f"✓", end=" ")
        except Exception as e:
            print(f"✗", end=" ")
    print()
    
    if not results:
        print("\n⚠ No successful trials.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Summary
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    for ansatz_type in ['Custom', 'EfficientSU2']:
        subset = df[df['ansatz_type'] == ansatz_type]
        if len(subset) > 0:
            print(f"\n{ansatz_type}:")
            print(f"  Energy:  {subset['energy'].mean():.6f} ± {subset['energy'].std():.6f}")
            print(f"  Time:    {subset['time'].mean():.3f}s ± {subset['time'].std():.3f}s")
            print(f"  Params:  {subset['params'].mean():.1f} ± {subset['params'].std():.1f}")
            print(f"  Success: {subset['converged'].sum()}/{len(subset)}")
    
    # Comparison
    custom = df[df['ansatz_type'] == 'Custom']
    su2 = df[df['ansatz_type'] == 'EfficientSU2']
    if len(custom) > 0 and len(su2) > 0:
        e_improve = ((su2['energy'].mean() - custom['energy'].mean()) / abs(su2['energy'].mean())) * 100
        t_speedup = su2['time'].mean() / custom['time'].mean()
        print(f"\n{'='*50}")
        if e_improve > 0:
            print(f"✓ Custom: {e_improve:.1f}% lower energy, {t_speedup:.2f}x faster")
        else:
            print(f"✗ EfficientSU2 performs better")
    
    print()
    return df


def main():
    """Main function."""
    H = load_qga_hamiltonian()
    n_qubits = H.num_qubits
    
    results = run_comparison(H, n_qubits, layers=5, reps=2, trials=3)
    
    if len(results) > 0:
        results.to_csv("AnsatzComparison.csv", index=False)
        print("Results saved to AnsatzComparison.csv")
    
    return results


if __name__ == "__main__":
    main()
