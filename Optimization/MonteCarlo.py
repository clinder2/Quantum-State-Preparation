# Imports
import numpy as np
import math
from qiskit.quantum_info import *

# Simulated Annealing optimization
def simulated_annealing(runs, params, ansatz, simulator):
    B = E(params, ansatz, simulator) 
    prev_E = B

    # Temperature function for simulated annealing
    def T(t):
        c = 0.02
        a = 0.01
        temperature = c / (a + math.log(t)) 
        return temperature

    # Main loop for simulated annealing
    for t in range(1, runs):
        delta = np.random.normal(0, .1, 4) 
        params_new = params + delta 
        E_new = E(params_new, ansatz, simulator) 
        delta_E = E_new - prev_E 

        if delta_E <= 0:
            params = params_new
            prev_E = E_new
        else:
            h = math.pow(math.e, -1 * delta_E / T(t)) 
            U = np.random.normal(.5, .5, 1) 
            if U < h:
                params = params_new
                prev_E = E_new

    return params 

# Global Best Particle Swarm optimization
def gbest_pso(runs, params, ansatz, simulator):
    num_particles = 20  # Number of particles in the swarm
    dimensions = len(params)  # Number of parameters to optimize

    particles = np.random.rand(num_particles, dimensions) * 2 - 1  # Random positions in range [-1, 1]
    velocities = np.zeros((num_particles, dimensions))  # Initial velocities
    personal_best_positions = np.copy(particles)  # Initial best positions for each particle
    personal_best_scores = [cost_func(p, ansatz, simulator) for p in particles]  # Initial best scores for each particle

    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]  # Best position from all particles

    for j in range(runs):
        # Update personal best and global best
        for i in range(num_particles):
            score = cost_func(particles[i], ansatz, simulator)  # New score

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

            # Update global best
            if score < cost_func(global_best_position, ansatz, simulator):
                global_best_position = particles[i]

        # Update velocities and positions
        for i in range(num_particles):
            c1 = 2.0  # Personal best weight
            c2 = 2.0  # Global best weight
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)

            # Update velocity
            velocities[i] = (0.5 * velocities[i] + 
                            c1 * r1 * (personal_best_positions[i] - particles[i]) + 
                            c2 * r2 * (global_best_position - particles[i]))
            
            # Limit the maximum velocity
            max_velocity = 0.1
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Update position
            particles[i] = particles[i] + velocities[i]

    return global_best_position

# Differential Evolution optimization
def diff_evolution(runs, params, ansatz, simulator):
    bounds = [(-1, 1)] * len(params)
    dimensions = len(bounds)
    popsize = 20
    mut = 0.8  # Mutation factor, usually chosen from the interval [0.5, 2.0]
    crossp = 0.7  # Crossover probability, between [0, 1]

    # Initialization
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([cost_func(ind, ansatz, simulator) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(runs):
        for j in range(popsize):
            # Mutation
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)

            # Recombination
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])

            # Replacement
            trial_denorm = min_b + trial * diff
            f = cost_func(trial_denorm, ansatz, simulator)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

    return best

# Stochastic Hill Climbing optimization
def stochastic_hill_climbing(runs, params, ansatz, simulator):
    max_iterations = 20
    step_size = 0.1 # Step size for perturbation
    threshold = 0.9
    best_state = None
    best_score = float('inf')

    for restart in range(1, runs + 1):
        current_state = params.copy()
        current_score = cost_func(current_state, ansatz, simulator)

        for iteration in range(1, max_iterations + 1):
            # Apply a random perturbation to generate a neighboring state
            neighbor = current_state + np.random.normal(0, step_size, size=current_state.shape)
            neighbor_score = cost_func(neighbor, ansatz, simulator)

            if neighbor_score < current_score:
                current_state = neighbor
                current_score = neighbor_score

            # Update the best state and score if current is better
            if current_score < best_score:
                best_state = current_state
                best_score = current_score

            if current_score >= threshold:
                break
    return best_state

# Function to calculate cost based on parameters
def cost_func(params, ansatz, simulator):
    circfinal = ansatz.assign_parameters(params) 
    results = simulator.run(circfinal, shots=1024).result() 
    counts = results.get_counts() 
    b = counts.get('1', 0) 
    s = -1 * (1 - ((2 / 1024) * b)) 
    return s 

# Function to calculate energy based on parameters
def E(params, ansatz, simulator):
    circfinal = ansatz.assign_parameters(params) 
    results = simulator.run(circfinal, shots=20).result() 
    temp = partial_trace(results.data(0)['ans'], [0, 3, 4])
    partial = np.diagonal(temp) 
    temp = partial_trace(results.data(0)['ans'], [0, 1, 2])
    partial2 = np.diagonal(temp) 
    norm = np.linalg.norm(partial - partial2) 
    return norm 
