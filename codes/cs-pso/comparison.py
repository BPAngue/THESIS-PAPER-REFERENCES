"""
Comparison Script: Standard PSO vs CS-PSO

This script runs both algorithms with identical parameters to compare:
1. Solution quality (fitness)
2. Convergence speed (iterations to solution)
3. Computational efficiency (time)
4. Solution diversity (ergodicity)
"""

import problem4
import pso_solver
import cs_pso_solver
import numpy as np
import math
import sys
import os
from datetime import datetime
import time

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"comparison_{timestamp}.txt")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_filename)

# --------------------------------------
# Test Parameters
# --------------------------------------
num_inputs = problem4.truth_table.num_inputs
num_outputs = problem4.truth_table.num_outputs
num_rows = problem4.num_rows
nVar = (num_rows * 3) + num_outputs
varSize = nVar
varMin = 0
varMax = num_inputs + num_rows

nPop = 100  # Smaller population for faster comparison
MAX_ITERATIONS = 100
NUM_RUNS = 5  # Run each algorithm multiple times

# PSO parameters
phi1, phi2 = 1.05, 2.05
phi = phi1 + phi2
chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
w, w_damp, c1, c2 = chi, 1, chi * phi1, chi * phi2

velMax = 0.2 * (varMax - varMin)
velMin = -velMax

pso_params = {'w': w, 'c1': c1, 'c2': c2, 'velMax': velMax, 'velMin': velMin, 'w_damp': w_damp}

print("=" * 80)
print("PSO vs CS-PSO COMPARISON")
print("=" * 80)
print(f"\nProblem: {num_inputs} inputs, {num_outputs} outputs, {num_rows} gates")
print(f"Truth table size: {problem4.truth_table.num_rows_tt} rows")
print(f"Total outputs to match: {problem4.truth_table.total_outputs}")
print(f"\nTest Parameters:")
print(f"  Population size: {nPop}")
print(f"  Max iterations: {MAX_ITERATIONS}")
print(f"  Number of runs: {NUM_RUNS}")
print(f"  PSO params: w={w:.4f}, c1={c1:.4f}, c2={c2:.4f}")
print("=" * 80)

# --------------------------------------
# Run Comparisons
# --------------------------------------

def run_single_test(solver_class, solver_name, run_idx):
    """Run a single test with a given solver."""
    start_time = time.perf_counter()
    
    if solver_name == "Standard PSO":
        solver = solver_class(nPop, varSize, pso_params)
    else:  # CS-PSO
        solver = solver_class(nPop, varSize, pso_params, num_inputs, num_rows, num_outputs)
    
    # Track convergence
    best_fitness_history = []
    iterations_to_perfect = None
    
    for iteration in range(MAX_ITERATIONS):
        solver.run_iterations(1)
        gbest = solver.get_gbest()
        best_fitness_history.append(gbest.fitness)
        
        # Check if perfect solution found
        if gbest.num_equal_tt == problem4.truth_table.total_outputs and iterations_to_perfect is None:
            iterations_to_perfect = iteration + 1
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    final_gbest = solver.get_gbest()
    
    return {
        'fitness': final_gbest.fitness,
        'correct': final_gbest.num_equal_tt,
        'gates': final_gbest.num_gates,
        'time': elapsed,
        'iterations_to_perfect': iterations_to_perfect,
        'fitness_history': best_fitness_history
    }

# Run Standard PSO
print("\n" + "=" * 80)
print("Running Standard PSO...")
print("=" * 80)
pso_results = []
for run in range(NUM_RUNS):
    print(f"\nRun {run + 1}/{NUM_RUNS}...", end=" ")
    result = run_single_test(pso_solver.BasePSOSolver, "Standard PSO", run)
    pso_results.append(result)
    print(f"Fitness: {result['fitness']:.2f}, Correct: {result['correct']}/{problem4.truth_table.total_outputs}, Time: {result['time']:.2f}s")

# Run CS-PSO
print("\n" + "=" * 80)
print("Running CS-PSO...")
print("=" * 80)
cs_pso_results = []
for run in range(NUM_RUNS):
    print(f"\nRun {run + 1}/{NUM_RUNS}...", end=" ")
    result = run_single_test(cs_pso_solver.CSPSOSolver, "CS-PSO", run)
    cs_pso_results.append(result)
    print(f"Fitness: {result['fitness']:.2f}, Correct: {result['correct']}/{problem4.truth_table.total_outputs}, Time: {result['time']:.2f}s")

# --------------------------------------
# Analyze Results
# --------------------------------------
print("\n\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

def analyze_results(results, name):
    """Compute statistics from results."""
    fitnesses = [r['fitness'] for r in results]
    corrects = [r['correct'] for r in results]
    gates = [r['gates'] for r in results]
    times = [r['time'] for r in results]
    iterations = [r['iterations_to_perfect'] for r in results if r['iterations_to_perfect'] is not None]
    
    print(f"\n{name}:")
    print(f"  Fitness:")
    print(f"    Mean: {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
    print(f"    Best: {np.max(fitnesses):.4f}")
    print(f"    Worst: {np.min(fitnesses):.4f}")
    
    print(f"  Correct Outputs:")
    print(f"    Mean: {np.mean(corrects):.2f}/{problem4.truth_table.total_outputs}")
    print(f"    Best: {np.max(corrects)}/{problem4.truth_table.total_outputs}")
    
    print(f"  Active Gates:")
    print(f"    Mean: {np.mean(gates):.2f}")
    print(f"    Best (fewest): {np.min(gates)}")
    
    print(f"  Computation Time:")
    print(f"    Mean: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    
    if iterations:
        print(f"  Iterations to Perfect Solution:")
        print(f"    Mean: {np.mean(iterations):.2f}")
        print(f"    Success rate: {len(iterations)}/{NUM_RUNS} ({100*len(iterations)/NUM_RUNS:.1f}%)")
    else:
        print(f"  Iterations to Perfect Solution: No perfect solutions found")
    
    return {
        'mean_fitness': np.mean(fitnesses),
        'best_fitness': np.max(fitnesses),
        'mean_correct': np.mean(corrects),
        'mean_gates': np.mean(gates),
        'mean_time': np.mean(times),
        'success_rate': len(iterations) / NUM_RUNS if iterations else 0
    }

pso_stats = analyze_results(pso_results, "Standard PSO")
cs_pso_stats = analyze_results(cs_pso_results, "CS-PSO")

# --------------------------------------
# Winner Determination
# --------------------------------------
print("\n" + "=" * 80)
print("WINNER ANALYSIS")
print("=" * 80)

metrics = {
    'Best Fitness': cs_pso_stats['mean_fitness'] > pso_stats['mean_fitness'],
    'Correctness': cs_pso_stats['mean_correct'] > pso_stats['mean_correct'],
    'Simplicity': cs_pso_stats['mean_gates'] < pso_stats['mean_gates'],
    'Success Rate': cs_pso_stats['success_rate'] > pso_stats['success_rate']
}

cs_pso_wins = sum(metrics.values())
pso_wins = len(metrics) - cs_pso_wins

print(f"\nMetric-by-metric comparison:")
for metric, cs_pso_better in metrics.items():
    winner = "CS-PSO" if cs_pso_better else "Standard PSO"
    print(f"  {metric}: {winner}")

print(f"\nOverall Score:")
print(f"  CS-PSO: {cs_pso_wins}/{len(metrics)} metrics")
print(f"  Standard PSO: {pso_wins}/{len(metrics)} metrics")

if cs_pso_wins > pso_wins:
    print(f"\n🏆 WINNER: CS-PSO")
elif pso_wins > cs_pso_wins:
    print(f"\n🏆 WINNER: Standard PSO")
else:
    print(f"\n🤝 TIE")

print("\n" + "=" * 80)
print("Key Advantages of CS-PSO (from paper):")
print("  1. Better ergodicity (explores search space more thoroughly)")
print("  2. Avoids premature convergence")
print("  3. Enhanced global search capability")
print("  4. Better diversity in solutions")
print("=" * 80)

print(f"\nResults saved to: {log_filename}")
