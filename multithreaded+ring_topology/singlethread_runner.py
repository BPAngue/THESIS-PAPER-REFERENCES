import problem4
import pso_solver
import numpy as np
import math
import sys
import os
import copy
from datetime import datetime
import time
import statistics
import csv

# --------------------------------------
# Configuration for Ablation Study
# --------------------------------------
NUM_RUNS = 10               # Total independent runs to perform
MAX_ITERATIONS = 100        # Total iterations (Equiv to MAX_VOTING_ROUNDS in multithreaded)
STAGNATION_LIMIT = 100       # Stop if no improvement for N iterations

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"pso_singlethread_ablation_{timestamp}.txt")
csv_filename = os.path.join(results_dir, f"pso_singlethread_convergence_{timestamp}.csv")

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
# Helper: Dynamic PSO Parameters
# --------------------------------------
def get_pso_params():
    num_inputs = problem4.truth_table.num_inputs
    num_rows = problem4.num_rows
    nVar = (num_rows * 3) + problem4.truth_table.num_outputs
    varSize = nVar
    
    phi1, phi2 = 2.05, 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    
    params = {
        'w': chi,
        'c1': chi * phi1,
        'c2': chi * phi2,
        'velMax': 0.2 * (problem4.num_rows + problem4.truth_table.num_inputs),
        'velMin': -0.2 * (problem4.num_rows + problem4.truth_table.num_inputs),
        'w_damp': 1
    }
    return varSize, params

# --------------------------------------
# Execution Core (Single Threaded)
# --------------------------------------
def execute_single_run(run_id, csv_writer):
    print(f"\n>>> STARTING RUN {run_id} (Standard PSO) <<<")
    start_time = time.perf_counter()
    
    # 1. Initialize Single Solver
    varSize, params = get_pso_params()
    # Note: Using 500 population to match the 'per-thread' population of your previous script
    solver = pso_solver.BasePSOSolver(500, varSize, params)
    
    # Trackers
    stagnation_counter = 0
    previous_best_fitness = -math.inf
    run_history = []
    
    # 2. Main Sequential Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        
        # Run 1 iteration of the solver
        solver.run_iterations(1)
        
        # Get current best
        current_gbest = solver.get_gbest()
        
        # Check for improvement
        if current_gbest.fitness > previous_best_fitness:
            previous_best_fitness = current_gbest.fitness
            stagnation_counter = 0
            # Log improvement
            print(f"  [Run {run_id}] Iter {iteration}: New Best Fitness {current_gbest.fitness:.2f} "
                  f"(Correct: {current_gbest.num_equal_tt}/{problem4.truth_table.total_outputs})")
        else:
            stagnation_counter += 1

        # CSV Data Collection
        run_history.append([run_id, iteration, current_gbest.fitness, current_gbest.num_equal_tt, current_gbest.num_gates])

        # Logging periodic status
        if iteration % 10 == 0:
             print(f"  [Run {run_id}] Iter {iteration}: Current: {current_gbest.fitness:.2f} (Stagnation: {stagnation_counter})")

        # Stop Checks
        if stagnation_counter >= STAGNATION_LIMIT:
            print(f"  [Run {run_id}] Stopping early due to stagnation.")
            break
            
        if current_gbest.num_equal_tt == problem4.truth_table.total_outputs and current_gbest.fitness >= 1000: 
             # Solution found, but we might continue to optimize for gates. 
             # Uncomment break if you want to stop immediately on perfect functional accuracy
             # break 
             pass

    duration = time.perf_counter() - start_time
    
    # Write history to CSV
    for row in run_history:
        csv_writer.writerow(row)
        
    # --- EXTRACTION OF FORMULA ---
    final_gbest = solver.get_gbest()
    print(f"\n[Run {run_id} Result]")
    formula_list = problem4.get_circuit_formula(final_gbest.position)
    
    return {
        "run_id": run_id,
        "is_perfect": (final_gbest.num_equal_tt == problem4.truth_table.total_outputs),
        "fitness": final_gbest.fitness,
        "gates": final_gbest.num_gates,
        "time": duration,
        "formulas": formula_list
    }

# --------------------------------------
# Main Entry Point
# --------------------------------------
if __name__ == "__main__":
    print(f"System: Single-Threaded Standard PSO (Ablation Study)")
    print(f"Total Runs: {NUM_RUNS}")
    print("Population: 500")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    
    f_csv = open(csv_filename, mode='w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["Run_ID", "Iteration", "Best_Fitness", "Correct_Outputs", "Active_Gates"])

    all_stats = []

    for i in range(1, NUM_RUNS + 1):
        stats = execute_single_run(i, writer)
        all_stats.append(stats)
    
    f_csv.close()

    # --------------------------------------
    # Final Evaluation Metrics
    # --------------------------------------
    print("\n" + "="*50)
    print(" FINAL ABLATION METRICS (SINGLE THREAD)")
    print("="*50)

    perfect_runs = [s for s in all_stats if s['is_perfect']]
    success_count = len(perfect_runs)
    success_rate = (success_count / NUM_RUNS) * 100
    print(f"1. Success Rate: {success_count}/{NUM_RUNS} ({success_rate:.1f}%)")

    if perfect_runs:
        gates = [s['gates'] for s in perfect_runs]
        print(f"2. Gate Efficiency (Successful Runs Only):")
        print(f"   - Best (Fewest): {min(gates)}")
        print(f"   - Average:       {statistics.mean(gates):.2f}")
    else:
        print("2. Gate Efficiency: N/A")

    times = [s['time'] for s in all_stats]
    print(f"3. Time Efficiency:")
    print(f"   - Average: {statistics.mean(times):.2f}s")

    print("\nDetailed iteration history saved to:", csv_filename)