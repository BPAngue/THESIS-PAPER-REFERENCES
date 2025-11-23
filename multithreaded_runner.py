import problem4
import pso_solver
import numpy as np
import math
import sys
import os
import copy
from datetime import datetime
import time
import threading
import statistics
import csv

# --------------------------------------
# Configuration
# --------------------------------------
NUM_RUNS = 10               # Total independent runs to perform
NUM_THREADS = 5            # Threads per run
VOTING_INTERVAL = 1        # Iterations per thread before voting
MAX_VOTING_ROUNDS = 100    # Total iterations
STAGNATION_LIMIT = 100      # Stop if no improvement for N rounds

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"pso_multirun_{timestamp}.txt")
csv_filename = os.path.join(results_dir, f"pso_convergence_{timestamp}.csv")

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
    
    phi1, phi2 = 2.05, 1.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    
    params = {
        'w': chi,
        'c1': chi * phi1,
        'c2': chi * phi2,
        'velMax': 0.2 * (problem4.num_rows + problem4.truth_table.num_inputs),
        'velMin': -0.2 * (problem4.num_rows + problem4.truth_table.num_inputs),
        'w_damp': 1.0
    }
    return varSize, params

# --------------------------------------
# Thread Worker
# --------------------------------------
def run_solver_thread(solver, thread_index, interval, results_dict, lock):
    try:
        solver.run_iterations(interval)
        gbest = solver.get_gbest()
        with lock:
            results_dict[thread_index] = gbest
    except Exception as e:
        print(f"Thread Error: {e}")

# --------------------------------------
# Execution Core
# --------------------------------------
def execute_single_run(run_id, csv_writer):
    print(f"\n>>> STARTING RUN {run_id} <<<")
    start_time = time.perf_counter()
    
    # Initialize Population (Fresh start)
    varSize, params = get_pso_params()
    solvers = [pso_solver.BasePSOSolver(500, varSize, params) for _ in range(NUM_THREADS)]
    
    Voted_GBest = pso_solver.Particle(varSize)
    Voted_GBest.fitness = -math.inf
    stagnation_counter = 0
    run_history = []

    for round_num in range(1, MAX_VOTING_ROUNDS + 1):
        # 1. Parallel Execution
        threads = []
        candidates = {} 
        lock = threading.Lock() 

        for i in range(NUM_THREADS):
            t = threading.Thread(target=run_solver_thread, args=(solvers[i], i, VOTING_INTERVAL, candidates, lock))
            threads.append(t)
            t.start()
        for t in threads: t.join()

        # 2. Voting
        if not candidates: continue
        round_best = max(candidates.values(), key=lambda p: p.fitness)

        improved = False
        if round_best.fitness > Voted_GBest.fitness:
            Voted_GBest = round_best
            stagnation_counter = 0
            improved = True
        else:
            stagnation_counter += 1

        # 3. Logging
        if round_num % 10 == 0 or improved:
            print(f"  [Run {run_id}] Iter {round_num}: Fitness {Voted_GBest.fitness:.2f} "
                  f"(Correct: {Voted_GBest.num_equal_tt}/{problem4.truth_table.total_outputs}, Gates: {Voted_GBest.num_gates})")

        run_history.append([run_id, round_num, Voted_GBest.fitness, Voted_GBest.num_equal_tt, Voted_GBest.num_gates])

        # 4. Communication (Ring)
        current_gbests = [solver.get_gbest() for solver in solvers]
        for i in range(NUM_THREADS):
            prev = (i - 1) % NUM_THREADS
            solvers[i].force_gbest_replacement(current_gbests[prev])

        # 5. Stop Checks
        if stagnation_counter >= STAGNATION_LIMIT:
            print(f"  [Run {run_id}] Stopping early due to stagnation.")
            break
            
        if Voted_GBest.num_equal_tt == problem4.truth_table.total_outputs and Voted_GBest.fitness >= 1000: 
             pass 

    duration = time.perf_counter() - start_time
    for row in run_history:
        csv_writer.writerow(row)
        
    # --- EXTRACTION OF FORMULA ---
    print(f"\n[Run {run_id} Result]")
    # We use your problem4 function to get the string representation
    # Note: This function usually prints to stdout, so we might see it twice,
    # but this ensures you see exactly what the particle represents.
    formula_list = problem4.get_circuit_formula(Voted_GBest.position)
    
    return {
        "run_id": run_id,
        "is_perfect": (Voted_GBest.num_equal_tt == problem4.truth_table.total_outputs),
        "fitness": Voted_GBest.fitness,
        "gates": Voted_GBest.num_gates,
        "time": duration,
        "formulas": formula_list # Store formulas for final summary if needed
    }

# --------------------------------------
# Main Entry Point
# --------------------------------------
if __name__ == "__main__":
    print(f"System: Multi-Run PSO ({NUM_RUNS} Runs)")
    print(f"Number of threads: {NUM_THREADS}")
    print("Number of population per thread: 500")
    print(f"Number of iterations: {MAX_VOTING_ROUNDS}")
    print("Phi1 and Phi2: 2.05")
    f_csv = open(csv_filename, mode='w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["Run_ID", "Iteration", "Best_Fitness", "Correct_Outputs", "Active_Gates"])

    all_stats = []

    for i in range(1, NUM_RUNS + 1):
        stats = execute_single_run(i, writer)
        all_stats.append(stats)
        
        # Explicitly print the formula again for clarity in the log
        print(f"  -> Run {i} Formula: {stats['formulas']}")
    
    f_csv.close()

    # --------------------------------------
    # Final Evaluation Metrics
    # --------------------------------------
    print("\n" + "="*50)
    print(" FINAL EVALUATION METRICS")
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
        print(f"   - Std Dev:       {statistics.stdev(gates) if len(gates)>1 else 0:.2f}")
    else:
        print("2. Gate Efficiency: N/A")

    times = [s['time'] for s in all_stats]
    print(f"3. Time Efficiency:")
    print(f"   - Average: {statistics.mean(times):.2f}s")

    print("\nDetailed iteration history saved to:", csv_filename)