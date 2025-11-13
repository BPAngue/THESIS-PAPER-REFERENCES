import problem3
import pso_solver
import numpy as np
import math
import sys
import os
import copy
from datetime import datetime
import time
import threading

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"pso_log_{timestamp}.txt")

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
print(f"[LOGGING ENABLED] Output is being saved to {log_filename}\n")
print("[Multi-Threaded PSO] System is running...")

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 4             # Number of PSO threads to run in parallel
VOTING_INTERVAL = 50        # Run 50 iterations, then vote
MAX_VOTING_ROUNDS = 40      # Total iterations = 50 * 40 = 2000
FEEDBACK_PERCENTAGE = 0.10  # Replace worst 10% of particles
STAGNATION_LIMIT = 5        # Stop if Voted_GBest doesn't improve for 5 rounds

print(f"Pipeline Config: {NUM_THREADS} Threads, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")

# --------------------------------------
# PSO Base Solver Parameters
# --------------------------------------
num_inputs = problem3.truth_table.num_inputs
num_outputs = problem3.truth_table.num_outputs
num_rows = problem3.num_rows
nVar = (num_rows * 3) + num_outputs
varSize = nVar
varMin = 0
varMax = num_inputs + num_rows

nPop_per_thread = 75  # Total particles = 75 * 4 = 300
constriction_coefficient = True

if not constriction_coefficient:
    w, w_damp, c1, c2 = 1, 0.99, 2, 2
else:
    phi1, phi2 = 2.05, 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    w, w_damp, c1, c2 = chi, 1, chi * phi1, chi * phi2

velMax = 0.2 * (varMax - varMin)
velMin = -velMax

pso_params = {'w': w, 'c1': c1, 'c2': c2, 'velMax': velMax, 'velMin': velMin}

# --------------------------------------
# Thread Execution Function
# --------------------------------------
def run_solver_thread(solver, interval_iterations, results_list, lock):
    """
    Function to be executed by each thread.
    Runs the solver and appends its gbest to the results list.
    """
    try:
        solver.run_iterations(interval_iterations)
        gbest = solver.get_gbest()
        
        # Use a lock to safely append to the shared list
        with lock:
            results_list.append(gbest)
    except Exception as e:
        print(f"\n[ERROR] Thread failed: {e}\n")

# --------------------------------------
# System Pipeline Main Loop
# --------------------------------------

# 1. Initialization Phase: Create all solver instances
solvers = [pso_solver.BasePSOSolver(nPop_per_thread, varSize, pso_params) for _ in range(NUM_THREADS)]

Voted_GBest = pso_solver.Particle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n--- Voting Round {round_num + 1} / {MAX_VOTING_ROUNDS} ---")
    
    threads = []
    gbest_candidates = []
    thread_lock = threading.Lock() # Lock for safely appending to candidates list

    # 2. Parallel Optimization Phase
    print(f"Starting {NUM_THREADS} threads for {VOTING_INTERVAL} iterations each...")
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], VOTING_INTERVAL, gbest_candidates, thread_lock)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish their iterations
    for thread in threads:
        thread.join()
    print("All threads finished.")

    # 3. Voting Phase
    if not gbest_candidates:
        print("[WARNING] No GBest candidates found. Skipping round.")
        continue

    # "Fitness-Based Ranked Voting": Find the best particle from the candidates
    round_best = max(gbest_candidates, key=lambda p: p.fitness)
    
    # 4. Check Stopping Conditions
    
    # Condition 1: Check for perfect solution
    perfect_score = problem3.truth_table.total_outputs
    if round_best.num_equal_tt == perfect_score:
        print(f"\n!!! PERFECT FUNCTIONAL SOLUTION FOUND (Fitness: {round_best.fitness:.2f}) !!!")
        
        # Now, check if it's *also* better or simpler than our Voted_GBest
        if round_best.fitness > Voted_GBest.fitness:
             print("New solution is also the simplest found so far. Updating Voted_GBest.")
             Voted_GBest = round_best
        
        # We can now stop, as we have a perfect solution
        break 
    
    # Condition 2: Check for Voted_GBest improvement
    if round_best.fitness > Voted_GBest.fitness:
        print(f"New Voted_GBest found! Fitness: {round_best.fitness:.2f} (Correct: {round_best.num_equal_tt}/{perfect_score})")
        Voted_GBest = round_best
        stagnation_counter = 0 # Reset stagnation
    else:
        stagnation_counter += 1
        print(f"Voted_GBest did not improve. (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")

    # Condition 3: Check for stagnation
    if stagnation_counter >= STAGNATION_LIMIT:
        print("Stopping conditions met: Voted_GBest has stagnated.")
        break
        
    # 5. Feedback Phase
    print("Feeding Voted_GBest back to all swarms...")
    for solver in solvers:
        solver.inject_solution(Voted_GBest, FEEDBACK_PERCENTAGE)

# --------------------------------------
# 7. Output Phase
# --------------------------------------
print("\n... Pipeline finished ...\n")
print("--- FINAL VOTED BEST PARTICLE ---")

# Get formulas and decoded matrices for the final report
final_formulas = problem3.get_circuit_formula(Voted_GBest.position)
circuit_matrix, output_array = problem3.decode_particle(Voted_GBest.position)

print("\nCircuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)
print("\n--- FINAL STATS ---")
print(f"Cost (Fitness): {Voted_GBest.fitness:.4f}")
print(f"Correct outputs: {Voted_GBest.num_equal_tt}/{problem3.truth_table.total_outputs}")
print(f"Active Gates Used: {Voted_GBest.num_gates}")
print(f"Unused Gates (Simplicity): {Voted_GBest.num_no_gates}")

print("\n[Multi-Threaded PSO] System finished!")

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"[TIMER] Elapsed seconds: {elapsed_seconds:.6f}")