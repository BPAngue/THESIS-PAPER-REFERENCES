import problem4
import pso_solver  # This will import the BPSO solver
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
print("[Multi-Threaded B-PSO] System is running...") # Changed to B-PSO

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 4
VOTING_INTERVAL = 50
MAX_VOTING_ROUNDS = 40
FEEDBACK_PERCENTAGE = 0.10
STAGNATION_LIMIT = 10

print(f"Pipeline Config: {NUM_THREADS} Threads, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")

# --------------------------------------
# PSO Base Solver Parameters
# --------------------------------------

# --- [START] MODIFIED SECTION ---
# We no longer calculate integer varSize here.
# We import the *correct* bit-based size from the solver.
varSize_bits = pso_solver.nVar_bits
print(f"Detected {varSize_bits} bits per particle.")
# --- [END] MODIFIED SECTION ---

nPop_per_thread = 75
constriction_coefficient = True

if not constriction_coefficient:
    w, w_damp, c1, c2 = 1, 0.99, 2, 2
else:
    # Use the exploration-focused parameters
    phi1 = 2.5
    phi2 = 1.5
    print(f"PSO Params: phi1={phi1} (personal), phi2={phi2} (social)")
    
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    w, w_damp, c1, c2 = chi, 1, chi * phi1, chi * phi2

# velMax/velMin are now handled *inside* the BPSO solver,
# so we only pass the core parameters.
pso_params = {'w': w, 'c1': c1, 'c2': c2}

# --------------------------------------
# Thread Execution Function
# --------------------------------------
def run_solver_thread(solver, interval_iterations, results_list, lock):
    try:
        solver.run_iterations(interval_iterations)
        gbest = solver.get_gbest()
        with lock:
            results_list.append(gbest)
    except Exception as e:
        print(f"\n[ERROR] Thread failed: {e}\n")

# --------------------------------------
# System Pipeline Main Loop
# --------------------------------------

# --- [START] MODIFIED SECTION ---
# Pass the correct `varSize_bits` (e.g., 127) to the solver
solvers = [pso_solver.BasePSOSolver(nPop_per_thread, varSize_bits, pso_params) for _ in range(NUM_THREADS)]

# The GBest tracker must also be a BPSO particle of the correct size
Voted_GBest_Tracker = pso_solver.Particle(varSize_bits)
# --- [END] MODIFIED SECTION ---

Voted_GBest_Tracker.fitness = -math.inf
stagnation_counter = 0

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n--- Voting Round {round_num + 1} / {MAX_VOTING_ROUNDS} ---")
    
    threads = []
    gbest_candidates = []
    thread_lock = threading.Lock()

    print(f"Starting {NUM_THREADS} threads for {VOTING_INTERVAL} iterations each...")
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], VOTING_INTERVAL, gbest_candidates, thread_lock)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("All threads finished.")

    if not gbest_candidates:
        print("[WARNING] No GBest candidates found. Skipping round.")
        continue
    
    gbest_candidates.sort(key=lambda p: p.fitness, reverse=True)
    round_best = gbest_candidates[0]
    
    # Check for perfect solution
    perfect_score_check = problem4.truth_table.total_outputs * 1000
    if round_best.num_equal_tt == problem4.truth_table.total_outputs:
        print(f"\n!!! PERFECT FUNCTIONAL SOLUTION FOUND (Fitness: {round_best.fitness:.2f}) !!!")
        if round_best.fitness > Voted_GBest_Tracker.fitness:
             print("New solution is also the simplest found so far. Updating tracker.")
             Voted_GBest_Tracker = round_best
        break 
    
    # Check for stagnation
    if round_best.fitness > Voted_GBest_Tracker.fitness:
        print(f"New Best-Ever solution found! Fitness: {round_best.fitness:.2f} (Correct: {round_best.num_equal_tt}/{problem4.truth_table.total_outputs})")
        Voted_GBest_Tracker = round_best
        stagnation_counter = 0
    else:
        stagnation_counter += 1
        print(f"No new Best-Ever solution. (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")

    if stagnation_counter >= STAGNATION_LIMIT:
        print("Stopping conditions met: Best-Ever solution has stagnated.")
        break
        
    # Feedback Phase (Migration)
    print("Migrating best particles between swarms...")
    current_gbests = [s.get_gbest() for s in solvers]
    
    for i in range(NUM_THREADS):
        particle_to_inject = current_gbests[(i - 1 + NUM_THREADS) % NUM_THREADS]
        print(f"  Injecting Thread {(i - 1 + NUM_THREADS) % NUM_THREADS}'s gbest into Thread {i}'s worst {FEEDBACK_PERCENTAGE*100}%")
        solvers[i].inject_solution(particle_to_inject, FEEDBACK_PERCENTAGE)

# --------------------------------------
# Output Phase
# --------------------------------------
print("\n... Pipeline finished ...\n")
print("--- FINAL VOTED BEST PARTICLE ---")

# --- [START] MODIFIED SECTION ---
# The BPSO particle stores its best *integer* solution in
# the `integer_genes` attribute, not `position`.
final_formulas = problem4.get_circuit_formula(Voted_GBest_Tracker.integer_genes)
circuit_matrix, output_array = problem4.decode_particle(Voted_GBest_Tracker.integer_genes)
# --- [END] MODIFIED SECTION ---

print("\nCircuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)
print("\n--- FINAL STATS ---")
print(f"Cost (Fitness): {Voted_GBest_Tracker.fitness:.4f}")
print(f"Correct outputs: {Voted_GBest_Tracker.num_equal_tt}/{problem4.truth_table.total_outputs}")
print(f"Active Gates Used: {Voted_GBest_Tracker.num_gates}")
print(f"Unused Gates (Simplicity): {Voted_GBest_Tracker.num_no_gates}")

print("\n[Multi-Threaded PSO] System finished!")

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"[TIMER] Elapsed seconds: {elapsed_seconds:.6f}")