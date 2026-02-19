import problem4
import cs_pso_solver
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
log_filename = os.path.join(results_dir, f"cs_pso_log_{timestamp}.txt")

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
print("[Multi-Threaded CS-PSO] System is running...")
print("Using Chaotic Particle Swarm Optimization (CS-PSO)")
print("Based on: Xu et al. (2018) - CS-PSO for combinatorial optimization\n")

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 5            # Number of CS-PSO threads to run in parallel
VOTING_INTERVAL = 1        # Run 1 iterations, then vote
MAX_VOTING_ROUNDS = 100    # Total iterations = 100
STAGNATION_LIMIT = 100     # Stop if Voted_GBest doesn't improve

print(f"Pipeline Config: {NUM_THREADS} Threads, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")

# --------------------------------------
# PSO Base Solver Parameters
# --------------------------------------
num_inputs = problem4.truth_table.num_inputs
num_outputs = problem4.truth_table.num_outputs
num_rows = problem4.num_rows
nVar = (num_rows * 3) + num_outputs
varSize = nVar
varMin = 0
varMax = num_inputs + num_rows

nPop_per_thread = 1000
constriction_coefficient = True

if not constriction_coefficient:
    w, w_damp, c1, c2 = 1, 0.99, 2, 2
else:
    phi1, phi2 = 1.05, 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    w, w_damp, c1, c2 = chi, 1, chi * phi1, chi * phi2

velMax = 0.2 * (varMax - varMin)
velMin = -velMax

pso_params = {'w': w, 'c1': c1, 'c2': c2, 'velMax': velMax, 'velMin': velMin, 'w_damp': w_damp}

print(f"\nCS-PSO Parameters:")
print(f"  Population per thread: {nPop_per_thread}")
print(f"  Inertia weight (w): {w:.4f}")
print(f"  Cognitive coefficient (c1): {c1:.4f}")
print(f"  Social coefficient (c2): {c2:.4f}")
print(f"  Velocity range: [{velMin:.4f}, {velMax:.4f}]")
print(f"  Inertia damping: {w_damp:.4f}\n")

# --------------------------------------
# Thread Execution Function
# --------------------------------------
def run_solver_thread(solver, thread_index, interval_iterations, results_dict, lock):
    """
    Function to be executed by each thread.
    Runs the CS-PSO solver and places its gbest in the results_dict using its index.
    """
    try:
        solver.run_iterations(interval_iterations)
        gbest = solver.get_gbest()
        
        # Use a lock to safely write to the shared dictionary
        with lock:
            results_dict[thread_index] = gbest
    except Exception as e:
        print(f"\n[ERROR] Thread {thread_index} failed: {e}\n")


# --------------------------------------
# System Pipeline Main Loop
# --------------------------------------
# 1. Initialization Phase
print("Initializing CS-PSO solvers with chaotic initialization...")
solvers = [
    cs_pso_solver.CSPSOSolver(
        nPop_per_thread, 
        varSize, 
        pso_params,
        num_inputs,
        num_rows,
        num_outputs
    ) 
    for _ in range(NUM_THREADS)
]
print("All solvers initialized using logistic map for enhanced diversity.\n")

Voted_GBest = cs_pso_solver.ChaoticParticle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n--- Iteration {round_num + 1} / {MAX_VOTING_ROUNDS} ---")

    threads = []
    gbest_candidates_dict = {} 
    thread_lock = threading.Lock() 

    # 2. Parallel Optimization Phase (with Chaos Perturbation)
    print(f"Starting {NUM_THREADS} CS-PSO threads for {VOTING_INTERVAL} iterations each...")
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], i, VOTING_INTERVAL, gbest_candidates_dict, thread_lock)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("All threads finished.")

    # 3. Voting Phase
    if len(gbest_candidates_dict) != NUM_THREADS:
        print("[WARNING] Not all threads reported a GBest. Skipping round.")
        continue

    round_best = max(gbest_candidates_dict.values(), key=lambda p: p.fitness)

    # 4. Check Stopping Conditions
    
    # Condition 1: Check for perfect solution
    perfect_score = problem4.truth_table.total_outputs
    if round_best.num_equal_tt == perfect_score:
        print(f"\n!!! PERFECT FUNCTIONAL SOLUTION FOUND (Fitness: {round_best.fitness:.2f}) !!!")
        
        if round_best.fitness > Voted_GBest.fitness:
            print("New solution is also the simplest found so far. Updating Voted_GBest.")
    
    # Condition 2: Check for Voted_GBest improvement
    if round_best.fitness > Voted_GBest.fitness:
        print(f"New Overall Champion found! Fitness: {round_best.fitness:.2f} (Correct: {round_best.num_equal_tt}/{perfect_score}, Gates: {round_best.num_gates})")
        Voted_GBest = round_best
        stagnation_counter = 0 
    else:
        stagnation_counter += 1
        print(f"Overall Champion did not improve. (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")

    # Condition 3: Check for stagnation
    if stagnation_counter >= STAGNATION_LIMIT:
        print("\nStopping conditions met: Overall Champion has stagnated.")
        break
        
    # 5. Feedback Phase (Ring Topology Communication)
    print("Performing Ring Topology communication (Influence model)...")
    # Get all global bests
    current_gbests = [solver.get_gbest() for solver in solvers]

    # Apply Circular Logic: Thread i gets Thread (i-1)'s GBest
    for i in range(NUM_THREADS):
        previous_thread_idx = (i - 1) % NUM_THREADS
        
        # FORCE replacement (no if check)
        solvers[i].force_gbest_replacement(current_gbests[previous_thread_idx])

# --------------------------------------
# 7. Output Phase
# --------------------------------------
print("\n... Pipeline finished ...\n")
print("=" * 60)
print("FINAL CHAMPION PARTICLE (CS-PSO)")
print("=" * 60)

final_formulas = problem4.get_circuit_formula(Voted_GBest.position)
circuit_matrix, output_array = problem4.decode_particle(Voted_GBest.position)

print("\nCircuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)
print("\n" + "=" * 60)
print("FINAL STATS")
print("=" * 60)
print(f"Cost (Fitness): {Voted_GBest.fitness:.4f}")

print(f"Correct outputs: {Voted_GBest.num_equal_tt}/{problem4.truth_table.total_outputs}")
print(f"Active Gates Used: {Voted_GBest.num_gates}")
print(f"Unused Gates (Simplicity): {Voted_GBest.num_no_gates}")

if Voted_GBest.num_equal_tt == problem4.truth_table.total_outputs:
    print("\n✓ PERFECT SOLUTION ACHIEVED!")
else:
    accuracy = (Voted_GBest.num_equal_tt / problem4.truth_table.total_outputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

print("\n" + "=" * 60)
print("[Multi-Threaded CS-PSO] System finished!")
print("=" * 60)

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"\n[TIMER] Elapsed seconds: {elapsed_seconds:.6f}")
print(f"[TIMER] Time per round: {elapsed_seconds/max(1, round_num+1):.6f} seconds")
