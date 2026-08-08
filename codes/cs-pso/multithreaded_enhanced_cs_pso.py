import problem4
import enhanced_cs_pso_solver
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
log_filename = os.path.join(results_dir, f"enhanced_cs_pso_log_{timestamp}.txt")

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
print("[Enhanced Multi-Threaded CS-PSO] System is running...")
print("Using Enhanced Chaotic Particle Swarm Optimization (CS-PSO)")
print("Improvements: Adaptive chaos, mutation, diversity maintenance\n")

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 5            # Number of CS-PSO threads to run in parallel
VOTING_INTERVAL = 5        # Run 5 iterations between votes (was 1)
MAX_VOTING_ROUNDS = 200    # Total rounds (was 100)
STAGNATION_LIMIT = 50      # Stop if Voted_GBest doesn't improve (was 100)

print(f"Pipeline Config: {NUM_THREADS} Threads, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")
print(f"Stagnation Limit: {STAGNATION_LIMIT} rounds")

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

nPop_per_thread = 200  # Reduced from 1000 for faster iterations
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

print(f"\nEnhanced CS-PSO Parameters:")
print(f"  Population per thread: {nPop_per_thread}")
print(f"  Inertia weight (w): {w:.4f}")
print(f"  Cognitive coefficient (c1): {c1:.4f}")
print(f"  Social coefficient (c2): {c2:.4f}")
print(f"  Velocity range: [{velMin:.4f}, {velMax:.4f}]")
print(f"  Inertia damping: {w_damp:.4f}")
print(f"\nEnhancements:")
print(f"  ✓ Adaptive chaos intensity")
print(f"  ✓ Mutation for stuck particles")
print(f"  ✓ Diversity monitoring")
print(f"  ✓ More iterations per round\n")

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
print("Initializing Enhanced CS-PSO solvers with chaotic initialization...")
solvers = [
    enhanced_cs_pso_solver.EnhancedCSPSOSolver(
        nPop_per_thread, 
        varSize, 
        pso_params,
        num_inputs,
        num_rows,
        num_outputs
    ) 
    for _ in range(NUM_THREADS)
]
print("All solvers initialized with adaptive chaos control.\n")

Voted_GBest = enhanced_cs_pso_solver.ChaoticParticle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0
best_fitness_history = []

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n--- Round {round_num + 1} / {MAX_VOTING_ROUNDS} ---")

    threads = []
    gbest_candidates_dict = {} 
    thread_lock = threading.Lock() 

    # 2. Parallel Optimization Phase (with Enhanced Chaos)
    print(f"Running {NUM_THREADS} Enhanced CS-PSO threads ({VOTING_INTERVAL} iter each)...", end=" ")
    start_time = time.perf_counter()
    
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], i, VOTING_INTERVAL, gbest_candidates_dict, thread_lock)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    
    elapsed = time.perf_counter() - start_time
    print(f"Done ({elapsed:.2f}s)")

    # 3. Voting Phase
    if len(gbest_candidates_dict) != NUM_THREADS:
        print("[WARNING] Not all threads reported a GBest. Skipping round.")
        continue

    round_best = max(gbest_candidates_dict.values(), key=lambda p: p.fitness)

    # Get diversity metrics
    avg_diversity = np.mean([s.get_diversity_metric() for s in solvers])
    avg_chaos = np.mean([s.chaos_intensity for s in solvers])
    
    # 4. Check Stopping Conditions
    perfect_score = problem4.truth_table.total_outputs
    
    # Condition 1: Check for perfect solution
    if round_best.num_equal_tt == perfect_score:
        print(f"✓ PERFECT FUNCTIONAL SOLUTION! Fitness: {round_best.fitness:.2f}, Gates: {round_best.num_gates}")
        
        # Keep going to find simpler solution
        if round_best.fitness > Voted_GBest.fitness:
            print(f"  → New best (simpler circuit found)")
            Voted_GBest = round_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
    
    # Condition 2: Check for Voted_GBest improvement
    elif round_best.fitness > Voted_GBest.fitness:
        improvement = round_best.fitness - Voted_GBest.fitness
        print(f"⬆ Champion improved! Fitness: {round_best.fitness:.2f} (+{improvement:.2f})")
        print(f"  Correct: {round_best.num_equal_tt}/{perfect_score}, Gates: {round_best.num_gates}")
        Voted_GBest = round_best
        stagnation_counter = 0 
    else:
        stagnation_counter += 1
        print(f"→ No improvement (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")
    
    # Log metrics
    print(f"  Metrics: Diversity={avg_diversity:.2f}, Chaos={avg_chaos:.3f}")
    best_fitness_history.append(Voted_GBest.fitness)
    
    # Condition 3: Check for stagnation
    if stagnation_counter >= STAGNATION_LIMIT:
        print(f"\n⚠ Stopping: Champion stagnated for {STAGNATION_LIMIT} rounds")
        break
        
    # 5. Feedback Phase (Ring Topology Communication)
    if round_num % 10 == 9:  # Every 10 rounds, show detailed stats
        print(f"  📊 Best so far: {Voted_GBest.fitness:.2f} ({Voted_GBest.num_equal_tt}/{perfect_score} correct)")
    
    # Get all global bests
    current_gbests = [solver.get_gbest() for solver in solvers]

    # Apply Ring Topology: Thread i gets Thread (i-1)'s GBest
    for i in range(NUM_THREADS):
        previous_thread_idx = (i - 1) % NUM_THREADS
        solvers[i].force_gbest_replacement(current_gbests[previous_thread_idx])

# --------------------------------------
# 7. Output Phase
# --------------------------------------
print("\n" + "=" * 80)
print("PIPELINE FINISHED")
print("=" * 80)
print(f"\nTotal rounds completed: {round_num + 1}/{MAX_VOTING_ROUNDS}")

if stagnation_counter >= STAGNATION_LIMIT:
    print(f"Stop reason: Stagnation limit reached ({STAGNATION_LIMIT} rounds)")
else:
    print(f"Stop reason: Maximum rounds reached")

print("\n" + "=" * 80)
print("FINAL CHAMPION PARTICLE (Enhanced CS-PSO)")
print("=" * 80)

final_formulas = problem4.get_circuit_formula(Voted_GBest.position)
circuit_matrix, output_array = problem4.decode_particle(Voted_GBest.position)

print("\nCircuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)

print("\n" + "=" * 80)
print("FINAL STATS")
print("=" * 80)
print(f"Cost (Fitness): {Voted_GBest.fitness:.4f}")
print(f"Correct outputs: {Voted_GBest.num_equal_tt}/{problem4.truth_table.total_outputs}")
print(f"Active Gates Used: {Voted_GBest.num_gates}")
print(f"Unused Gates (Simplicity): {Voted_GBest.num_no_gates}")

if Voted_GBest.num_equal_tt == problem4.truth_table.total_outputs:
    print("\n✓ PERFECT SOLUTION ACHIEVED!")
    if Voted_GBest.num_no_gates > 0:
        print(f"✓ Optimized for simplicity ({Voted_GBest.num_no_gates} unused gates)")
else:
    accuracy = (Voted_GBest.num_equal_tt / problem4.truth_table.total_outputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    missing = problem4.truth_table.total_outputs - Voted_GBest.num_equal_tt
    print(f"Missing: {missing} output(s)")

# Convergence analysis
print("\n" + "=" * 80)
print("CONVERGENCE ANALYSIS")
print("=" * 80)
if best_fitness_history:
    print(f"Initial fitness: {best_fitness_history[0]:.2f}")
    print(f"Final fitness: {best_fitness_history[-1]:.2f}")
    print(f"Total improvement: {best_fitness_history[-1] - best_fitness_history[0]:.2f}")
    
    # Find when first perfect solution was found (if any)
    perfect_rounds = [i for i, f in enumerate(best_fitness_history) 
                     if f >= problem4.truth_table.total_outputs]
    if perfect_rounds:
        print(f"First perfect solution: Round {perfect_rounds[0] + 1}")

print("\n" + "=" * 80)
print("[Enhanced Multi-Threaded CS-PSO] System finished!")
print("=" * 80)

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"\n[TIMER] Total elapsed: {elapsed_seconds:.2f}s")
print(f"[TIMER] Time per round: {elapsed_seconds/(round_num+1):.2f}s")
print(f"[TIMER] Effective iterations: {(round_num+1) * VOTING_INTERVAL * NUM_THREADS}")
