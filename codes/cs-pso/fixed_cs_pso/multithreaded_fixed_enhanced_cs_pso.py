import problem4
import fixed_enhanced_cs_pso_solver
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
log_filename = os.path.join(results_dir, f"fixed_enhanced_cs_pso_log_{timestamp}.txt")

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
print("[FIXED Enhanced Multi-Threaded CS-PSO] System is running...")
print("Using FIXED Enhanced Chaotic Particle Swarm Optimization")
print("🔧 BUG FIX: Adaptive chaos now works correctly!")
print("Improvements: Properly accumulating stagnation, adaptive chaos, mutation\n")

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 5
VOTING_INTERVAL = 5
MAX_VOTING_ROUNDS = 200
STAGNATION_LIMIT = 50

print(f"Pipeline Config: {NUM_THREADS} Threads, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")
print(f"Stagnation Limit: {STAGNATION_LIMIT} rounds")

# --------------------------------------
# PSO Parameters
# --------------------------------------
num_inputs = problem4.truth_table.num_inputs
num_outputs = problem4.truth_table.num_outputs
num_rows = problem4.num_rows
nVar = (num_rows * 3) + num_outputs
varSize = nVar
varMin = 0
varMax = num_inputs + num_rows

nPop_per_thread = 100
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

print(f"\nFIXED Enhanced CS-PSO Parameters:")
print(f"  Population per thread: {nPop_per_thread}")
print(f"  Inertia weight (w): {w:.4f}")
print(f"  Cognitive coefficient (c1): {c1:.4f}")
print(f"  Social coefficient (c2): {c2:.4f}")
print(f"  Velocity range: [{velMin:.4f}, {velMax:.4f}]")
print(f"\n🔧 KEY FIXES:")
print(f"  ✓ Global stagnation accumulates properly")
print(f"  ✓ Chaos intensity starts at 0.5 (was 0.3)")
print(f"  ✓ Mutation threshold lowered to 3 (was 5)")
print(f"  ✓ Adaptive chaos triggers at: 5, 10, 20+ rounds")
print(f"\n✨ ENHANCEMENTS:")
print(f"  ✓ Adaptive chaos intensity (0.4-1.0)")
print(f"  ✓ Mutation for stuck particles")
print(f"  ✓ Diversity monitoring")
print(f"  ✓ More iterations per round\n")

# --------------------------------------
# Thread Execution Function
# --------------------------------------
def run_solver_thread(solver, thread_index, interval_iterations, results_dict, lock):
    """Execute a single solver thread."""
    try:
        solver.run_iterations(interval_iterations)
        gbest = solver.get_gbest()
        
        with lock:
            results_dict[thread_index] = gbest
    except Exception as e:
        print(f"\n[ERROR] Thread {thread_index} failed: {e}\n")

# --------------------------------------
# System Pipeline Main Loop
# --------------------------------------
print("Initializing FIXED Enhanced CS-PSO solvers...")
solvers = [
    fixed_enhanced_cs_pso_solver.FixedEnhancedCSPSOSolver(
        nPop_per_thread, 
        varSize, 
        pso_params,
        num_inputs,
        num_rows,
        num_outputs
    ) 
    for _ in range(NUM_THREADS)
]
print("All solvers initialized with corrected adaptive chaos control.\n")

Voted_GBest = fixed_enhanced_cs_pso_solver.ChaoticParticle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0
best_fitness_history = []

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n--- Round {round_num + 1} / {MAX_VOTING_ROUNDS} ---")

    threads = []
    gbest_candidates_dict = {} 
    thread_lock = threading.Lock() 

    # Parallel Optimization Phase
    print(f"Running {NUM_THREADS} FIXED CS-PSO threads ({VOTING_INTERVAL} iter each)...", end=" ")
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

    # Voting Phase
    if len(gbest_candidates_dict) != NUM_THREADS:
        print("[WARNING] Not all threads reported a GBest. Skipping round.")
        continue

    round_best = max(gbest_candidates_dict.values(), key=lambda p: p.fitness)

    # Get metrics from all solvers
    avg_diversity = np.mean([s.get_diversity_metric() for s in solvers])
    avg_chaos = np.mean([s.chaos_intensity for s in solvers])
    avg_stagnation = np.mean([s.global_stagnation for s in solvers])
    
    # Check Stopping Conditions
    perfect_score = problem4.truth_table.total_outputs
    
    # Perfect solution found
    if round_best.num_equal_tt == perfect_score:
        print(f"✓ PERFECT FUNCTIONAL SOLUTION! Fitness: {round_best.fitness:.2f}, Gates: {round_best.num_gates}")
        
        if round_best.fitness > Voted_GBest.fitness:
            print(f"  → New best (simpler circuit found)")
            Voted_GBest = round_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
    
    # Improvement found
    elif round_best.fitness > Voted_GBest.fitness:
        improvement = round_best.fitness - Voted_GBest.fitness
        print(f"⬆ Champion improved! Fitness: {round_best.fitness:.2f} (+{improvement:.2f})")
        print(f"  Correct: {round_best.num_equal_tt}/{perfect_score}, Gates: {round_best.num_gates}")
        Voted_GBest = round_best
        stagnation_counter = 0 
    else:
        stagnation_counter += 1
        print(f"→ No improvement (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")
    
    # Log detailed metrics
    print(f"  Metrics: Diversity={avg_diversity:.2f}, Chaos={avg_chaos:.3f}, GlobStag={avg_stagnation:.1f}")
    best_fitness_history.append(Voted_GBest.fitness)
    
    # Show chaos activation status
    if avg_chaos > 0.7:
        print(f"  🔥 HIGH CHAOS MODE ACTIVE! (Breaking out of local optimum)")
    elif avg_chaos > 0.5:
        print(f"  ⚡ Moderate chaos exploration")
    
    # Check for stagnation
    if stagnation_counter >= STAGNATION_LIMIT:
        print(f"\n⚠ Stopping: Champion stagnated for {STAGNATION_LIMIT} rounds")
        break
        
    # Feedback Phase - Every 10 rounds show detailed stats
    if round_num % 10 == 9:
        print(f"  📊 Best so far: {Voted_GBest.fitness:.2f} ({Voted_GBest.num_equal_tt}/{perfect_score} correct)")
        print(f"  📈 Chaos levels: {[f'{s.chaos_intensity:.2f}' for s in solvers]}")
    
    # Ring Topology Communication
    current_gbests = [solver.get_gbest() for solver in solvers]
    for i in range(NUM_THREADS):
        previous_thread_idx = (i - 1) % NUM_THREADS
        solvers[i].force_gbest_replacement(current_gbests[previous_thread_idx])

# --------------------------------------
# Output Phase
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
print("FINAL CHAMPION PARTICLE (FIXED Enhanced CS-PSO)")
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
    print("\n✅ PERFECT SOLUTION ACHIEVED!")
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
    
    perfect_rounds = [i for i, f in enumerate(best_fitness_history) 
                     if f >= problem4.truth_table.total_outputs]
    if perfect_rounds:
        print(f"✓ First perfect solution: Round {perfect_rounds[0] + 1}")
        print(f"✓ Total perfect rounds: {len(perfect_rounds)}")
    else:
        print(f"✗ No perfect solution found")
        print(f"  Best achieved: {best_fitness_history[-1]:.2f}/{problem4.truth_table.total_outputs}")

print("\n" + "=" * 80)
print("[FIXED Enhanced Multi-Threaded CS-PSO] System finished!")
print("=" * 80)

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"\n[TIMER] Total elapsed: {elapsed_seconds:.2f}s")
print(f"[TIMER] Time per round: {elapsed_seconds/(round_num+1):.2f}s")
print(f"[TIMER] Effective iterations: {(round_num+1) * VOTING_INTERVAL * NUM_THREADS}")
