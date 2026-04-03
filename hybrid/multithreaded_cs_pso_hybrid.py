import problem_def
import hybrid_cs_pso_solver
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
log_filename = os.path.join(results_dir, f"hybrid_cs_pso_log_{timestamp}.txt")

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
print("=" * 80)
print("HYBRID CS-PSO WITH LOCAL HILL CLIMBING")
print("=" * 80)
print("\nCombining global exploration (CS-PSO) with local exploitation (Hill Climbing)")
print("\nKey Features:")
print("  🌍 Global Search: Chaotic PSO for exploration")
print("  🎯 Local Search: Hill climbing when close to solution (≥14/16)")
print("  ⚡ Adaptive: Chaos intensity adapts to search progress")
print("  🔄 Hybrid: Best of both worlds!\n")

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 5
VOTING_INTERVAL = 5
MAX_VOTING_ROUNDS = 200
STAGNATION_LIMIT = 50

print(f"Pipeline Config:")
print(f"  Threads: {NUM_THREADS}")
print(f"  Iterations per round: {VOTING_INTERVAL}")
print(f"  Max rounds: {MAX_VOTING_ROUNDS}")
print(f"  Stagnation limit: {STAGNATION_LIMIT}\n")

# --------------------------------------
# PSO Parameters
# --------------------------------------
num_inputs = problem_def.truth_table.num_inputs
num_outputs = problem_def.truth_table.num_outputs
num_rows = problem_def.num_rows
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

print(f"Hybrid CS-PSO Parameters:")
print(f"  Population per thread: {nPop_per_thread}")
print(f"  PSO inertia (w): {w:.4f}")
print(f"  Cognitive (c1): {c1:.4f}")
print(f"  Social (c2): {c2:.4f}")
print(f"\nHill Climbing Parameters:")
print(f"  Activation threshold: ≥14/{problem_def.truth_table.total_outputs} correct outputs")
print(f"  Max iterations per climb: 50 (gbest), 20 (others)")
print(f"  Frequency: Every 5 PSO iterations")
print(f"  Strategies: Output-focused, Gate-focused, Comprehensive\n")

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
print("Initializing Hybrid CS-PSO solvers...")
solvers = [
    hybrid_cs_pso_solver.HybridCSPSOSolver(
        nPop_per_thread, 
        varSize, 
        pso_params,
        num_inputs,
        num_rows,
        num_outputs
    ) 
    for _ in range(NUM_THREADS)
]
print("All solvers initialized with hybrid search capability.\n")

Voted_GBest = hybrid_cs_pso_solver.ChaoticParticle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0
best_fitness_history = []
total_hill_climbs = 0
successful_hill_climbs = 0

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n{'=' * 80}")
    print(f"Round {round_num + 1} / {MAX_VOTING_ROUNDS}")
    print(f"{'=' * 80}")

    threads = []
    gbest_candidates_dict = {} 
    thread_lock = threading.Lock() 

    # Parallel Optimization Phase
    print(f"Running {NUM_THREADS} Hybrid CS-PSO threads ({VOTING_INTERVAL} iter each)...", end=" ")
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

    # Collect statistics from all solvers
    all_stats = [s.get_stats() for s in solvers]
    avg_diversity = np.mean([s['diversity'] for s in all_stats])
    avg_chaos = np.mean([s['chaos_intensity'] for s in all_stats])
    avg_stagnation = np.mean([s['global_stagnation'] for s in all_stats])
    
    # Hill climbing statistics
    round_hill_climbs = sum([s['total_hill_climbs'] for s in all_stats])
    round_successful_climbs = sum([s['successful_hill_climbs'] for s in all_stats])
    total_hill_climbs = round_hill_climbs
    successful_hill_climbs = round_successful_climbs
    
    # Check Stopping Conditions
    perfect_score = problem_def.truth_table.total_outputs
    
    # Perfect solution found
    if round_best.num_equal_tt == perfect_score:
        print(f"✅ PERFECT FUNCTIONAL SOLUTION!")
        print(f"   Fitness: {round_best.fitness:.2f}")
        print(f"   Gates: {round_best.num_gates}")
        print(f"   Unused gates: {round_best.num_no_gates}")
        
        if round_best.fitness > Voted_GBest.fitness:
            print(f"   🎯 New champion (simpler circuit)")
            Voted_GBest = round_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
    
    # Improvement found
    elif round_best.fitness > Voted_GBest.fitness:
        improvement = round_best.fitness - Voted_GBest.fitness
        print(f"⬆ Champion improved!")
        print(f"   Fitness: {round_best.fitness:.2f} (+{improvement:.2f})")
        print(f"   Correct: {round_best.num_equal_tt}/{perfect_score}")
        print(f"   Gates: {round_best.num_gates}")
        Voted_GBest = round_best
        stagnation_counter = 0 
    else:
        stagnation_counter += 1
        print(f"→ No improvement (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")
    
    # Display metrics
    print(f"\n📊 Metrics:")
    print(f"   Diversity: {avg_diversity:.2f}")
    print(f"   Chaos: {avg_chaos:.3f}", end="")
    if avg_chaos > 0.7:
        print(f" 🔥 HIGH CHAOS MODE!")
    elif avg_chaos > 0.5:
        print(f" ⚡ Moderate chaos")
    else:
        print(f" 🌱 Low chaos (exploiting)")
    
    print(f"   Global stagnation: {avg_stagnation:.1f}")
    
    # Hill climbing statistics
    if total_hill_climbs > 0:
        success_rate = (successful_hill_climbs / total_hill_climbs) * 100
        print(f"\n🎯 Hill Climbing Stats:")
        print(f"   Total climbs: {total_hill_climbs}")
        print(f"   Successful: {successful_hill_climbs} ({success_rate:.1f}%)")
        if round_hill_climbs > 0:
            print(f"   This round: {round_hill_climbs} climbs")
    
    best_fitness_history.append(Voted_GBest.fitness)
    
    # Check for stagnation
    if stagnation_counter >= STAGNATION_LIMIT:
        print(f"\n⚠ Stopping: Champion stagnated for {STAGNATION_LIMIT} rounds")
        break
        
    # Detailed stats every 10 rounds
    if round_num % 10 == 9:
        print(f"\n📈 Progress Summary:")
        print(f"   Best fitness: {Voted_GBest.fitness:.2f}")
        print(f"   Correct outputs: {Voted_GBest.num_equal_tt}/{perfect_score}")
        print(f"   Active gates: {Voted_GBest.num_gates}")
    
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
print("FINAL CHAMPION PARTICLE (Hybrid CS-PSO)")
print("=" * 80)

final_formulas = problem_def.get_circuit_formula(Voted_GBest.position)
circuit_matrix, output_array = problem_def.decode_particle(Voted_GBest.position)

print("\nCircuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)

print("\n" + "=" * 80)
print("FINAL STATS")
print("=" * 80)
print(f"Cost (Fitness): {Voted_GBest.fitness:.4f}")
print(f"Correct outputs: {Voted_GBest.num_equal_tt}/{problem_def.truth_table.total_outputs}")
print(f"Active Gates Used: {Voted_GBest.num_gates}")
print(f"Unused Gates (Simplicity): {Voted_GBest.num_no_gates}")

if Voted_GBest.num_equal_tt == problem_def.truth_table.total_outputs:
    print("\n✅ PERFECT SOLUTION ACHIEVED!")
    if Voted_GBest.num_no_gates > 0:
        print(f"✓ Optimized for simplicity ({Voted_GBest.num_no_gates} unused gates)")
else:
    accuracy = (Voted_GBest.num_equal_tt / problem_def.truth_table.total_outputs) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    missing = problem_def.truth_table.total_outputs - Voted_GBest.num_equal_tt
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
                     if f >= problem_def.truth_table.total_outputs]
    if perfect_rounds:
        print(f"\n✓ First perfect solution: Round {perfect_rounds[0] + 1}")
        print(f"✓ Total perfect rounds: {len(perfect_rounds)}")
    else:
        print(f"\n✗ No perfect solution found")
        print(f"  Best achieved: {best_fitness_history[-1]:.2f}/{problem_def.truth_table.total_outputs}")

# Hill climbing analysis
print("\n" + "=" * 80)
print("HILL CLIMBING ANALYSIS")
print("=" * 80)
if total_hill_climbs > 0:
    success_rate = (successful_hill_climbs / total_hill_climbs) * 100
    print(f"Total hill climbs: {total_hill_climbs}")
    print(f"Successful climbs: {successful_hill_climbs}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate > 50:
        print("\n✓ Hill climbing was very effective!")
    elif success_rate > 20:
        print("\n→ Hill climbing provided moderate improvements")
    else:
        print("\n✗ Hill climbing struggled (solution may be in basin of attraction)")
else:
    print("No hill climbing was triggered")
    print("(Threshold not reached: need ≥14/16 correct)")

print("\n" + "=" * 80)
print("[Hybrid CS-PSO] System finished!")
print("=" * 80)

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"\n[TIMER] Total elapsed: {elapsed_seconds:.2f}s ({elapsed_seconds/60:.1f} min)")
print(f"[TIMER] Time per round: {elapsed_seconds/(round_num+1):.2f}s")
print(f"[TIMER] Effective iterations: {(round_num+1) * VOTING_INTERVAL * NUM_THREADS}")