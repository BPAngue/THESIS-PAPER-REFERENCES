from datetime import datetime
from mo_pso_solver import MOPSOSolver, ParetoArchive, MOParticle
import problem_def as problem_def
import numpy as np
import math
import sys
import os
import time
import threading
from collections import deque

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"mopso_adaptive_{timestamp}.txt")

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
print("[Adaptive Multi-Objective PSO] System starting...\n")

_perf_start = time.perf_counter()

# --------------------------------------
# Configuration Parameters
# --------------------------------------
NUM_THREADS = 5
VOTING_INTERVAL = 1  # Iterations per round
MAX_VOTING_ROUNDS = 500
STAGNATION_LIMIT = 20  # Increased for better exploration

# Archive and diversity parameters
GLOBAL_ARCHIVE_SIZE = 200
LOCAL_ARCHIVE_SIZE = 100

# Early stopping
PERFECT_SOLUTION_PATIENCE = 5  # Stop after finding perfect solution this many times

# --------------------------------------
# Problem Configuration
# --------------------------------------
num_inputs = problem_def.truth_table.num_inputs
num_outputs = problem_def.truth_table.num_outputs
num_rows = problem_def.num_rows
nVar = (num_rows * 3) + num_outputs
varSize = nVar

nPop_per_thread = 200

# PSO Parameters - FIXED CALCULATION
# Use standard PSO parameters instead of constriction coefficient
# to avoid the sqrt issue
USE_CONSTRICTION = False

if USE_CONSTRICTION:
    # Constriction coefficient (only if phi values are valid)
    phi1, phi2 = 2.05, 2.05
    phi = phi1 + phi2
    # Check if calculation will be valid
    if phi > 4:
        chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi)))
        w, w_damp = chi, 0.99
        c1_start, c2_start = chi * phi1, chi * phi2
    else:
        # Fallback to standard PSO
        w, w_damp = 0.729, 0.99
        c1_start, c2_start = 1.49445, 1.49445
else:
    # Standard PSO parameters (more stable)
    w, w_damp = 0.729, 0.99
    c1_start, c2_start = 1.49445, 1.49445

varMin = 0
varMax = num_inputs + num_rows
velMax = 0.2 * (varMax - varMin)
velMin = -velMax

pso_params = {
    'w': w,
    'w_damp': w_damp,
    'c1_start': c1_start,
    'c2_start': c2_start,
    'velMax': velMax,
    'velMin': velMin,
    'archive_size': LOCAL_ARCHIVE_SIZE,
    'mutation_rate': 0.3
}

print(f"Configuration:")
print(f"  Threads: {NUM_THREADS}")
print(f"  Population per thread: {nPop_per_thread}")
print(f"  PSO params: w={w:.4f}, c1={c1_start:.4f}, c2={c2_start:.4f}")
print(f"  Iterations per round: {VOTING_INTERVAL}")
print(f"  Max rounds: {MAX_VOTING_ROUNDS}")
print(f"  Stagnation limit: {STAGNATION_LIMIT}")
print(f"  Global archive size: {GLOBAL_ARCHIVE_SIZE}")
print(f"  Local archive size: {LOCAL_ARCHIVE_SIZE}\n")

# --------------------------------------
# Global Archive and Diversity Tracker
# --------------------------------------
global_archive = ParetoArchive(max_size=GLOBAL_ARCHIVE_SIZE)
perfect_solution_count = 0
stagnation_counter = 0
best_correctness_history = deque(maxlen=STAGNATION_LIMIT)

# --------------------------------------
# Thread Execution
# --------------------------------------
def run_solver_thread(solver, thread_idx, iterations, round_num, max_rounds, results_dict, lock):
    """Execute solver thread"""
    try:
        current_gen = round_num * iterations
        max_gen = max_rounds * iterations
        solver.run_iterations(iterations, current_gen, max_gen)
        
        with lock:
            results_dict[thread_idx] = solver.archive.archive.copy()
    except Exception as e:
        print(f"[ERROR] Thread {thread_idx} failed: {e}")
        import traceback
        traceback.print_exc()

# --------------------------------------
# Diversity Injection Strategies
# --------------------------------------
def inject_diversity_to_solver(solver, strategy='mixed'):
    """Inject diversity using various strategies"""
    n_inject = max(1, int(0.2 * solver.nPop))  # Replace 20% of population
    
    # Sort by crowding distance (keep diverse solutions)
    solver.pop.sort(key=lambda p: -p.crowding_distance if hasattr(p, 'crowding_distance') else 0)
    
    if strategy == 'random':
        # Replace worst with random
        for i in range(solver.nPop - n_inject, solver.nPop):
            solver.pop[i] = MOParticle(solver.varSize)
    
    elif strategy == 'guided':
        # Replace worst with solutions guided by best in archive
        if len(solver.archive.archive) > 0:
            best = solver.archive.get_best_for_objective('correctness')
            for i in range(solver.nPop - n_inject, solver.nPop):
                new_particle = MOParticle(solver.varSize)
                # Blend with best
                alpha = np.random.rand()
                new_particle.position = (alpha * best.position + 
                                       (1 - alpha) * new_particle.position)
                new_particle.position = solver._clamp_position(new_particle.position)
                solver.pop[i] = new_particle
    
    elif strategy == 'mixed':
        # Mix of random and guided
        half = n_inject // 2
        # Random half
        for i in range(solver.nPop - n_inject, solver.nPop - half):
            solver.pop[i] = MOParticle(solver.varSize)
        # Guided half
        if len(solver.archive.archive) > 0:
            best = solver.archive.get_best_for_objective('correctness')
            for i in range(solver.nPop - half, solver.nPop):
                new_particle = MOParticle(solver.varSize)
                alpha = np.random.uniform(0.3, 0.7)
                new_particle.position = (alpha * best.position + 
                                       (1 - alpha) * new_particle.position)
                new_particle.position = solver._clamp_position(new_particle.position)
                solver.pop[i] = new_particle
    
    # Re-evaluate population
    solver.evaluate_population()
    
    # Reset inertia to encourage exploration
    solver.w = 0.85

# --------------------------------------
# Archive Sharing Mechanism
# --------------------------------------
def share_archives_ring_topology(solvers):
    """Share archives in ring topology with selective migration"""
    n_solvers = len(solvers)
    n_migrate = 3  # Number of solutions to share
    
    for i in range(n_solvers):
        prev_idx = (i - 1) % n_solvers
        
        # Get best solutions from previous solver
        prev_archive = solvers[prev_idx].archive.archive
        if len(prev_archive) == 0:
            continue
        
        # Select diverse solutions to migrate
        # Sort by crowding distance and take top N
        prev_archive_sorted = sorted(prev_archive, 
                                    key=lambda p: -p.crowding_distance)
        migrants = prev_archive_sorted[:min(n_migrate, len(prev_archive_sorted))]
        
        # Add to current solver's archive
        for migrant in migrants:
            solvers[i].archive.add(migrant)

def share_with_global_archive(solvers, global_archive):
    """Bidirectional sharing with global archive"""
    # Collect all solutions from local archives
    for solver in solvers:
        for particle in solver.archive.archive:
            global_archive.add(particle)
    
    # Inject best global solutions back to threads
    if len(global_archive.archive) > 0:
        # Get top solutions from global archive
        global_archive._calculate_crowding_distances()
        top_global = sorted(global_archive.archive, 
                          key=lambda p: -p.crowding_distance)[:min(5, len(global_archive.archive))]
        
        for solver in solvers:
            for particle in top_global:
                solver.archive.add(particle)

# --------------------------------------
# Main Pipeline
# --------------------------------------
print("Initializing solvers...\n")
solvers = [MOPSOSolver(nPop_per_thread, varSize, pso_params) 
           for _ in range(NUM_THREADS)]

# Initial evaluation
for solver in solvers:
    for particle in solver.archive.archive:
        global_archive.add(particle)

perfect_score = problem_def.truth_table.total_outputs
best_ever_correctness = 0
consecutive_perfect_rounds = 0

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n{'='*70}")
    print(f"Round {round_num + 1}/{MAX_VOTING_ROUNDS}")
    print(f"{'='*70}")
    
    # Parallel optimization
    threads = []
    results_dict = {}
    thread_lock = threading.Lock()
    
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], i, VOTING_INTERVAL, round_num, MAX_VOTING_ROUNDS,
                  results_dict, thread_lock)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Update global archive
    share_with_global_archive(solvers, global_archive)
    
    # Get best solution from global archive
    best_solution = global_archive.get_best_for_objective('correctness')
    
    if best_solution:
        current_correctness = best_solution.obj1
        current_gates = best_solution.obj2
        
        print(f"\nCurrent Best:")
        print(f"  Correctness: {int(current_correctness)}/{perfect_score} "
              f"({100*current_correctness/perfect_score:.1f}%)")
        print(f"  Active gates: {int(current_gates)}/{num_rows}")
        print(f"  Archive size: {len(global_archive.archive)}")
        
        # Track improvement
        if current_correctness > best_ever_correctness:
            best_ever_correctness = current_correctness
            stagnation_counter = 0
            consecutive_perfect_rounds = 0
            print(f"  ✓ New best correctness!")
        else:
            stagnation_counter += 1
            print(f"  Stagnation: {stagnation_counter}/{STAGNATION_LIMIT}")
        
        # Check for perfect solution
        if current_correctness == perfect_score:
            consecutive_perfect_rounds += 1
            print(f"\n  ★★★ PERFECT SOLUTION FOUND! ★★★")
            print(f"  Consecutive perfect rounds: {consecutive_perfect_rounds}")
            
            if consecutive_perfect_rounds >= PERFECT_SOLUTION_PATIENCE:
                print(f"\n  Stopping: Found perfect solution {consecutive_perfect_rounds} times")
                break
        else:
            consecutive_perfect_rounds = 0
        
        # Adaptive strategy based on progress
        if stagnation_counter >= STAGNATION_LIMIT:
            print(f"\n  ⚡ DIVERSITY INJECTION TRIGGERED ⚡")
            
            # Determine strategy based on current progress
            progress = current_correctness / perfect_score
            if progress < 0.5:
                strategy = 'random'  # Need more exploration
                print(f"  Strategy: Random (exploration)")
            elif progress < 0.8:
                strategy = 'mixed'  # Balance
                print(f"  Strategy: Mixed (balance)")
            else:
                strategy = 'guided'  # Fine-tuning
                print(f"  Strategy: Guided (exploitation)")
            
            # Inject diversity
            for i, solver in enumerate(solvers):
                inject_diversity_to_solver(solver, strategy)
                print(f"    Thread {i}: Diversity injected")
            
            stagnation_counter = 0
    
    # Archive sharing (ring topology)
    if round_num % 2 == 0:  # Every 2 rounds
        print(f"\n  Sharing archives (ring topology)...")
        share_archives_ring_topology(solvers)
    
    # Track history
    best_correctness_history.append(best_solution.obj1 if best_solution else 0)

# --------------------------------------
# Final Results
# --------------------------------------
print(f"\n{'='*70}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*70}\n")

final_solution = global_archive.get_best_for_objective('correctness')

if final_solution:
    print("--- BEST SOLUTION FOUND ---\n")
    
    # Get circuit formula
    formulas = problem_def.get_circuit_formula(final_solution.position)
    circuit_matrix, output_array = problem_def.decode_particle(final_solution.position)
    
    print("\nCircuit Matrix ([In1, Gate, In2]):")
    print(circuit_matrix)
    print("\nOutput Array (Pointers):")
    print(output_array)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Correctness: {int(final_solution.obj1)}/{perfect_score} "
          f"({100*final_solution.obj1/perfect_score:.1f}%)")
    print(f"Active gates: {int(final_solution.obj2)}/{num_rows} "
          f"(Unused: {num_rows - int(final_solution.obj2)})")
    
    if final_solution.obj1 == perfect_score:
        print("\n✓✓✓ PERFECT FUNCTIONAL SOLUTION ACHIEVED! ✓✓✓")
    
    print(f"\nTotal Pareto solutions in archive: {len(global_archive.archive)}")
    
    # Show diversity of solutions
    if len(global_archive.archive) > 1:
        correctness_vals = [p.obj1 for p in global_archive.archive]
        gates_vals = [p.obj2 for p in global_archive.archive]
        print(f"Correctness range: {int(min(correctness_vals))} - {int(max(correctness_vals))}")
        print(f"Gates range: {int(min(gates_vals))} - {int(max(gates_vals))}")

else:
    print("No solution found.")

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"\n[TIMER] Elapsed time: {elapsed_seconds:.2f} seconds")
if round_num > 0:
    print(f"[TIMER] Time per round: {elapsed_seconds/min(round_num+1, MAX_VOTING_ROUNDS):.2f} seconds")