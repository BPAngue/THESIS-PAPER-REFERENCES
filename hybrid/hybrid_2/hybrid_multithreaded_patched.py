from datetime import datetime
import math
import os
import sys
import time
import threading
import copy

import numpy as np

# Use patched problem_def (adds curriculum row support)
import problem_def_patched as problem_def

# Use patched hybrid solver (fixes pointer clamping by reusing PSO clamp)
from hybrid_solver_patched import HybridPSOGWOSolver

import pso_solver

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"pso-gwo_log_{timestamp}.txt")

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
print("[Multi-Threaded PSO-GWO Hybrid | Full-Adder Success Mode] System is running...")

_perf_start = time.perf_counter()

# --------------------------------------
# System Pipeline Parameters (tuned for full adder success)
# --------------------------------------
NUM_THREADS = 6
VOTING_INTERVAL = 5              # iterations per round per thread
MAX_VOTING_ROUNDS = 2500         # total iterations ~= NUM_THREADS-independent
STAGNATION_LIMIT = 20            # rounds without improvement before soft reset

# Migration (ring) - but as INJECTION (not gbest overwrite)
MIGRATION_INTERVAL = 20          # rounds between migrations (20*5 = every 100 iters)
MIGRANTS = 2                     # how many elites to send to next island

# Soft reset proportions (keep building blocks)
KEEP_FRAC = 0.15
MUTATE_FRAC = 0.35

# Curriculum schedule (rows to evaluate)
# Full adder has 8 rows; we gradually increase evaluated rows.
CURRIC_PHASE_ITERS = 200         # iterations per phase boundary (global iterations)
CURRIC_SIZES = [4, 6, 8]         # number of rows evaluated in each phase
CURRIC_SEED = 12345              # fixed so runs are comparable

# --------------------------------------
# PSO Base Solver Parameters
# --------------------------------------
num_inputs = problem_def.truth_table.num_inputs
num_outputs = problem_def.truth_table.num_outputs
num_rows = problem_def.num_rows

nVar = (num_rows * 3) + num_outputs
varSize = nVar
varMin = 0
varMax = num_inputs + num_rows

# For success rate on full-adder: favor more particles over fewer threads if CPU limited.
nPop_per_thread = 220

constriction_coefficient = True
if not constriction_coefficient:
    w, w_damp, c1, c2 = 1, 0.99, 2, 2
else:
    # keep your original chi formulation
    phi1, phi2 = 1.05, 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    w, w_damp, c1, c2 = chi, 1, chi * phi1, chi * phi2

velMax = 0.2 * (varMax - varMin)
velMin = -velMax

pso_params = {'w': w, 'c1': c1, 'c2': c2, 'velMax': velMax, 'velMin': velMin, 'w_damp': w_damp}

print(
    "Pipeline Config:",
    f"{NUM_THREADS} Threads,",
    f"{nPop_per_thread} Pop/Thread,",
    f"{VOTING_INTERVAL} Iter/Round,",
    f"{MAX_VOTING_ROUNDS} Max Rounds,",
    f"{STAGNATION_LIMIT} Stagnation Limit,",
    f"Migration: every {MIGRATION_INTERVAL} rounds, {MIGRANTS} migrants",
)
print("Curriculum:", f"{CURRIC_SIZES} rows, phase every {CURRIC_PHASE_ITERS} iterations")

# --------------------------------------
# Curriculum helper
# --------------------------------------
_all_rows = list(range(problem_def.truth_table.num_rows_tt))
_rng = np.random.default_rng(CURRIC_SEED)

# pre-pick fixed row subsets per phase so the landscape is stable inside a phase
_curric_subsets = {}
for k in CURRIC_SIZES:
    _curric_subsets[k] = sorted(_rng.choice(_all_rows, size=k, replace=False).tolist())

def apply_curriculum(global_iter: int):
    # phase index determined by global_iter
    phase = min(global_iter // CURRIC_PHASE_ITERS, len(CURRIC_SIZES) - 1)
    k = CURRIC_SIZES[phase]
    problem_def.set_active_rows(_curric_subsets[k])
    return k, phase

# --------------------------------------
# Thread Execution Function
# --------------------------------------
def run_solver_thread(solver, thread_index, interval_iterations, results_dict, lock):
    try:
        solver.run_iterations(interval_iterations)
        gbest = solver.get_gbest()
        with lock:
            results_dict[thread_index] = gbest
    except Exception as e:
        print(f"\n[ERROR] Thread {thread_index} failed: {e}\n")

# --------------------------------------
# Migration injection (ring, but diversity-preserving)
# --------------------------------------
def inject_migrants_ring(solvers, migrants=MIGRANTS):
    n = len(solvers)
    # snapshot elites from each island
    elites = []
    for s in solvers:
        pop_sorted = sorted(s.pop, key=lambda p: p.fitness, reverse=True)
        elites.append([copy.deepcopy(p) for p in pop_sorted[:migrants]])

    # inject into next island by replacing worst
    for i in range(n):
        dst = solvers[(i + 1) % n]
        dst_sorted = sorted(dst.pop, key=lambda p: p.fitness, reverse=False)  # worst first
        for j in range(migrants):
            worst = dst_sorted[j]
            elite = elites[i][j]
            worst.position = elite.position.copy()
            worst.velocity = elite.velocity.copy()
            worst.fitness = elite.fitness
            worst.best_position = elite.best_position.copy()
            worst.best_fitness = elite.best_fitness
            worst.num_equal_tt = elite.num_equal_tt
            worst.num_gates = elite.num_gates
            worst.num_no_gates = elite.num_no_gates

# --------------------------------------
# Soft reset (keeps building blocks)
# --------------------------------------
def random_valid_gene_value(gene_idx: int) -> int:
    gate_block = num_rows * 3
    if gene_idx < gate_block:
        col = gene_idx % 3
        row = gene_idx // 3  # 0-based gate row
        if col == 1:
            return int(_rng.integers(0, len(problem_def.GateType)))
        else:
            # pointers allowed: 1..(num_inputs + row)
            high = num_inputs + row
            return int(_rng.integers(1, high + 1))
    else:
        # output pointer: 1..(num_inputs + num_rows)
        high = num_inputs + num_rows
        return int(_rng.integers(1, high + 1))

def targeted_mutation(particle, flips=2):
    # bias later gates: choose gene indices with higher probability near end
    for _ in range(flips):
        # quadratic bias towards later indices
        u = _rng.random()
        idx = int((u ** 0.5) * (varSize - 1))  # sqrt biases high
        particle.position[idx] = random_valid_gene_value(idx)

def soft_reset_solver(solver):
    solver.pop.sort(key=lambda p: p.fitness, reverse=True)
    n = solver.nPop
    k_keep = max(1, int(KEEP_FRAC * n))
    k_mut = max(1, int(MUTATE_FRAC * n))

    # Keep best k_keep as-is
    # Mutate next k_mut
    for i in range(k_keep, min(k_keep + k_mut, n)):
        p = solver.pop[i]
        targeted_mutation(p, flips=3)
        (p.fitness, p.num_equal_tt, p.num_gates, p.num_no_gates) = problem_def.fitness_function(p.position)
        if p.fitness > p.best_fitness:
            p.best_fitness = p.fitness
            p.best_position = p.position.copy()

    # Re-randomize the rest
    for i in range(k_keep + k_mut, n):
        solver.pop[i] = pso_solver.Particle(varSize)
        (solver.pop[i].fitness,
         solver.pop[i].num_equal_tt,
         solver.pop[i].num_gates,
         solver.pop[i].num_no_gates) = problem_def.fitness_function(solver.pop[i].position)
        solver.pop[i].best_fitness = solver.pop[i].fitness
        solver.pop[i].best_position = solver.pop[i].position.copy()

    # Reset inertia higher for exploration
    solver.w = 0.9
    solver.evaluate_population()

# --------------------------------------
# Main Loop
# --------------------------------------
solvers = [HybridPSOGWOSolver(nPop_per_thread, varSize, pso_params) for _ in range(NUM_THREADS)]

Voted_GBest = pso_solver.Particle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0

perfect_score = problem_def.truth_table.total_outputs
global_iter = 0

for round_num in range(MAX_VOTING_ROUNDS):
    # curriculum selection based on global iterations so far
    k_rows, phase = apply_curriculum(global_iter)

    print(f"\n--- Round {round_num + 1} / {MAX_VOTING_ROUNDS} | global_iter={global_iter} | curriculum_rows={k_rows} (phase {phase}) ---")

    threads = []
    gbest_candidates_dict = {}
    thread_lock = threading.Lock()

    # parallel optimization
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], i, VOTING_INTERVAL, gbest_candidates_dict, thread_lock)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_iter += VOTING_INTERVAL

    if len(gbest_candidates_dict) != NUM_THREADS:
        print("[WARNING] Not all threads reported a GBest. Skipping round.")
        continue

    round_best = max(gbest_candidates_dict.values(), key=lambda p: p.fitness)

    # If we are in curriculum phases, "perfect" means perfect on the active subset.
    # But we only stop when we are evaluating all rows.
    full_rows_active = (problem_def.ACTIVE_ROWS is None) or (len(problem_def.ACTIVE_ROWS) == problem_def.truth_table.num_rows_tt)

    if full_rows_active and round_best.num_equal_tt == perfect_score:
        print(f"\n!!! PERFECT FULL-ADDER SOLUTION FOUND (Fitness: {round_best.fitness:.2f}) !!!")
        Voted_GBest = round_best
        break

    # global champion improvement
    if round_best.fitness > Voted_GBest.fitness:
        print(f"New Overall Champion! Fitness: {round_best.fitness:.2f} (Correct: {round_best.num_equal_tt}/{perfect_score})")
        Voted_GBest = round_best
        stagnation_counter = 0
    else:
        stagnation_counter += 1
        print(f"Champion did not improve. Stagnation: {stagnation_counter}/{STAGNATION_LIMIT}")

    # stagnation soft reset
    if stagnation_counter >= STAGNATION_LIMIT:
        print("!!! Stagnation detected. Triggering SOFT RESET across islands !!!")
        for s in solvers:
            soft_reset_solver(s)
        stagnation_counter = 0

    # migration injection (every MIGRATION_INTERVAL rounds)
    if (round_num + 1) % MIGRATION_INTERVAL == 0:
        print("Performing ring migration (elite injection)...")
        inject_migrants_ring(solvers, migrants=MIGRANTS)

# --------------------------------------
# Output Phase
# --------------------------------------
print("\n... PSO-GWO Pipeline finished ...\n")
print("--- FINAL CHAMPION PARTICLE ---")

final_formulas = problem_def.get_circuit_formula(Voted_GBest.position)
circuit_matrix, output_array = problem_def.decode_particle(Voted_GBest.position)

print("\nCircuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)
print("\n--- FINAL STATS ---")
print(f"Cost (Fitness): {Voted_GBest.fitness:.4f}")
print(f"Correct outputs: {Voted_GBest.num_equal_tt}/{problem_def.truth_table.total_outputs}")
print(f"Active Gates Used: {Voted_GBest.num_gates}")
print(f"Unused Gates (Simplicity): {Voted_GBest.num_no_gates}")

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"\n[TIMER] Elapsed seconds: {elapsed_seconds:.6f}")
print("\n[Multi-Threaded PSO-GWO Hybrid] Finished.")
