import problem4
import pso_solver # Assumes the file above is named pso_solver.py
import numpy as np
import math
import sys
import os
import copy
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor


# --------------------------------------
# Logging Setup
# --------------------------------------
# --- MOVED: Logger setup is now inside the __main__ block ---

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

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
# --- These are now just definitions, they don't print anything ---
NUM_PROCESSES = 5
VOTING_INTERVAL = 150
MAX_VOTING_ROUNDS = 20
STAGNATION_LIMIT = 5

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

nPop_per_thread = 200
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

pso_params = {'w': w, 'c1': c1, 'c2': c2, 'velMax': velMax, 'velMin': velMin, 'w_damp': w_damp}

# --------------------------------------
# Process Execution Function
# --------------------------------------
def run_solver_iterations(solver):
    """
    Takes a solver, runs it for VOTING_INTERVAL iterations,
    and returns the MODIFIED solver.
    """
    try:
        # Note: pso_solver.py MUST have the w_damp bug fixed
        solver.run_iterations(VOTING_INTERVAL)
        return solver
    except Exception as e:
        # This print will go to the child process's log
        print(f"\n[ERROR] Process failed: {e}\n")
        return None

# --------------------------------------
# System Pipeline Main Loop
# --------------------------------------
if __name__ == "__main__":

    # --- MOVED: All setup code is now inside the main guard ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(results_dir, f"pso_log_{timestamp}.txt")
    
    sys.stdout = Logger(log_filename) # This now only happens ONCE
    
    print(f"[LOGGING ENABLED] Output is being saved to {log_filename}\n")
    print("[Multi-Process PSO] System is running...")
    
    _perf_start = time.perf_counter()

    print(f"Pipeline Config: {NUM_PROCESSES} Processes, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")
    # --- End of moved code ---

    # 1. Initialization Phase
    solvers = [pso_solver.BasePSOSolver(nPop_per_thread, varSize, pso_params) for _ in range(NUM_PROCESSES)]

    Voted_GBest = pso_solver.Particle(varSize)
    Voted_GBest.fitness = -math.inf
    stagnation_counter = 0

    # Create the ProcessPoolExecutor once
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as pool:
        for round_num in range(MAX_VOTING_ROUNDS):
            print(f"\n--- Voting Round {round_num + 1} / {MAX_VOTING_ROUNDS} ---")

            # 2. Parallel Optimization Phase
            print(f"Starting {NUM_PROCESSES} processes for {VOTING_INTERVAL} iterations each...")
            
            results = list(pool.map(run_solver_iterations, solvers))
            
            solvers = results 
            print("All processes finished.")

            # 3. Voting Phase
            gbest_candidates = [s.get_gbest() for s in solvers if s is not None]
            
            if len(gbest_candidates) != NUM_PROCESSES:
                print("[WARNING] Not all processes reported a GBest. Skipping round.")
                continue

            round_best = max(gbest_candidates, key=lambda p: p.fitness)

            # 4. Check Stopping Conditions
            
            # Condition 1: Check for perfect solution
            perfect_score = problem4.truth_table.total_outputs
            if round_best.num_equal_tt == perfect_score:
                print(f"\n!!! PERFECT FUNCTIONAL SOLUTION FOUND (Fitness: {round_best.fitness:.2f}) !!!")
                if round_best.fitness > Voted_GBest.fitness:
                    print("New solution is also the simplest found so far. Updating Voted_GBest.")
                    Voted_GBest = round_best
                break 
            
            # Condition 2: Check for Voted_GBest improvement
            if round_best.fitness > Voted_GBest.fitness:
                print(f"New Overall Champion found! Fitness: {round_best.fitness:.2f} (Correct: {round_best.num_equal_tt}/{perfect_score})")
                Voted_GBest = round_best
                stagnation_counter = 0 
            else:
                stagnation_counter += 1
                print(f"Overall Champion did not improve. (Stagnation: {stagnation_counter}/{STAGNATION_LIMIT})")

            # Condition 3: Check for stagnation
            if stagnation_counter >= STAGNATION_LIMIT:
                print("Stopping conditions met: Overall Champion has stagnated.")
                break
                
            # 5. Feedback Phase (VON NEUMANN TOPOLOGY)
            print("Performing Von Neumann (1D Ring) communication...")
            
            new_targets = {}
            
            for i in range(NUM_PROCESSES):
                left_neighbor_idx = (i - 1) % NUM_PROCESSES
                right_neighbor_idx = (i + 1) % NUM_PROCESSES
                
                particle_self = gbest_candidates[i]
                particle_left = gbest_candidates[left_neighbor_idx]
                particle_right = gbest_candidates[right_neighbor_idx]
                
                neighborhood = [particle_self, particle_left, particle_right]
                
                local_best_particle = max(neighborhood, key=lambda p: p.fitness)
                new_targets[i] = local_best_particle

            # Now, influence each solver with its new "local best" target
            for i in range(NUM_PROCESSES):
                local_best = new_targets[i]
                
                # Find which solver the local_best came from (for logging)
                origin_solver = -1
                for j, p in enumerate(gbest_candidates):
                    if p is local_best:
                        origin_solver = j
                        break
                
                print(f"  Solver {i}'s new target is (Fitness: {local_best.fitness:.2f}) from Solver {origin_solver}")
                solvers[i].set_new_gbest(local_best)

    # --------------------------------------
    # 7. Output Phase
    # --------------------------------------
    print("\n... Pipeline finished ...\n")
    print("--- FINAL CHAMPION PARTICLE ---")

    final_formulas = problem4.get_circuit_formula(Voted_GBest.position)
    circuit_matrix, output_array = problem4.decode_particle(Voted_GBest.position)

    print("\nCircuit Matrix ([In1, Gate, In2]):")
    print(circuit_matrix)
    print("\nOutput Array (Pointers):")
    print(output_array)
    print("\n--- FINAL STATS ---")
    print(f"Cost (Fitness): {Voted_GBest.fitness:.4f}")
    print(f"Correct outputs: {Voted_GBest.num_equal_tt}/{problem4.truth_table.total_outputs}")
    print(f"Active Gates Used: {Voted_GBest.num_gates}")
    print(f"Unused Gates (Simplicity): {Voted_GBest.num_no_gates}")

    print("\n[Multi-Threaded PSO] System finished!")

    _perf_end = time.perf_counter()
    elapsed_seconds = _perf_end - _perf_start
    print(f"[TIMER] Elapsed seconds: {elapsed_seconds:.6f}")