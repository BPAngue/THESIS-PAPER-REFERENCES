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

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"pso_sa_v3_log_{timestamp}.txt")

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
print("[SA-Enhanced V3 with Diversity Pressure] System is running...")

_perf_start = time.perf_counter()

# --------------------------------------
# V3 IMPROVEMENTS
# --------------------------------------
print("\n=== V3 IMPROVEMENTS ===")
print("1. Diversity-driven fitness function to break plateaus")
print("2. Increased circuit complexity (more gates)")
print("3. Tournament selection for SA particle targeting")
print("4. Strategic restart with elite preservation + heavy mutation")
print("5. Adaptive diversity bonus based on stagnation")
print("========================\n")

# --------------------------------------
# Simulated Annealing Parameters
# --------------------------------------
SA_INITIAL_TEMP = 800.0      # Even hotter (was 500)
SA_COOLING_RATE = 0.98       
SA_MIN_TEMP = 0.001          
SA_ITERATIONS_PER_TEMP = 150 # More thorough (was 100)
SA_TRIGGER_STAGNATION = 8    # Trigger faster (was 10)
SA_MAX_GENES_PERTURB = 4     # More aggressive (was 3)

print(f"SA Config V3: Initial Temp={SA_INITIAL_TEMP}, Cooling={SA_COOLING_RATE}, "
      f"Min Temp={SA_MIN_TEMP}, Iters/Temp={SA_ITERATIONS_PER_TEMP}, Max Genes={SA_MAX_GENES_PERTURB}")

# --------------------------------------
# System Pipeline Parameters
# --------------------------------------
NUM_THREADS = 5
VOTING_INTERVAL = 1
MAX_VOTING_ROUNDS = 200      # More time
STAGNATION_LIMIT = 200
RESTART_THRESHOLD = 30       # Restart sooner

print(f"Pipeline Config: {NUM_THREADS} Threads, {VOTING_INTERVAL} Iter/Round, {MAX_VOTING_ROUNDS} Max Rounds")

# --------------------------------------
# PSO Base Solver Parameters
# --------------------------------------
num_inputs = problem4.truth_table.num_inputs
num_outputs = problem4.truth_table.num_outputs

# *** INCREASE CIRCUIT COMPLEXITY ***
# Override the problem definition to use more gates
original_num_rows = problem4.num_rows
problem4.num_rows = 8  # Increased from 5 to 8!
num_rows = problem4.num_rows

print(f"\n*** INCREASED CIRCUIT COMPLEXITY: {original_num_rows} → {num_rows} gates ***\n")

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

# --------------------------------------
# DIVERSITY-DRIVEN FITNESS WRAPPER
# --------------------------------------
class DiversityFitnessWrapper:
    """
    Wraps the fitness function to add diversity pressure.
    This breaks fitness plateaus by rewarding circuits that differ from the champion.
    """
    def __init__(self):
        self.champion_position = None
        self.diversity_weight = 0.0  # Starts at 0, increases with stagnation
        self.stagnation_rounds = 0
        
    def update_champion(self, position):
        """Update the reference champion position."""
        self.champion_position = position.copy() if position is not None else None
        
    def increase_diversity_pressure(self):
        """Increase diversity weight when stagnation occurs."""
        self.stagnation_rounds += 1
        # Gradually increase diversity pressure (caps at 0.1)
        self.diversity_weight = min(0.1, 0.005 * self.stagnation_rounds)
        print(f"[DIVERSITY] Increased diversity pressure: {self.diversity_weight:.4f}")
        
    def reset_diversity_pressure(self):
        """Reset diversity pressure when progress is made."""
        self.stagnation_rounds = 0
        self.diversity_weight = 0.0
        print(f"[DIVERSITY] Reset diversity pressure")
        
    def __call__(self, position):
        """Enhanced fitness function with diversity bonus."""
        # Get base fitness
        base_fitness, num_equal, num_gates, num_no_gates = problem4.fitness_function(position)
        
        # Add diversity bonus if we have a champion and are not functional yet
        diversity_bonus = 0.0
        if self.champion_position is not None and num_equal < problem4.truth_table.total_outputs:
            # Count how many genes differ from champion
            differences = np.sum(position != self.champion_position)
            total_genes = len(position)
            
            # Normalize to [0, 1] and scale by diversity weight
            diversity_ratio = differences / total_genes
            diversity_bonus = diversity_ratio * self.diversity_weight
        
        final_fitness = base_fitness + diversity_bonus
        
        return final_fitness, num_equal, num_gates, num_no_gates

# Global fitness wrapper
diversity_fitness = DiversityFitnessWrapper()

def fitnessFunction(position):
    """Fitness function used by PSO."""
    return diversity_fitness(position)

# Update pso_solver to use our fitness function
pso_solver.fitnessFunction = fitnessFunction

# --------------------------------------
# IMPROVED Neighbor Generation
# --------------------------------------
def generate_neighbor_solution(position, num_genes_to_perturb=None):
    """
    V3: More aggressive perturbation (1-4 genes at once).
    """
    neighbor = position.copy()
    
    if num_genes_to_perturb is None:
        num_genes_to_perturb = np.random.randint(1, SA_MAX_GENES_PERTURB + 1)
    
    genes_to_modify = np.random.choice(len(neighbor), 
                                       size=min(num_genes_to_perturb, len(neighbor)), 
                                       replace=False)
    
    circuit_matrix_size = num_rows * 3
    
    for gene_idx in genes_to_modify:
        idx = gene_idx
        
        if idx < circuit_matrix_size:
            row_in_matrix = idx // 3
            col_in_row = idx % 3
            
            if col_in_row == 0 or col_in_row == 2:  # Input genes
                input_high_inclusive = num_inputs + row_in_matrix
                neighbor[idx] = np.random.randint(1, input_high_inclusive + 1)
            else:  # Gate type gene
                gate_type_high = len(problem4.GateType)
                neighbor[idx] = np.random.randint(0, gate_type_high)
        else:
            output_high_inclusive = num_inputs + num_rows
            neighbor[idx] = np.random.randint(1, output_high_inclusive + 1)
    
    return neighbor


def acceptance_probability(current_fitness, new_fitness, temperature):
    """Calculate probability of accepting a worse solution."""
    if new_fitness > current_fitness:
        return 1.0
    else:
        delta = new_fitness - current_fitness
        return math.exp(delta / temperature)


def simulated_annealing_adaptive(initial_particle, max_iterations=None):
    """
    Adaptive SA with better diagnostics.
    """
    print("\n[SA V3] Starting Adaptive Simulated Annealing...")
    
    current = copy.deepcopy(initial_particle)
    best = copy.deepcopy(current)
    
    temperature = SA_INITIAL_TEMP
    iteration = 0
    improvements = 0
    plateau_moves = 0  # Track lateral moves on plateau
    
    if max_iterations is None:
        max_iterations = int(math.log(SA_MIN_TEMP / SA_INITIAL_TEMP) / 
                            math.log(SA_COOLING_RATE)) * SA_ITERATIONS_PER_TEMP
    
    while temperature > SA_MIN_TEMP and iteration < max_iterations:
        accepted_count = 0
        
        for _ in range(SA_ITERATIONS_PER_TEMP):
            iteration += 1
            
            neighbor_position = generate_neighbor_solution(current.position)
            
            neighbor_fitness, neighbor_equal, neighbor_gates, neighbor_no_gates = \
                fitnessFunction(neighbor_position)
            
            accept_prob = acceptance_probability(current.fitness, neighbor_fitness, temperature)
            
            if np.random.random() < accept_prob:
                accepted_count += 1
                
                # Track plateau moves
                if abs(neighbor_fitness - current.fitness) < 0.001:
                    plateau_moves += 1
                
                current.position = neighbor_position
                current.fitness = neighbor_fitness
                current.num_equal_tt = neighbor_equal
                current.num_gates = neighbor_gates
                current.num_no_gates = neighbor_no_gates
                
                if current.fitness > best.fitness:
                    best = copy.deepcopy(current)
                    improvements += 1
        
        acceptance_rate = accepted_count / SA_ITERATIONS_PER_TEMP
        
        # Adaptive cooling
        if acceptance_rate < 0.05:
            cooling_factor = 0.99
        elif acceptance_rate > 0.6:
            cooling_factor = 0.93
        else:
            cooling_factor = SA_COOLING_RATE
        
        temperature *= cooling_factor
        
        if iteration % 750 == 0:
            plateau_ratio = plateau_moves / iteration if iteration > 0 else 0
            print(f"[SA V3] Iter {iteration}: Temp={temperature:.2f}, Accept={acceptance_rate:.2%}, "
                  f"Best={best.fitness:.2f}, Plateau={plateau_ratio:.1%}")
    
    plateau_ratio = plateau_moves / iteration if iteration > 0 else 0
    print(f"[SA V3] Completed {iteration} iters, {improvements} improvements, {plateau_ratio:.1%} plateau moves")
    print(f"[SA V3] {initial_particle.fitness:.4f} → {best.fitness:.4f} (Δ={best.fitness-initial_particle.fitness:+.4f})")
    
    return best


def tournament_selection(particles, tournament_size=5):
    """
    Select particles using tournament selection for diversity.
    Returns particles that are both low-fitness AND diverse.
    """
    selected = []
    
    for _ in range(min(5, len(particles))):
        # Random tournament
        tournament = np.random.choice(particles, size=min(tournament_size, len(particles)), replace=False)
        
        # Select worst from tournament (encourages diversity by not always picking global worst)
        worst = min(tournament, key=lambda p: p.fitness)
        selected.append(worst)
    
    return selected


def apply_sa_to_population(solver, num_particles_to_refine=5):
    """V3: Use tournament selection instead of just worst particles."""
    print(f"\n[SA V3] Applying SA to {num_particles_to_refine} tournament-selected particles...")
    
    # Use tournament selection for diversity
    selected_particles = tournament_selection(solver.pop, tournament_size=7)
    
    for i, particle in enumerate(selected_particles[:num_particles_to_refine]):
        print(f"  Refining particle {i+1}/{num_particles_to_refine} (fitness: {particle.fitness:.2f})...")
        
        refined = simulated_annealing_adaptive(particle, max_iterations=500)
        
        particle.position = refined.position.copy()
        particle.fitness = refined.fitness
        particle.num_equal_tt = refined.num_equal_tt
        particle.num_gates = refined.num_gates
        particle.num_no_gates = refined.num_no_gates
        
        if particle.fitness > particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = particle.position.copy()
        
        if particle.best_fitness > solver.gbest.fitness:
            solver.gbest = copy.deepcopy(particle)
            print(f"  ✓ New solver gbest: {solver.gbest.fitness:.4f}")


def strategic_restart(solver, elite_particles):
    """
    V3: Strategic restart that preserves elite + adds heavy mutations.
    """
    print(f"\n[STRATEGIC RESTART] Preserving top 10%, heavily mutating rest...")
    
    keep_count = max(1, int(0.1 * len(solver.pop)))
    
    # Sort and keep elite
    sorted_pop = sorted(solver.pop, key=lambda p: p.fitness, reverse=True)
    elite = sorted_pop[:keep_count]
    
    print(f"  Keeping {keep_count} elite particles (fitness: {elite[0].fitness:.2f} to {elite[-1].fitness:.2f})")
    
    # Create new population
    new_pop = []
    
    # Add elite
    for p in elite:
        new_pop.append(copy.deepcopy(p))
    
    # Fill rest with heavily mutated versions of elite + global elite
    all_elite = elite + elite_particles
    while len(new_pop) < len(solver.pop):
        # Pick a random elite
        parent = copy.deepcopy(np.random.choice(all_elite))
        
        # Heavy mutation (change 30-60% of genes)
        mutation_rate = np.random.uniform(0.3, 0.6)
        num_mutations = int(mutation_rate * len(parent.position))
        
        mutant_position = parent.position.copy()
        genes_to_mutate = np.random.choice(len(mutant_position), size=num_mutations, replace=False)
        
        circuit_matrix_size = num_rows * 3
        for gene_idx in genes_to_mutate:
            if gene_idx < circuit_matrix_size:
                row_in_matrix = gene_idx // 3
                col_in_row = gene_idx % 3
                
                if col_in_row == 0 or col_in_row == 2:
                    input_high_inclusive = num_inputs + row_in_matrix
                    mutant_position[gene_idx] = np.random.randint(1, input_high_inclusive + 1)
                else:
                    gate_type_high = len(problem4.GateType)
                    mutant_position[gene_idx] = np.random.randint(0, gate_type_high)
            else:
                output_high_inclusive = num_inputs + num_rows
                mutant_position[gene_idx] = np.random.randint(1, output_high_inclusive + 1)
        
        # Evaluate mutant
        mutant = pso_solver.Particle(varSize)
        mutant.position = mutant_position
        mutant.fitness, mutant.num_equal_tt, mutant.num_gates, mutant.num_no_gates = fitnessFunction(mutant.position)
        mutant.best_position = mutant.position.copy()
        mutant.best_fitness = mutant.fitness
        
        new_pop.append(mutant)
    
    # Replace population
    solver.pop = new_pop
    
    # Update solver gbest
    solver.gbest = copy.deepcopy(max(new_pop, key=lambda p: p.fitness))
    
    print(f"  ✓ Restart complete. New gbest: {solver.gbest.fitness:.4f}")


# --------------------------------------
# Thread Execution Function
# --------------------------------------
def run_solver_thread(solver, thread_index, interval_iterations, results_dict, lock):
    """Run solver in thread."""
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
solvers = [pso_solver.BasePSOSolver(nPop_per_thread, varSize, pso_params) for _ in range(NUM_THREADS)]

Voted_GBest = pso_solver.Particle(varSize)
Voted_GBest.fitness = -math.inf
stagnation_counter = 0
sa_applied_count = 0
restart_count = 0

# Track elite particles across all solvers
global_elite_particles = []

for round_num in range(MAX_VOTING_ROUNDS):
    print(f"\n{'='*70}")
    print(f"ROUND {round_num + 1} / {MAX_VOTING_ROUNDS}")
    print(f"{'='*70}")

    threads = []
    gbest_candidates_dict = {} 
    thread_lock = threading.Lock() 

    # Parallel PSO
    print(f"Running {NUM_THREADS} PSO threads...")
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=run_solver_thread,
            args=(solvers[i], i, VOTING_INTERVAL, gbest_candidates_dict, thread_lock)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Voting
    if len(gbest_candidates_dict) != NUM_THREADS:
        print("[WARNING] Not all threads reported. Skipping.")
        continue

    round_best = max(gbest_candidates_dict.values(), key=lambda p: p.fitness)

    # Check for perfect solution
    perfect_score = problem4.truth_table.total_outputs
    if round_best.num_equal_tt == perfect_score:
        print(f"\n🎉 PERFECT FUNCTIONAL SOLUTION! Fitness: {round_best.fitness:.2f}")
        if round_best.fitness > Voted_GBest.fitness:
            print("✓ Also the simplest found!")
    
    # Update global best
    previous_fitness = Voted_GBest.fitness
    if round_best.fitness > Voted_GBest.fitness:
        print(f"✓ NEW CHAMPION: {round_best.fitness:.2f} (Correct: {round_best.num_equal_tt}/{perfect_score})")
        Voted_GBest = round_best
        stagnation_counter = 0
        
        # Update diversity reference
        diversity_fitness.update_champion(Voted_GBest.position)
        diversity_fitness.reset_diversity_pressure()
        
        # Update global elite
        global_elite_particles = [copy.deepcopy(solver.gbest) for solver in solvers]
        global_elite_particles.sort(key=lambda p: p.fitness, reverse=True)
        global_elite_particles = global_elite_particles[:3]  # Keep top 3
    else:
        stagnation_counter += 1
        print(f"✗ No improvement. Stagnation: {stagnation_counter}/{SA_TRIGGER_STAGNATION} (Restart: {stagnation_counter}/{RESTART_THRESHOLD})")
        
        # Increase diversity pressure
        if stagnation_counter % 5 == 0:
            diversity_fitness.increase_diversity_pressure()

    # ===== SA INTERVENTION =====
    if stagnation_counter >= SA_TRIGGER_STAGNATION and stagnation_counter % SA_TRIGGER_STAGNATION == 0:
        sa_applied_count += 1
        print(f"\n{'='*70}")
        print(f"🔥 SA INTERVENTION #{sa_applied_count} (Stagnation: {stagnation_counter})")
        print(f"{'='*70}")
        
        if Voted_GBest.num_equal_tt < perfect_score:
            print(f"⚠️  NON-FUNCTIONAL: {Voted_GBest.num_equal_tt}/{perfect_score} correct")
        else:
            print(f"✓ FUNCTIONAL but sub-optimal")
        
        # Refine champion
        print("\n[SA V3] Refining champion...")
        refined_champion = simulated_annealing_adaptive(Voted_GBest, max_iterations=1500)
        
        if refined_champion.fitness > Voted_GBest.fitness:
            print(f"✅ CHAMPION IMPROVED! {Voted_GBest.fitness:.4f} → {refined_champion.fitness:.4f}")
            Voted_GBest = refined_champion
            stagnation_counter = 0
            diversity_fitness.update_champion(Voted_GBest.position)
            diversity_fitness.reset_diversity_pressure()
        else:
            print(f"✗ Champion not improved")
        
        # Diversify populations
        print(f"\n[SA V3] Diversifying all {NUM_THREADS} solver populations...")
        for solver_idx, solver in enumerate(solvers):
            print(f"\n--- Solver {solver_idx + 1}/{NUM_THREADS} ---")
            apply_sa_to_population(solver, num_particles_to_refine=4)
        
        print(f"\n{'='*70}")
        print(f"SA INTERVENTION #{sa_applied_count} COMPLETE")
        print(f"{'='*70}\n")

    # ===== STRATEGIC RESTART =====
    if stagnation_counter >= RESTART_THRESHOLD and stagnation_counter % RESTART_THRESHOLD == 0:
        restart_count += 1
        print(f"\n{'!'*70}")
        print(f"🔄 STRATEGIC RESTART #{restart_count}")
        print(f"{'!'*70}")
        
        # Find worst solver
        solver_fitness = [(i, solver.gbest.fitness) for i, solver in enumerate(solvers)]
        worst_idx = min(solver_fitness, key=lambda x: x[1])[0]
        
        print(f"Restarting Solver {worst_idx} (fitness: {solvers[worst_idx].gbest.fitness:.2f})")
        strategic_restart(solvers[worst_idx], global_elite_particles)
        print(f"{'!'*70}\n")

    # Final stopping condition
    if stagnation_counter >= STAGNATION_LIMIT:
        print("\n[STOP] Maximum stagnation reached.")
        break
        
    # Ring Topology
    print("🔄 Ring topology communication...")
    current_gbests = [solver.get_gbest() for solver in solvers]
    for i in range(NUM_THREADS):
        previous_thread_idx = (i - 1) % NUM_THREADS
        solvers[i].force_gbest_replacement(current_gbests[previous_thread_idx])

# --------------------------------------
# Output Phase
# --------------------------------------
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

final_formulas = problem4.get_circuit_formula(Voted_GBest.position)
circuit_matrix, output_array = problem4.decode_particle(Voted_GBest.position)

print("\nCircuit Matrix:")
print(circuit_matrix)
print("\nOutput Array:")
print(output_array)
print("\n--- STATISTICS ---")
print(f"Final Fitness: {Voted_GBest.fitness:.4f}")
print(f"Correct Outputs: {Voted_GBest.num_equal_tt}/{problem4.truth_table.total_outputs}")
print(f"Active Gates: {Voted_GBest.num_gates}")
print(f"Unused Gates: {Voted_GBest.num_no_gates}")
print(f"\nSA Interventions: {sa_applied_count}")
print(f"Strategic Restarts: {restart_count}")
print(f"Final Diversity Weight: {diversity_fitness.diversity_weight:.4f}")

print("\n[SA-Enhanced V3] Complete!")

_perf_end = time.perf_counter()
print(f"[TIMER] {_perf_end - _perf_start:.2f} seconds")