import problem4
import numpy as np
import math
import copy

# --------------------------------------
# Particle Class (MODIFIED)
# --------------------------------------
class Particle:
    """Holds the state for a single particle."""
    def __init__(self, varsize):
        self.position: np.ndarray = create_random_particle(varsize)
        self.velocity = np.random.uniform(-1, 1, size=varsize)
        self.best_position: np.ndarray = self.position.copy()
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        # --- [NEW] Add stagnation counter for particle re-initialization ---
        self.stagnation_counter = 0
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0
        self.prev_static_fitness = 0.0
        self.prev_derivative = 0.0

# --------------------------------------
# Encoding Helper Functions (Unchanged)
# --------------------------------------
def create_random_particle(varsize):
    """Creates a 1D particle vector with valid, random integers for each gene."""
    particle = np.zeros(varsize, dtype=int)
    idx = 0
    num_inputs = problem4.truth_table.num_inputs
    num_rows = problem4.num_rows
    num_outputs = problem4.truth_table.num_outputs
    gate_type_high = len(problem4.GateType)

    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        input_high_exclusive = input_high_inclusive + 1
        
        particle[idx] = np.random.randint(1, input_high_exclusive)
        idx += 1
        particle[idx] = np.random.randint(0, gate_type_high)
        idx += 1
        particle[idx] = np.random.randint(1, input_high_exclusive)
        idx += 1

    output_high_inclusive = num_inputs + num_rows
    output_high_exclusive = output_high_inclusive + 1
    
    for _ in range(num_outputs):
        particle[idx] = np.random.randint(1, output_high_exclusive)
        idx += 1
        
    return particle

def clamp_particle_position(position_float):
    """Clamps a particle's position vector to valid integer ranges."""
    x = np.round(position_float).astype(int)
    idx = 0
    num_inputs = problem4.truth_table.num_inputs
    num_rows = problem4.num_rows
    num_outputs = problem4.truth_table.num_outputs
    gate_type_high = len(problem4.GateType) - 1

    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        x[idx] = max(1, min(x[idx], input_high_inclusive))
        idx += 1
        x[idx] = max(0, min(x[idx], gate_type_high))
        idx += 1
        x[idx] = max(1, min(x[idx], input_high_inclusive))
        idx += 1

    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        x[idx] = max(1, min(x[idx], output_high_inclusive))
        idx += 1
        
    return x

# --------------------------------------
# Fitness Function Wrapper (Unchanged)
# --------------------------------------
def fitnessFunction(position, prev_static_fitness, previous_derivative):
    """Wrapper for the problem4 fitness function."""
    return problem4.problem(position, prev_static_fitness, previous_derivative)

# --------------------------------------
# Base PSO Solver Class (MODIFIED)
# --------------------------------------
class BasePSOSolver:
    """
    Represents a single, independent PSO "Thread" (or swarm).
    Uses internal LBEST and particle re-initialization for exploration.
    """
    def __init__(self, nPop, varSize, pso_params):
        self.nPop = nPop
        self.varSize = varSize
        self.w = pso_params['w']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']
        self.velMax = pso_params['velMax']
        self.velMin = pso_params['velMin']
        
        self.neighborhood_size = 5
        self.half_k = self.neighborhood_size // 2
        
        # --- [NEW] Stagnation limit for individual particles ---
        self.particle_stagnation_limit = 20
        
        self.pop = [Particle(varSize) for _ in range(nPop)]
        self.gbest_tracker = Particle(varSize)
        self.gbest_tracker.fitness = -math.inf
        
        self.evaluate_population()

    def evaluate_population(self):
        """Calculates fitness for all particles and updates gbest_tracker."""
        for particle in self.pop:
            particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
            particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                particle.position, 
                particle.prev_static_fitness, 
                particle.prev_derivative
            )

            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
                particle.stagnation_counter = 0 # Reset on improvement

            if particle.best_fitness > self.gbest_tracker.fitness:
                self.gbest_tracker = copy.deepcopy(particle)

    def run_iterations(self, num_iterations):
        """Run the standard LBEST PSO for a number of iterations."""
        for _ in range(num_iterations):
            for i, particle in enumerate(self.pop):
                
                # --- [NEW] Check for particle stagnation ---
                if particle.stagnation_counter > self.particle_stagnation_limit:
                    # This particle is stuck. Re-initialize it.
                    particle.position = create_random_particle(self.varSize)
                    particle.velocity = np.random.uniform(self.velMin, self.velMax, size=self.varSize)
                    particle.fitness = -math.inf # Will be re-evaluated
                    particle.best_fitness = -math.inf
                    particle.stagnation_counter = 0
                    particle.prev_static_fitness = 0.0
                    particle.prev_derivative = 0.0
                
                # --- Find LBest ---
                lbest_particle = particle
                for j in range(1, self.half_k + 1):
                    neighbor_idx_up = (i + j) % self.nPop
                    if self.pop[neighbor_idx_up].best_fitness > lbest_particle.best_fitness:
                        lbest_particle = self.pop[neighbor_idx_up]
                    neighbor_idx_down = (i - j + self.nPop) % self.nPop
                    if self.pop[neighbor_idx_down].best_fitness > lbest_particle.best_fitness:
                        lbest_particle = self.pop[neighbor_idx_down]

                # --- Update Velocity (lbest) ---
                r1 = np.random.random(size=self.varSize)
                r2 = np.random.random(size=self.varSize)
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position - particle.position)) + \
                                    (self.c2 * r2 * (lbest_particle.best_position - particle.position))

                particle.velocity = np.clip(particle.velocity, self.velMin, self.velMax)
                
                # --- Update Position ---
                particle.position = particle.position + particle.velocity
                particle.position = clamp_particle_position(particle.position)
                
                # --- Evaluate ---
                particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
                particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                    particle.position, 
                    particle.prev_static_fitness, 
                    particle.prev_derivative
                )

                # --- Update PBest and Stagnation Counter ---
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                    particle.stagnation_counter = 0 # Reset on improvement

                    if particle.best_fitness > self.gbest_tracker.fitness:
                        self.gbest_tracker = copy.deepcopy(particle)
                else:
                    # No improvement, increment stagnation
                    particle.stagnation_counter += 1

    def get_gbest(self):
        """Returns this solver's best-found particle (the tracker)."""
        return copy.deepcopy(self.gbest_tracker)

    def inject_solution(self, solution_particle, percentage_to_replace):
        """Implements the "Feedback" step."""
        self.pop.sort(key=lambda p: p.fitness)
        num_to_replace = int(self.nPop * percentage_to_replace)
        if num_to_replace == 0:
            return

        print(f"  [Injector] Replacing worst {num_to_replace} particles with new solution.")
        for i in range(num_to_replace):
            self.pop[i].position = solution_particle.position.copy()
            self.pop[i].velocity = np.random.uniform(self.velMin, self.velMax, size=self.varSize)
            self.pop[i].fitness = solution_particle.fitness
            self.pop[i].best_fitness = solution_particle.best_fitness
            self.pop[i].best_position = solution_particle.best_position.copy()
            self.pop[i].prev_static_fitness = solution_particle.prev_static_fitness
            self.pop[i].prev_derivative = 0.0
            self.pop[i].num_equal_tt = solution_particle.num_equal_tt
            self.pop[i].num_gates = solution_particle.num_gates
            self.pop[i].num_no_gates = solution_particle.num_no_gates
            # Reset stagnation counter for the new particle
            self.pop[i].stagnation_counter = 0