import problem4
import numpy as np
import math
import copy

# --------------------------------------
# Particle Class
# --------------------------------------
class Particle:
    """Holds the state for a single particle."""
    def __init__(self, varsize):
        self.position: np.ndarray = create_random_particle(varsize)
        self.velocity = np.random.uniform(-1, 1, size=varsize) # Initial velocity
        self.best_position: np.ndarray = self.position.copy()
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0
        self.prev_static_fitness = 0.0
        self.prev_derivative = 0.0

# --------------------------------------
# Encoding Helper Functions
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
# Fitness Function Wrapper
# --------------------------------------
def fitnessFunction(position, prev_static_fitness, previous_derivative):
    """Wrapper for the problem4 fitness function."""
    return problem4.problem(position, prev_static_fitness, previous_derivative)

# --------------------------------------
# Base PSO Solver Class
# --------------------------------------
class BasePSOSolver:
    """
    Represents a single, independent PSO "Thread" (or swarm).
    This class contains the logic from your `pso_circuit_design_v2.py`.
    """
    def __init__(self, nPop, varSize, pso_params):
        self.nPop = nPop
        self.varSize = varSize
        self.w = pso_params['w']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']
        self.velMax = pso_params['velMax']
        self.velMin = pso_params['velMin']
        
        # Each solver has its own population
        self.pop = [Particle(varSize) for _ in range(nPop)]
        self.gbest = Particle(varSize)
        self.gbest.fitness = -math.inf
        
        # Initialize population fitness
        self.evaluate_population()

    def evaluate_population(self):
        """Calculates fitness for all particles and updates gbest."""
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

            if particle.best_fitness > self.gbest.fitness:
                self.gbest = copy.deepcopy(particle)

    def run_iterations(self, num_iterations):
        """Run the standard GBest PSO for a number of iterations."""
        for _ in range(num_iterations):
            for particle in self.pop:
                # Update velocity (standard gbest PSO)
                r1 = np.random.random(size=self.varSize)
                r2 = np.random.random(size=self.varSize)
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position - particle.position)) + \
                                    (self.c2 * r2 * (self.gbest.position - particle.position))

                particle.velocity = np.clip(particle.velocity, self.velMin, self.velMax)
                
                # Update position
                particle.position = particle.position + particle.velocity
                particle.position = clamp_particle_position(particle.position)
                
                # Evaluate new position
                particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
                particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                    particle.position, 
                    particle.prev_static_fitness, 
                    particle.prev_derivative
                )

                # Update pbest
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                    # Update gbest
                    if particle.best_fitness > self.gbest.fitness:
                        self.gbest = copy.deepcopy(particle)

    def get_gbest(self):
        """Returns this solver's best-found particle."""
        return copy.deepcopy(self.gbest)

    def inject_solution(self, solution_particle, percentage_to_replace):
        """
        Implements the "Feedback" step.
        Replaces the worst particles with the Voted_GBest.
        """
        # Sort population by fitness (worst first)
        self.pop.sort(key=lambda p: p.fitness)
        
        # Calculate how many to replace
        num_to_replace = int(self.nPop * percentage_to_replace)
        if num_to_replace == 0:
            return

        print(f"  [Injector] Replacing worst {num_to_replace} particles with new Voted_GBest.")
        for i in range(num_to_replace):
            # Replace position and velocity
            self.pop[i].position = solution_particle.position.copy()
            self.pop[i].velocity = np.random.uniform(self.velMin, self.velMax, size=self.varSize)
            
            # Reset fitness to the new solution's fitness
            self.pop[i].fitness = solution_particle.fitness
            self.pop[i].best_fitness = solution_particle.best_fitness
            self.pop[i].best_position = solution_particle.best_position.copy()
            
            # Reset dynamic fitness trackers
            self.pop[i].prev_static_fitness = solution_particle.prev_static_fitness
            self.pop[i].prev_derivative = 0.0 # Reset derivative to avoid jump
            
            # Update stats
            self.pop[i].num_equal_tt = solution_particle.num_equal_tt
            self.pop[i].num_gates = solution_particle.num_gates
            self.pop[i].num_no_gates = solution_particle.num_no_gates