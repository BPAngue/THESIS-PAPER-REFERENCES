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
        # Initialize with a VALID, random particle
        self.position: np.ndarray = create_random_particle(varsize)
        self.velocity = np.random.uniform(-1, 1, size=varsize)
        self.best_position: np.ndarray = self.position.copy()
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0

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

    # 1. Create Circuit Matrix (num_rows * 3)
    # Can point to primary inputs (1...num_inputs) or previous gates (num_inputs+1...num_inputs+i)
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        input_high_exclusive = input_high_inclusive + 1

        # Input 1 (Col 0)
        particle[idx] = np.random.randint(1, input_high_exclusive)
        idx += 1

        # Gate Type (Col 1)
        particle[idx] = np.random.randint(0, gate_type_high)
        idx += 1

        # Input 2 (Col 2)
        particle[idx] = np.random.randint(1, input_high_exclusive)
        idx += 1

    # 2. Create Output Array (num_outputs)
    # Can point to primary inputs (1...num_inputs) or ANY gate (num_inputs+1...num_inputs+num_rows)
    output_high_inclusive = num_inputs + num_rows
    output_high_exclusive = output_high_inclusive + 1

    for _ in range(num_outputs):
        particle[idx] = np.random.randint(1, output_high_exclusive)
        idx += 1

    return particle

def clamp_particle_position(position_float):
    """
    Clamps a particle's position vector (float) to valid integer ranges 
    according to the combinational encoding rules.
    """
    # Round first to get integer genes
    x = np.round(position_float).astype(int)
    idx = 0
    num_inputs = problem4.truth_table.num_inputs
    num_rows = problem4.num_rows
    num_outputs = problem4.truth_table.num_outputs
    gate_type_high = len(problem4.GateType) # Inclusive

    # 1. Clamp Circuit Matrix genes
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i

        # Clamp Input 1 (Col 0)
        # x[idx] = max(1, min(x[idx], input_high_inclusive))
        x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
        
        # Clamp Gate Type (Col 1)
        # x[idx] = max(0, min(x[idx], gate_type_high))
        x[idx] = x[idx] % gate_type_high
        idx += 1
        
        # Clamp Input 2 (Col 2)
        # x[idx] = max(1, min(x[idx], input_high_inclusive))
        x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1

    # 2. Clamp Output Array genes
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        # x[idx] = max(1, min(x[idx], output_high_inclusive))
        x[idx] = (x[idx] - 1) % output_high_inclusive + 1
        idx += 1
        
    return x

# --------------------------------------
# Fitness Function Wrapper
# --------------------------------------
def fitnessFunction(position):
    return problem4.problem(position)

# --------------------------------------
# Base PSO Solver Class
# --------------------------------------
class BasePSOSolver:
    """
    Represents a single, independent PSO "Thread" (or swarm).
    """
    def __init__(self, nPop, varSize, pso_params):
        self.nPop = nPop
        self.varSize = varSize
        self.w = pso_params['w']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']
        self.velMax = pso_params['velMax']
        self.velMin = pso_params['velMin']
        self.w_damp = pso_params['w_damp']

        # Each solver has its own population
        self.pop = [Particle(varSize) for _ in range(nPop)]
        self.gbest = Particle(varSize)
        self.gbest.fitness = -math.inf

        # Initialize population fitness
        self.evaluate_population()

    def evaluate_population(self):
        """Calculates fitness for all particles and updates gbest."""
        for particle in self.pop:
            
            particle.fitness, particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(particle.position)

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
                
                # --- UPDATED: Evaluate new position (4 return values) ---
                (particle.fitness, 
                 particle.num_equal_tt, 
                 particle.num_gates, 
                 particle.num_no_gates) = fitnessFunction(particle.position)

                # Update pbest
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                # Update gbest
                if particle.best_fitness > self.gbest.fitness:
                    self.gbest = copy.deepcopy(particle)
            
            self.w = self.w * self.w_damp 

    def get_gbest(self):
        """Returns this solver's best-found particle."""
        return copy.deepcopy(self.gbest)
    
    def force_gbest_replacement(self, new_gbest):
        """
        Forces the solver to accept a new global best, even if it is worse.
        This is for the specific Ring Topology requested.
        """
        self.gbest = copy.deepcopy(new_gbest)