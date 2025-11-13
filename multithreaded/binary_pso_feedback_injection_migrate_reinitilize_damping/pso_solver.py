import problem4
import numpy as np
import math
import copy

# --------------------------------------
# BPSO Component/Bit Definitions
# --------------------------------------
num_inputs = problem4.truth_table.num_inputs
num_rows = problem4.num_rows
num_outputs = problem4.truth_table.num_outputs
gate_types_count = len(problem4.GateType)

BITS_PER_GATE = math.ceil(math.log2(gate_types_count))
max_input_ptr = num_inputs + num_rows
BITS_PER_INPUT_PTR = math.ceil(math.log2(max_input_ptr + 1))
max_output_ptr = num_inputs + num_rows
BITS_PER_OUTPUT_PTR = math.ceil(math.log2(max_output_ptr + 1))

COMPONENT_BITS_MAP = []
for i in range(num_rows):
    max_val = num_inputs + i
    bits_needed = math.ceil(math.log2(max_val + 1))
    COMPONENT_BITS_MAP.append(bits_needed)
    COMPONENT_BITS_MAP.append(BITS_PER_GATE)
    COMPONENT_BITS_MAP.append(bits_needed)
for _ in range(num_outputs):
    COMPONENT_BITS_MAP.append(BITS_PER_OUTPUT_PTR)
nVar_bits = sum(COMPONENT_BITS_MAP)

# --------------------------------------
# Particle Class (BPSO)
# --------------------------------------
class Particle:
    """Holds the state for a single BINARY particle."""
    def __init__(self, varsize_bits):
        self.position_prob = np.random.uniform(0.0, 1.0, size=varsize_bits)
        self.velocity = np.random.uniform(-0.1, 0.1, size=varsize_bits)
        self.best_position_prob: np.ndarray = self.position_prob.copy()
        self.decoded_position = np.zeros(len(COMPONENT_BITS_MAP), dtype=int)
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        self.stagnation_counter = 0
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0
        self.prev_static_fitness = 0.0
        self.prev_derivative = 0.0

# --------------------------------------
# BPSO Encoding/Decoding Functions
# --------------------------------------

def sigmoid(v):
    """Sigmoid function to squash velocity into a probability."""
    v_clipped = np.clip(v, -10, 10)
    return 1 / (1 + np.exp(-v_clipped))

def decode_particle_to_integers(particle: Particle):
    """Converts a particle's probability vector into a vector of integers."""
    bit_string = (np.random.rand(nVar_bits) < particle.position_prob).astype(int)
    bit_idx = 0
    decoded_components = []
    
    for num_bits in COMPONENT_BITS_MAP:
        if num_bits == 0:
            decoded_components.append(0)
            continue
        component_bits = bit_string[bit_idx : bit_idx + num_bits]
        component_int = 0
        for bit in component_bits:
            component_int = (component_int << 1) | bit
        decoded_components.append(component_int)
        bit_idx += num_bits
        
    particle.decoded_position = np.array(decoded_components, dtype=int)

def clamp_and_format_integers(particle: Particle):
    """Clamps the raw "rolled" integers to valid functional ranges."""
    int_components = particle.decoded_position.copy()
    idx = 0
    gate_type_max = gate_types_count - 1
    
    for i in range(num_rows):
        input_max_inclusive = num_inputs + i
        
        int_components[idx] = max(1, int_components[idx] % (input_max_inclusive + 1))
        if int_components[idx] == 0: int_components[idx] = 1
        idx += 1
        
        int_components[idx] = int_components[idx] % (gate_type_max + 1)
        idx += 1
        
        int_components[idx] = max(1, int_components[idx] % (input_max_inclusive + 1))
        if int_components[idx] == 0: int_components[idx] = 1
        idx += 1

    output_max_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        int_components[idx] = max(1, int_components[idx] % (output_max_inclusive + 1))
        if int_components[idx] == 0: int_components[idx] = 1
        idx += 1
        
    return int_components

# --------------------------------------
# Fitness Function Wrapper
# --------------------------------------
def fitnessFunction(decoded_position_vector, prev_static_fitness, previous_derivative):
    """Wrapper for the problem4 fitness function."""
    return problem4.problem(decoded_position_vector, prev_static_fitness, previous_derivative)

# --------------------------------------
# Base PSO Solver Class (BPSO)
# --------------------------------------
class BasePSOSolver:
    """
    Represents a single, independent PSO "Thread" (or swarm).
    Uses a BINARY PSO (BPSO) MODEL.
    """
    def __init__(self, nPop, varSize_bits, pso_params):
        self.nPop = nPop
        self.varSize_bits = varSize_bits
        self.w = pso_params['w']
        # --- [NEW] Store damping parameters ---
        self.w_damp = pso_params['w_damp']
        self.w_min = pso_params['w_min']
        # --- [END NEW] ---
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']
        
        self.velMax_BPSO = 4.0
        self.velMin_BPSO = -4.0
        
        self.neighborhood_size = 5
        self.half_k = self.neighborhood_size // 2
        self.particle_stagnation_limit = 50 
        
        self.pop = [Particle(varSize_bits) for _ in range(nPop)]
        self.gbest_tracker = Particle(varSize_bits)
        self.gbest_tracker.fitness = -math.inf
        
        self.evaluate_population()

    def evaluate_population(self):
        """Calculates fitness for all particles and updates gbest_tracker."""
        for particle in self.pop:
            decode_particle_to_integers(particle)
            final_decoded_position = clamp_and_format_integers(particle)
            
            particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
            particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                final_decoded_position, 
                particle.prev_static_fitness, 
                particle.prev_derivative
            )

            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position_prob = particle.position_prob.copy()
                particle.stagnation_counter = 0

            if particle.best_fitness > self.gbest_tracker.fitness:
                self.gbest_tracker = copy.deepcopy(particle)
                self.gbest_tracker.decoded_position = final_decoded_position

    def run_iterations(self, num_iterations):
        """Run the standard LBEST B-PSO for a number of iterations."""
        for _ in range(num_iterations):
            for i, particle in enumerate(self.pop):
                
                if particle.stagnation_counter > self.particle_stagnation_limit:
                    particle.position_prob = np.random.uniform(0.0, 1.0, size=self.varSize_bits)
                    particle.velocity = np.random.uniform(-0.1, 0.1, size=self.varSize_bits)
                    particle.fitness = -math.inf
                    particle.best_fitness = -math.inf
                    particle.stagnation_counter = 0
                
                lbest_particle = particle
                for j in range(1, self.half_k + 1):
                    neighbor_idx_up = (i + j) % self.nPop
                    if self.pop[neighbor_idx_up].best_fitness > lbest_particle.best_fitness:
                        lbest_particle = self.pop[neighbor_idx_up]
                    neighbor_idx_down = (i - j + self.nPop) % self.nPop
                    if self.pop[neighbor_idx_down].best_fitness > lbest_particle.best_fitness:
                        lbest_particle = self.pop[neighbor_idx_down]

                r1 = np.random.random(size=self.varSize_bits)
                r2 = np.random.random(size=self.varSize_bits)
                
                # --- [MODIFIED] Use the *current* self.w ---
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position_prob - particle.position_prob)) + \
                                    (self.c2 * r2 * (lbest_particle.best_position_prob - particle.position_prob))
                # --- [END MODIFIED] ---

                particle.velocity = np.clip(particle.velocity, self.velMin_BPSO, self.velMax_BPSO)
                particle.position_prob = sigmoid(particle.velocity)
                
                decode_particle_to_integers(particle)
                final_decoded_position = clamp_and_format_integers(particle)
                
                particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
                particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                    final_decoded_position, 
                    particle.prev_static_fitness, 
                    particle.prev_derivative
                )

                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position_prob = particle.position_prob.copy()
                    particle.stagnation_counter = 0

                    if particle.best_fitness > self.gbest_tracker.fitness:
                        self.gbest_tracker = copy.deepcopy(particle)
                        self.gbest_tracker.decoded_position = final_decoded_position
                else:
                    particle.stagnation_counter += 1
            
            # --- [NEW] Apply inertia weight damping once per iteration ---
            if self.w > self.w_min:
                self.w = self.w * self.w_damp
            # --- [END NEW] ---

    def get_gbest(self):
        """Returns this solver's best-found particle."""
        return copy.deepcopy(self.gbest_tracker)

    def inject_solution(self, solution_particle, percentage_to_replace):
        """Replaces the worst particles with the Voted_GBest."""
        self.pop.sort(key=lambda p: p.fitness)
        num_to_replace = int(self.nPop * percentage_to_replace)
        if num_to_replace == 0:
            return

        print(f"  [Injector] Replacing worst {num_to_replace} particles with new solution.")
        for i in range(num_to_replace):
            self.pop[i].position_prob = solution_particle.best_position_prob.copy()
            self.pop[i].velocity = np.random.uniform(-0.1, 0.1, size=self.varSize_bits)
            self.pop[i].fitness = solution_particle.fitness
            self.pop[i].best_fitness = solution_particle.best_fitness
            self.pop[i].best_position_prob = solution_particle.best_position_prob.copy()
            self.pop[i].prev_static_fitness = solution_particle.prev_static_fitness
            self.pop[i].prev_derivative = 0.0
            self.pop[i].num_equal_tt = solution_particle.num_equal_tt
            self.pop[i].num_gates = solution_particle.num_gates
            self.pop[i].num_no_gates = solution_particle.num_no_gates
            self.pop[i].decoded_position = solution_particle.decoded_position.copy()
            self.pop[i].stagnation_counter = 0