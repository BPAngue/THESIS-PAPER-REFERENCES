import problem4
import numpy as np
import math
import copy

# --------------------------------------
# NEW: BPSO Gene/Bit Definitions
# --------------------------------------
# We must calculate how many bits each gene needs.
num_inputs = problem4.truth_table.num_inputs
num_rows = problem4.num_rows
num_outputs = problem4.truth_table.num_outputs
gate_types_count = len(problem4.GateType)

# 1. Gate Type (e.g., 4 types -> 2 bits)
BITS_PER_GATE = math.ceil(math.log2(gate_types_count))

# 2. Input Pointers (e.g., max 16 ptrs -> 4 bits)
# An input to the last gate (row 11) can point to 4(inputs) + 11(gates) = 15.
# Max pointer is num_inputs + num_rows - 1.
max_input_ptr = num_inputs + num_rows
BITS_PER_INPUT_PTR = math.ceil(math.log2(max_input_ptr + 1)) # +1 for 0-based

# 3. Output Pointers (e.g., max 16 ptrs -> 4 bits)
max_output_ptr = num_inputs + num_rows
BITS_PER_OUTPUT_PTR = math.ceil(math.log2(max_output_ptr + 1))

# --- This defines the *structure* of the 1D binary vector ---
GENE_BITS_MAP = []
# Add circuit_matrix genes (3 genes per row)
for i in range(num_rows):
    # Input 1: Can point up to (num_inputs + i)
    max_val = num_inputs + i
    bits_needed = math.ceil(math.log2(max_val + 1))
    GENE_BITS_MAP.append(bits_needed) # Gene for Input 1
    
    # Gate Type
    GENE_BITS_MAP.append(BITS_PER_GATE) # Gene for Gate Type
    
    # Input 2
    GENE_BITS_MAP.append(bits_needed) # Gene for Input 2

# Add output_array genes
for _ in range(num_outputs):
    GENE_BITS_MAP.append(BITS_PER_OUTPUT_PTR)

# This is the new nVar: the *total number of bits* to optimize
nVar_bits = sum(GENE_BITS_MAP)

# --------------------------------------
# Particle Class (BPSO)
# --------------------------------------
class Particle:
    """Holds the state for a single BINARY particle."""
    def __init__(self, varsize_bits):
        # Position is now a vector of probabilities (0.0 to 1.0)
        self.position_prob = np.random.uniform(0.0, 1.0, size=varsize_bits)
        # Velocity is for changing these probabilities
        self.velocity = np.random.uniform(-0.1, 0.1, size=varsize_bits)
        
        # pbest is the *probability vector* that gave the best fitness
        self.best_position_prob: np.ndarray = self.position_prob.copy()
        
        # The *actual* integer genes that were last evaluated
        self.integer_genes = np.zeros(len(GENE_BITS_MAP), dtype=int)
        
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
    return 1 / (1 + np.exp(-v))

def decode_particle_to_integers(particle: Particle):
    """
    Converts a particle's probability vector into a vector of integers.
    This is the "dice roll" step.
    """
    bit_string = (np.random.rand(nVar_bits) < particle.position_prob).astype(int)
    
    bit_idx = 0
    integer_genes = []
    
    # This loop reconstructs the integers from the bits
    for num_bits in GENE_BITS_MAP:
        if num_bits == 0: # Should not happen, but safety
            integer_genes.append(0)
            continue
            
        gene_bits = bit_string[bit_idx : bit_idx + num_bits]
        
        # Convert bits to integer
        gene_int = 0
        for bit in gene_bits:
            gene_int = (gene_int << 1) | bit
            
        integer_genes.append(gene_int)
        bit_idx += num_bits
        
    particle.integer_genes = np.array(integer_genes, dtype=int)

def clamp_and_format_integers(particle: Particle):
    """
    Takes the raw "rolled" integers and clamps them to valid
    functional ranges (e.g., pointer 1-N, gate type 0-M).
    This is the *final* integer vector used by the fitness function.
    """
    int_genes = particle.integer_genes.copy()
    idx = 0
    gate_type_max = gate_types_count - 1
    
    # 1. Clamp Circuit Matrix
    for i in range(num_rows):
        input_max_inclusive = num_inputs + i
        
        # Clamp Input 1
        int_genes[idx] = max(1, int_genes[idx] % (input_max_inclusive + 1))
        idx += 1
        
        # Clamp Gate Type
        int_genes[idx] = int_genes[idx] % (gate_type_max + 1)
        idx += 1
        
        # Clamp Input 2
        int_genes[idx] = max(1, int_genes[idx] % (input_max_inclusive + 1))
        idx += 1

    # 2. Clamp Output Array
    output_max_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        int_genes[idx] = max(1, int_genes[idx] % (output_max_inclusive + 1))
        idx += 1
        
    # 'int_genes' is now the 1D integer vector (size 39)
    # that problem4.py's 'decode_particle' function expects.
    return int_genes

# --------------------------------------
# Fitness Function Wrapper
# --------------------------------------
def fitnessFunction(integer_gene_vector, prev_static_fitness, previous_derivative):
    """Wrapper for the problem4 fitness function."""
    return problem4.problem(integer_gene_vector, prev_static_fitness, previous_derivative)

# --------------------------------------
# Base PSO Solver Class (BPSO)
# --------------------------------------
class BasePSOSolver:
    """
    Represents a single, independent PSO "Thread" (or swarm).
    NOW USES A BINARY PSO (BPSO) MODEL.
    """
    def __init__(self, nPop, varSize_bits, pso_params):
        self.nPop = nPop
        self.varSize_bits = varSize_bits # This is the bit-length
        self.w = pso_params['w']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']
        
        # BPSO uses different velocity clamping
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
            # 1. "Roll the dice" to get integers
            decode_particle_to_integers(particle)
            # 2. Clamp integers to valid ranges
            final_genes = clamp_and_format_integers(particle)
            
            # 3. Evaluate
            particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
            particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                final_genes, 
                particle.prev_static_fitness, 
                particle.prev_derivative
            )

            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position_prob = particle.position_prob.copy() # Save the probabilities
                particle.stagnation_counter = 0

            if particle.best_fitness > self.gbest_tracker.fitness:
                self.gbest_tracker = copy.deepcopy(particle)
                self.gbest_tracker.integer_genes = final_genes # Store the best *actual* circuit

    def run_iterations(self, num_iterations):
        """Run the standard LBEST B-PSO for a number of iterations."""
        for _ in range(num_iterations):
            for i, particle in enumerate(self.pop):
                
                # Check for particle stagnation
                if particle.stagnation_counter > self.particle_stagnation_limit:
                    # Re-initialize *probabilities*
                    particle.position_prob = np.random.uniform(0.0, 1.0, size=self.varSize_bits)
                    particle.velocity = np.random.uniform(-0.1, 0.1, size=self.varSize_bits)
                    particle.fitness = -math.inf
                    particle.best_fitness = -math.inf
                    particle.stagnation_counter = 0
                
                # Find LBest
                lbest_particle = particle
                for j in range(1, self.half_k + 1):
                    neighbor_idx_up = (i + j) % self.nPop
                    if self.pop[neighbor_idx_up].best_fitness > lbest_particle.best_fitness:
                        lbest_particle = self.pop[neighbor_idx_up]
                    neighbor_idx_down = (i - j + self.nPop) % self.nPop
                    if self.pop[neighbor_idx_down].best_fitness > lbest_particle.best_fitness:
                        lbest_particle = self.pop[neighbor_idx_down]

                # BPSO Velocity Update (on probabilities)
                r1 = np.random.random(size=self.varSize_bits)
                r2 = np.random.random(size=self.varSize_bits)
                
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position_prob - particle.position_prob)) + \
                                    (self.c2 * r2 * (lbest_particle.best_position_prob - particle.position_prob))

                particle.velocity = np.clip(particle.velocity, self.velMin_BPSO, self.velMax_BPSO)
                
                # BPSO Position (Probability) Update
                # We update the *probability* by squashing the velocity
                particle.position_prob = sigmoid(particle.velocity)
                
                # Evaluate
                decode_particle_to_integers(particle)
                final_genes = clamp_and_format_integers(particle)
                
                particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
                particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
                    final_genes, 
                    particle.prev_static_fitness, 
                    particle.prev_derivative
                )

                # Update PBest and Stagnation Counter
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position_prob = particle.position_prob.copy() # Save prob vector
                    particle.stagnation_counter = 0

                    if particle.best_fitness > self.gbest_tracker.fitness:
                        self.gbest_tracker = copy.deepcopy(particle)
                        self.gbest_tracker.integer_genes = final_genes # Save the winning circuit
                else:
                    particle.stagnation_counter += 1

    def get_gbest(self):
        """Returns this solver's best-found particle."""
        return copy.deepcopy(self.gbest_tracker)

    def inject_solution(self, solution_particle, percentage_to_replace):
        """
        Replaces the worst particles with the Voted_GBest.
        We inject the *probability vector* (pbest) of the winner.
        """
        self.pop.sort(key=lambda p: p.fitness)
        num_to_replace = int(self.nPop * percentage_to_replace)
        if num_to_replace == 0:
            return

        print(f"  [Injector] Replacing worst {num_to_replace} particles with new solution.")
        for i in range(num_to_replace):
            # Inject the winning *probability vector*
            self.pop[i].position_prob = solution_particle.best_position_prob.copy()
            # Reset velocity
            self.pop[i].velocity = np.random.uniform(-0.1, 0.1, size=self.varSize_bits)
            
            # Copy all fitness/stats from the winner
            self.pop[i].fitness = solution_particle.fitness
            self.pop[i].best_fitness = solution_particle.best_fitness
            self.pop[i].best_position_prob = solution_particle.best_position_prob.copy()
            self.pop[i].prev_static_fitness = solution_particle.prev_static_fitness
            self.pop[i].prev_derivative = 0.0
            self.pop[i].num_equal_tt = solution_particle.num_equal_tt
            self.pop[i].num_gates = solution_particle.num_gates
            self.pop[i].num_no_gates = solution_particle.num_no_gates
            self.pop[i].integer_genes = solution_particle.integer_genes.copy()
            self.pop[i].stagnation_counter = 0