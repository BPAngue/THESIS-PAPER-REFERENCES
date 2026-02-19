import problem4
import numpy as np
import math
import copy

# --------------------------------------
# Enhanced Chaotic Map Functions
# --------------------------------------
def logistic_map(alpha, mu=4.0):
    """
    Logistic map equation for chaos generation.
    alpha_n+1 = mu * alpha_n * (1 - alpha_n)
    
    Args:
        alpha: Current chaotic variable value (0 < alpha < 1)
        mu: Control parameter (typically 4.0 for full chaos)
    
    Returns:
        Next chaotic variable value
    """
    return mu * alpha * (1 - alpha)

def initialize_chaotic_variable():
    """
    Initialize a chaotic variable with a random value not equal to 
    0, 0.25, 0.5, 0.75, or 1 (to avoid eventually periodic behavior).
    """
    while True:
        alpha = np.random.random()
        # Avoid problematic initial values
        if alpha not in [0, 0.25, 0.5, 0.75, 1.0] and 0.01 < alpha < 0.99:
            return alpha

# --------------------------------------
# Enhanced Particle Class for CS-PSO
# --------------------------------------
class ChaoticParticle:
    """Holds the state for a single particle in CS-PSO."""
    def __init__(self, varsize):
        # Position will be initialized using chaos initialization
        self.position: np.ndarray = np.zeros(varsize, dtype=int)
        self.velocity = np.zeros(varsize, dtype=float)
        self.best_position: np.ndarray = np.zeros(varsize, dtype=int)
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0
        
        # Chaotic variables matrix for this particle (one per gene)
        self.chaotic_vars = np.array([initialize_chaotic_variable() for _ in range(varsize)])
        
        # Track stagnation
        self.stagnation_count = 0

# --------------------------------------
# CS-PSO Encoding Helper Functions
# --------------------------------------
def chaos_initialize_particle(particle, num_inputs, num_rows, num_outputs):
    """
    Initialize particle position using chaos search method.
    Uses logistic map to generate pseudorandom sequences with ergodicity.
    
    This replaces the random initialization in standard PSO.
    """
    idx = 0
    gate_type_high = len(problem4.GateType)
    
    # 1. Initialize Circuit Matrix (num_rows * 3)
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        
        # Input 1 (Col 0) - Use chaos to select
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        # Map chaotic value [0,1] to valid range [1, input_high_inclusive]
        particle.position[idx] = int(particle.chaotic_vars[idx] * input_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], input_high_inclusive))
        idx += 1
        
        # Gate Type (Col 1) - Use chaos to select
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * gate_type_high)
        particle.position[idx] = max(0, min(particle.position[idx], gate_type_high - 1))
        idx += 1
        
        # Input 2 (Col 2) - Use chaos to select
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * input_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], input_high_inclusive))
        idx += 1
    
    # 2. Initialize Output Array (num_outputs)
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * output_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], output_high_inclusive))
        idx += 1
    
    # Initialize velocity and best_position
    particle.velocity = np.random.uniform(-1, 1, size=len(particle.position))
    particle.best_position = particle.position.copy()

def chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, gene_idx):
    """
    Perturb a specific gene using chaos search method.
    This replaces simple clamping with chaotic exploration.
    
    Args:
        particle: The particle to perturb
        num_inputs: Number of primary inputs
        num_rows: Number of gate rows
        num_outputs: Number of outputs
        gene_idx: Index of the gene to perturb
    
    Returns:
        New value for the gene at gene_idx
    """
    gate_type_high = len(problem4.GateType)
    circuit_matrix_size = num_rows * 3
    
    # Update chaotic variable for this gene
    particle.chaotic_vars[gene_idx] = logistic_map(particle.chaotic_vars[gene_idx])
    
    # Determine which part of the encoding this gene belongs to
    if gene_idx < circuit_matrix_size:
        # Circuit matrix gene
        row_idx = gene_idx // 3
        col_idx = gene_idx % 3
        input_high_inclusive = num_inputs + row_idx
        
        if col_idx == 0:  # Input 1
            new_val = int(particle.chaotic_vars[gene_idx] * input_high_inclusive) + 1
            return max(1, min(new_val, input_high_inclusive))
        elif col_idx == 1:  # Gate Type
            new_val = int(particle.chaotic_vars[gene_idx] * gate_type_high)
            return max(0, min(new_val, gate_type_high - 1))
        else:  # Input 2
            new_val = int(particle.chaotic_vars[gene_idx] * input_high_inclusive) + 1
            return max(1, min(new_val, input_high_inclusive))
    else:
        # Output array gene
        output_high_inclusive = num_inputs + num_rows
        new_val = int(particle.chaotic_vars[gene_idx] * output_high_inclusive) + 1
        return max(1, min(new_val, output_high_inclusive))

def clamp_particle_position_chaotic(position_float, particle, num_inputs, num_rows, num_outputs, 
                                     chaos_intensity=1.0):
    """
    Clamps particle position using chaotic perturbation instead of simple modulo.
    This implements the chaos perturbing phase from the CS-PSO paper.
    
    Args:
        chaos_intensity: Controls how aggressively to apply chaos (0=never, 1=always when out of bounds)
    """
    x = np.round(position_float).astype(int)
    idx = 0
    gate_type_high = len(problem4.GateType)
    
    # 1. Clamp Circuit Matrix genes
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        
        # Input 1 (Col 0)
        if x[idx] < 1 or x[idx] > input_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
        
        # Gate Type (Col 1)
        if x[idx] < 0 or x[idx] >= gate_type_high:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = x[idx] % gate_type_high
        idx += 1
        
        # Input 2 (Col 2)
        if x[idx] < 1 or x[idx] > input_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
    
    # 2. Clamp Output Array genes
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        if x[idx] < 1 or x[idx] > output_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % output_high_inclusive + 1
        idx += 1
    
    return x

# --------------------------------------
# Fitness Function Wrapper
# --------------------------------------
def fitnessFunction(position):
    return problem4.problem(position)

# --------------------------------------
# Enhanced CS-PSO Solver Class
# --------------------------------------
class EnhancedCSPSOSolver:
    """
    Enhanced Chaotic Particle Swarm Optimization (CS-PSO) Solver.
    
    Improvements over basic CS-PSO:
    1. Adaptive chaos intensity based on stagnation
    2. Mutation operator for diversity maintenance
    3. Better handling of stuck particles
    4. Dynamic inertia weight adjustment
    """
    
    def __init__(self, nPop, varSize, pso_params, num_inputs, num_rows, num_outputs):
        self.nPop = nPop
        self.varSize = varSize
        self.num_inputs = num_inputs
        self.num_rows = num_rows
        self.num_outputs = num_outputs
        
        # PSO parameters
        self.w = pso_params['w']
        self.w_initial = pso_params['w']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']
        self.velMax = pso_params['velMax']
        self.velMin = pso_params['velMin']
        self.w_damp = pso_params['w_damp']
        
        # Chaos control parameters
        self.chaos_intensity = 0.5  # Start with moderate chaos
        self.stagnation_threshold = 3
        self.global_stagnation = 0
        
        # Initialize population with chaotic particles
        self.pop = []
        for _ in range(nPop):
            particle = ChaoticParticle(varSize)
            chaos_initialize_particle(particle, num_inputs, num_rows, num_outputs)
            self.pop.append(particle)
        
        # Global best
        self.gbest = ChaoticParticle(varSize)
        self.gbest.fitness = -math.inf
        
        # Initialize population fitness
        self.evaluate_population()
    
    def evaluate_population(self):
        """Calculates fitness for all particles and updates gbest."""
        for particle in self.pop:
            (particle.fitness, 
             particle.num_equal_tt, 
             particle.num_gates, 
             particle.num_no_gates) = fitnessFunction(particle.position)
            
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
                particle.stagnation_count = 0
            else:
                particle.stagnation_count += 1
            
            if particle.best_fitness > self.gbest.fitness:
                self.gbest = copy.deepcopy(particle)
                self.global_stagnation = 0
            else:
                self.global_stagnation += 1
    
    def mutate_particle(self, particle):
        """
        Apply mutation to a stagnant particle to increase diversity.
        Mutates 20-30% of genes using chaos.
        """
        num_genes_to_mutate = np.random.randint(
            int(0.2 * self.varSize), 
            int(0.3 * self.varSize)
        )
        genes_to_mutate = np.random.choice(self.varSize, num_genes_to_mutate, replace=False)
        
        for gene_idx in genes_to_mutate:
            particle.position[gene_idx] = chaos_perturb_position(
                particle, 
                self.num_inputs, 
                self.num_rows, 
                self.num_outputs, 
                gene_idx
            )
    
    def adapt_chaos_intensity(self):
        """
        Dynamically adjust chaos intensity based on global stagnation.
        More stagnation = more chaos for exploration.
        """
        if self.global_stagnation > 20:
            self.chaos_intensity = min(1.0, self.chaos_intensity + 0.05)
        elif self.global_stagnation > 10:
            self.chaos_intensity = min(0.8, self.chaos_intensity + 0.02)
        else:
            self.chaos_intensity = max(0.3, self.chaos_intensity - 0.01)
    
    def run_iterations(self, num_iterations):
        """
        Run enhanced CS-PSO for a number of iterations.
        
        Implements:
        - Standard PSO velocity update
        - Chaotic perturbation with adaptive intensity
        - Mutation for stuck particles
        - Adaptive chaos control
        """
        for iteration in range(num_iterations):
            # Adapt chaos intensity based on global performance
            self.adapt_chaos_intensity()
            
            for particle in self.pop:
                # Update velocity (standard PSO velocity update)
                r1 = np.random.random(size=self.varSize)
                r2 = np.random.random(size=self.varSize)
                
                particle.velocity = (
                    self.w * particle.velocity +
                    self.c1 * r1 * (particle.best_position - particle.position) +
                    self.c2 * r2 * (self.gbest.position - particle.position)
                )
                
                particle.velocity = np.clip(particle.velocity, self.velMin, self.velMax)
                
                # Update position
                new_position_float = particle.position + particle.velocity
                
                # Apply chaotic clamping (chaos perturbation phase)
                particle.position = clamp_particle_position_chaotic(
                    new_position_float, 
                    particle, 
                    self.num_inputs, 
                    self.num_rows, 
                    self.num_outputs,
                    chaos_intensity=self.chaos_intensity
                )
                
                # If particle is stuck, apply mutation
                if particle.stagnation_count > self.stagnation_threshold:
                    self.mutate_particle(particle)
                    particle.stagnation_count = 0
                
                # Evaluate new position
                (particle.fitness, 
                 particle.num_equal_tt, 
                 particle.num_gates, 
                 particle.num_no_gates) = fitnessFunction(particle.position)
                
                # Update personal best
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                    particle.stagnation_count = 0
                else:
                    particle.stagnation_count += 1
                
                # Update global best
                if particle.best_fitness > self.gbest.fitness:
                    self.gbest = copy.deepcopy(particle)
                    self.global_stagnation = 0
            
            # Dampen inertia weight
            self.w = self.w * self.w_damp
    
    def get_gbest(self):
        """Returns this solver's best-found particle."""
        return copy.deepcopy(self.gbest)
    
    def force_gbest_replacement(self, new_gbest):
        """
        Forces the solver to accept a new global best.
        Used for ring topology communication.
        """
        if new_gbest.fitness > self.gbest.fitness:
            self.gbest = copy.deepcopy(new_gbest)
            self.global_stagnation = 0
        else:
            # Even if not better, still update to encourage diversity
            self.gbest = copy.deepcopy(new_gbest)
    
    def get_diversity_metric(self):
        """
        Calculate population diversity as average distance from gbest.
        Higher = more diverse population.
        """
        distances = []
        for particle in self.pop:
            dist = np.sum(particle.position != self.gbest.position)
            distances.append(dist)
        return np.mean(distances) if distances else 0
