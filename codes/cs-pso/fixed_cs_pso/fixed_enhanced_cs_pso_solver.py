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
    """
    return mu * alpha * (1 - alpha)

def initialize_chaotic_variable():
    """Initialize a chaotic variable avoiding problematic values."""
    while True:
        alpha = np.random.random()
        if alpha not in [0, 0.25, 0.5, 0.75, 1.0] and 0.01 < alpha < 0.99:
            return alpha

# --------------------------------------
# Enhanced Particle Class
# --------------------------------------
class ChaoticParticle:
    """Holds the state for a single particle in CS-PSO."""
    def __init__(self, varsize):
        self.position: np.ndarray = np.zeros(varsize, dtype=int)
        self.velocity = np.zeros(varsize, dtype=float)
        self.best_position: np.ndarray = np.zeros(varsize, dtype=int)
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0
        self.chaotic_vars = np.array([initialize_chaotic_variable() for _ in range(varsize)])
        self.stagnation_count = 0

# --------------------------------------
# Helper Functions
# --------------------------------------
def chaos_initialize_particle(particle, num_inputs, num_rows, num_outputs):
    """Initialize particle position using chaos search method."""
    idx = 0
    gate_type_high = len(problem4.GateType)
    
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        
        # Input 1
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * input_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], input_high_inclusive))
        idx += 1
        
        # Gate Type
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * gate_type_high)
        particle.position[idx] = max(0, min(particle.position[idx], gate_type_high - 1))
        idx += 1
        
        # Input 2
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * input_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], input_high_inclusive))
        idx += 1
    
    # Output Array
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * output_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], output_high_inclusive))
        idx += 1
    
    particle.velocity = np.random.uniform(-1, 1, size=len(particle.position))
    particle.best_position = particle.position.copy()

def chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, gene_idx):
    """Perturb a specific gene using chaos search method."""
    gate_type_high = len(problem4.GateType)
    circuit_matrix_size = num_rows * 3
    
    particle.chaotic_vars[gene_idx] = logistic_map(particle.chaotic_vars[gene_idx])
    
    if gene_idx < circuit_matrix_size:
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
        output_high_inclusive = num_inputs + num_rows
        new_val = int(particle.chaotic_vars[gene_idx] * output_high_inclusive) + 1
        return max(1, min(new_val, output_high_inclusive))

def clamp_particle_position_chaotic(position_float, particle, num_inputs, num_rows, num_outputs, 
                                     chaos_intensity=1.0):
    """Clamps particle position using chaotic perturbation."""
    x = np.round(position_float).astype(int)
    idx = 0
    gate_type_high = len(problem4.GateType)
    
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        
        # Input 1
        if x[idx] < 1 or x[idx] > input_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
        
        # Gate Type
        if x[idx] < 0 or x[idx] >= gate_type_high:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = x[idx] % gate_type_high
        idx += 1
        
        # Input 2
        if x[idx] < 1 or x[idx] > input_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
    
    # Output Array
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        if x[idx] < 1 or x[idx] > output_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % output_high_inclusive + 1
        idx += 1
    
    return x

def fitnessFunction(position):
    return problem4.problem(position)

# --------------------------------------
# FIXED Enhanced CS-PSO Solver Class
# --------------------------------------
class FixedEnhancedCSPSOSolver:
    """
    FIXED Enhanced Chaotic Particle Swarm Optimization Solver.
    
    KEY FIX: global_stagnation now accumulates properly!
    This allows adaptive chaos to actually activate.
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
        self.chaos_intensity = 0.5  # Start higher (was 0.3)
        self.stagnation_threshold = 3  # Mutate sooner (was 5)
        self.global_stagnation = 0
        
        # Initialize population
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
        """
        FIXED: Calculates fitness for all particles and updates gbest.
        NOW PROPERLY TRACKS GLOBAL STAGNATION!
        """
        improved_gbest = False  # Track if we improved THIS evaluation
        
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
                improved_gbest = True
        
        # FIX: Only reset if we actually improved gbest
        if improved_gbest:
            self.global_stagnation = 0
        # Don't reset otherwise - let it accumulate!
    
    def mutate_particle(self, particle):
        """Apply mutation to a stagnant particle."""
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
        NOW ACTUALLY WORKS because global_stagnation accumulates!
        """
        if self.global_stagnation > 20:
            self.chaos_intensity = min(1.0, self.chaos_intensity + 0.05)
        elif self.global_stagnation > 10:
            self.chaos_intensity = min(0.9, self.chaos_intensity + 0.03)
        elif self.global_stagnation > 5:
            self.chaos_intensity = min(0.7, self.chaos_intensity + 0.02)
        else:
            self.chaos_intensity = max(0.4, self.chaos_intensity - 0.01)
    
    def run_iterations(self, num_iterations):
        """Run enhanced CS-PSO for a number of iterations."""
        for iteration in range(num_iterations):
            # Track if gbest improved this iteration
            old_gbest_fitness = self.gbest.fitness
            
            # Adapt chaos intensity based on global performance
            self.adapt_chaos_intensity()
            
            for particle in self.pop:
                # Update velocity
                r1 = np.random.random(size=self.varSize)
                r2 = np.random.random(size=self.varSize)
                
                particle.velocity = (
                    self.w * particle.velocity +
                    self.c1 * r1 * (particle.best_position - particle.position) +
                    self.c2 * r2 * (self.gbest.position - particle.position)
                )
                
                particle.velocity = np.clip(particle.velocity, self.velMin, self.velMax)
                
                # Update position with chaotic clamping
                new_position_float = particle.position + particle.velocity
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
            
            # Update global stagnation counter
            if self.gbest.fitness <= old_gbest_fitness:
                self.global_stagnation += 1
            else:
                self.global_stagnation = 0
            
            # Dampen inertia weight
            self.w = self.w * self.w_damp
    
    def get_gbest(self):
        """Returns this solver's best-found particle."""
        return copy.deepcopy(self.gbest)
    
    def force_gbest_replacement(self, new_gbest):
        """Forces the solver to accept a new global best."""
        if new_gbest.fitness > self.gbest.fitness:
            self.gbest = copy.deepcopy(new_gbest)
            self.global_stagnation = 0
        else:
            self.gbest = copy.deepcopy(new_gbest)
    
    def get_diversity_metric(self):
        """Calculate population diversity."""
        distances = []
        for particle in self.pop:
            dist = np.sum(particle.position != self.gbest.position)
            distances.append(dist)
        return np.mean(distances) if distances else 0
    
    def get_stats(self):
        """Get current solver statistics."""
        return {
            'chaos_intensity': self.chaos_intensity,
            'global_stagnation': self.global_stagnation,
            'diversity': self.get_diversity_metric()
        }
