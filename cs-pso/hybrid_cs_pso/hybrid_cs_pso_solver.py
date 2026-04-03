import problem4
import numpy as np
import math
import copy

# --------------------------------------
# Chaotic Map Functions
# --------------------------------------
def logistic_map(alpha, mu=4.0):
    """Logistic map equation for chaos generation."""
    return mu * alpha * (1 - alpha)

def initialize_chaotic_variable():
    """Initialize a chaotic variable avoiding problematic values."""
    while True:
        alpha = np.random.random()
        if alpha not in [0, 0.25, 0.5, 0.75, 1.0] and 0.01 < alpha < 0.99:
            return alpha

# --------------------------------------
# Particle Class
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
        
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * input_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], input_high_inclusive))
        idx += 1
        
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * gate_type_high)
        particle.position[idx] = max(0, min(particle.position[idx], gate_type_high - 1))
        idx += 1
        
        particle.chaotic_vars[idx] = logistic_map(particle.chaotic_vars[idx])
        particle.position[idx] = int(particle.chaotic_vars[idx] * input_high_inclusive) + 1
        particle.position[idx] = max(1, min(particle.position[idx], input_high_inclusive))
        idx += 1
    
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
        
        if col_idx == 0:
            new_val = int(particle.chaotic_vars[gene_idx] * input_high_inclusive) + 1
            return max(1, min(new_val, input_high_inclusive))
        elif col_idx == 1:
            new_val = int(particle.chaotic_vars[gene_idx] * gate_type_high)
            return max(0, min(new_val, gate_type_high - 1))
        else:
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
        
        if x[idx] < 1 or x[idx] > input_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
        
        if x[idx] < 0 or x[idx] >= gate_type_high:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = x[idx] % gate_type_high
        idx += 1
        
        if x[idx] < 1 or x[idx] > input_high_inclusive:
            if np.random.random() < chaos_intensity:
                x[idx] = chaos_perturb_position(particle, num_inputs, num_rows, num_outputs, idx)
            else:
                x[idx] = (x[idx] - 1) % input_high_inclusive + 1
        idx += 1
    
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
# Local Hill Climbing Implementation
# --------------------------------------
class LocalHillClimber:
    """
    Local hill climbing for fine-tuning near-optimal solutions.
    
    Strategies:
    1. Single-gene flip: Try changing one gene at a time
    2. Gate-type sweep: Try all gate types for critical gates
    3. Output pointer optimization: Try all valid output pointers
    4. Greedy improvement: Accept any improvement immediately
    """
    
    def __init__(self, num_inputs, num_rows, num_outputs):
        self.num_inputs = num_inputs
        self.num_rows = num_rows
        self.num_outputs = num_outputs
        self.varSize = (num_rows * 3) + num_outputs
        self.gate_type_high = len(problem4.GateType)
        
    def hill_climb(self, particle, max_iterations=50, strategy='comprehensive'):
        """
        Apply local hill climbing to improve a particle.
        
        Args:
            particle: Particle to improve
            max_iterations: Maximum iterations of hill climbing
            strategy: 'comprehensive', 'gate_focused', or 'output_focused'
        
        Returns:
            improved: True if particle was improved
            num_improvements: Number of improvements found
        """
        initial_fitness = particle.fitness
        num_improvements = 0
        
        for iteration in range(max_iterations):
            improved_this_iteration = False
            
            if strategy == 'comprehensive':
                # Try all improvement strategies
                improved_this_iteration = (
                    self._single_gene_flip(particle) or
                    self._gate_type_sweep(particle) or
                    self._output_pointer_optimization(particle)
                )
            elif strategy == 'gate_focused':
                improved_this_iteration = (
                    self._gate_type_sweep(particle) or
                    self._single_gene_flip(particle)
                )
            elif strategy == 'output_focused':
                improved_this_iteration = (
                    self._output_pointer_optimization(particle) or
                    self._single_gene_flip(particle)
                )
            
            if improved_this_iteration:
                num_improvements += 1
            else:
                # No improvement found, stop early
                break
        
        total_improvement = particle.fitness - initial_fitness
        return total_improvement > 0, num_improvements
    
    def _single_gene_flip(self, particle):
        """
        Try flipping each gene to its neighbors.
        Accept first improvement found (greedy).
        """
        current_fitness = particle.fitness
        
        # Randomize order to avoid bias
        gene_indices = np.random.permutation(self.varSize)
        
        for gene_idx in gene_indices:
            original_value = particle.position[gene_idx]
            
            # Try different values for this gene
            for new_value in self._get_valid_gene_values(gene_idx, original_value):
                particle.position[gene_idx] = new_value
                
                # Evaluate
                (particle.fitness, 
                 particle.num_equal_tt, 
                 particle.num_gates, 
                 particle.num_no_gates) = fitnessFunction(particle.position)
                
                # Accept improvement immediately (greedy)
                if particle.fitness > current_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                    return True  # Improved!
            
            # Restore original value if no improvement
            particle.position[gene_idx] = original_value
        
        # Restore original fitness
        (particle.fitness, 
         particle.num_equal_tt, 
         particle.num_gates, 
         particle.num_no_gates) = fitnessFunction(particle.position)
        
        return False  # No improvement
    
    def _gate_type_sweep(self, particle):
        """
        For each gate, try all 8 gate types.
        Focuses on critical gates that affect output.
        """
        current_fitness = particle.fitness
        circuit_matrix_size = self.num_rows * 3
        
        # Identify which gates are used in outputs
        output_start_idx = circuit_matrix_size
        output_pointers = particle.position[output_start_idx:]
        used_gates = set(output_pointers) - set(range(1, self.num_inputs + 1))
        
        # Prioritize gates that are used in outputs
        gate_indices = []
        for gate_num in used_gates:
            if gate_num > self.num_inputs:
                gate_row = gate_num - self.num_inputs - 1
                if 0 <= gate_row < self.num_rows:
                    gate_type_idx = gate_row * 3 + 1  # Gate type is column 1
                    gate_indices.append(gate_type_idx)
        
        # Also check other gates
        for i in range(self.num_rows):
            gate_type_idx = i * 3 + 1
            if gate_type_idx not in gate_indices:
                gate_indices.append(gate_type_idx)
        
        # Try each gate type
        for gate_type_idx in gate_indices:
            original_gate_type = particle.position[gate_type_idx]
            
            # Try all 8 gate types
            for gate_type in range(self.gate_type_high):
                if gate_type == original_gate_type:
                    continue
                
                particle.position[gate_type_idx] = gate_type
                
                # Evaluate
                (particle.fitness, 
                 particle.num_equal_tt, 
                 particle.num_gates, 
                 particle.num_no_gates) = fitnessFunction(particle.position)
                
                # Accept improvement immediately
                if particle.fitness > current_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                    return True
            
            # Restore original
            particle.position[gate_type_idx] = original_gate_type
        
        # Restore original fitness
        (particle.fitness, 
         particle.num_equal_tt, 
         particle.num_gates, 
         particle.num_no_gates) = fitnessFunction(particle.position)
        
        return False
    
    def _output_pointer_optimization(self, particle):
        """
        Try all valid pointers for each output.
        This is critical for combinational circuits.
        """
        current_fitness = particle.fitness
        circuit_matrix_size = self.num_rows * 3
        output_high_inclusive = self.num_inputs + self.num_rows
        
        for output_idx in range(self.num_outputs):
            gene_idx = circuit_matrix_size + output_idx
            original_pointer = particle.position[gene_idx]
            
            # Try all valid pointers
            for pointer in range(1, output_high_inclusive + 1):
                if pointer == original_pointer:
                    continue
                
                particle.position[gene_idx] = pointer
                
                # Evaluate
                (particle.fitness, 
                 particle.num_equal_tt, 
                 particle.num_gates, 
                 particle.num_no_gates) = fitnessFunction(particle.position)
                
                # Accept improvement immediately
                if particle.fitness > current_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                    return True
            
            # Restore original
            particle.position[gene_idx] = original_pointer
        
        # Restore original fitness
        (particle.fitness, 
         particle.num_equal_tt, 
         particle.num_gates, 
         particle.num_no_gates) = fitnessFunction(particle.position)
        
        return False
    
    def _get_valid_gene_values(self, gene_idx, current_value):
        """
        Get valid alternative values for a gene.
        Returns nearby values to try (neighborhood search).
        """
        circuit_matrix_size = self.num_rows * 3
        
        if gene_idx < circuit_matrix_size:
            # Circuit matrix gene
            row_idx = gene_idx // 3
            col_idx = gene_idx % 3
            
            if col_idx == 0 or col_idx == 2:  # Input pointers
                input_high_inclusive = self.num_inputs + row_idx
                # Try nearby values and random values
                candidates = []
                
                # Nearby values
                for delta in [-2, -1, 1, 2]:
                    val = current_value + delta
                    if 1 <= val <= input_high_inclusive:
                        candidates.append(val)
                
                # Random values
                for _ in range(3):
                    candidates.append(np.random.randint(1, input_high_inclusive + 1))
                
                return list(set(candidates))  # Remove duplicates
            
            else:  # Gate type (col_idx == 1)
                # Try all gate types
                return [g for g in range(self.gate_type_high) if g != current_value]
        
        else:
            # Output pointer
            output_high_inclusive = self.num_inputs + self.num_rows
            # Try all possible output pointers
            return [p for p in range(1, output_high_inclusive + 1) if p != current_value]

# --------------------------------------
# Hybrid CS-PSO Solver Class
# --------------------------------------
class HybridCSPSOSolver:
    """
    Hybrid CS-PSO with Local Hill Climbing.
    
    Strategy:
    1. Use CS-PSO for global exploration
    2. When close to solution (14+/16 correct), apply hill climbing
    3. Hill climb gbest and top particles
    4. Continue PSO with improved particles
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
        self.chaos_intensity = 0.5
        self.stagnation_threshold = 3
        self.global_stagnation = 0
        
        # Hill climbing control
        self.hill_climber = LocalHillClimber(num_inputs, num_rows, num_outputs)
        self.hill_climb_threshold = 14  # Apply hill climbing when >= 14/16 correct
        self.hill_climb_frequency = 5   # Every 5 iterations
        self.total_hill_climbs = 0
        self.successful_hill_climbs = 0
        
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
        """Calculates fitness for all particles and updates gbest."""
        improved_gbest = False
        
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
        
        if improved_gbest:
            self.global_stagnation = 0
    
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
        """Dynamically adjust chaos intensity based on global stagnation."""
        if self.global_stagnation > 20:
            self.chaos_intensity = min(1.0, self.chaos_intensity + 0.05)
        elif self.global_stagnation > 10:
            self.chaos_intensity = min(0.9, self.chaos_intensity + 0.03)
        elif self.global_stagnation > 5:
            self.chaos_intensity = min(0.7, self.chaos_intensity + 0.02)
        else:
            self.chaos_intensity = max(0.4, self.chaos_intensity - 0.01)
    
    def apply_hill_climbing(self, iteration):
        """
        Apply hill climbing to promising particles.
        
        Strategy:
        1. Always hill climb gbest if close to solution
        2. Hill climb top 20% particles periodically
        3. Use appropriate strategy based on progress
        """
        improvements = 0
        
        # Always try to improve gbest if close to solution
        if self.gbest.num_equal_tt >= self.hill_climb_threshold:
            # Choose strategy based on how close we are
            if self.gbest.num_equal_tt >= problem4.truth_table.total_outputs:
                strategy = 'comprehensive'  # Perfect solution, optimize for gates
            else:
                strategy = 'output_focused'  # Close to perfect, focus on outputs
            
            self.total_hill_climbs += 1
            improved, num_improvements = self.hill_climber.hill_climb(
                self.gbest, 
                max_iterations=50, 
                strategy=strategy
            )
            
            if improved:
                self.successful_hill_climbs += 1
                improvements += num_improvements
        
        # Periodically hill climb top particles
        if iteration % self.hill_climb_frequency == 0:
            # Sort particles by fitness
            sorted_particles = sorted(self.pop, key=lambda p: p.fitness, reverse=True)
            top_count = max(1, int(0.2 * self.nPop))  # Top 20%
            
            for particle in sorted_particles[:top_count]:
                if particle.num_equal_tt >= self.hill_climb_threshold:
                    self.total_hill_climbs += 1
                    improved, num_improvements = self.hill_climber.hill_climb(
                        particle, 
                        max_iterations=20,  # Less iterations for regular particles
                        strategy='gate_focused'
                    )
                    
                    if improved:
                        self.successful_hill_climbs += 1
                        improvements += num_improvements
                        
                        # Update gbest if this particle improved
                        if particle.fitness > self.gbest.fitness:
                            self.gbest = copy.deepcopy(particle)
        
        return improvements
    
    def run_iterations(self, num_iterations):
        """Run hybrid CS-PSO with hill climbing."""
        for iteration in range(num_iterations):
            old_gbest_fitness = self.gbest.fitness
            
            # Adapt chaos intensity
            self.adapt_chaos_intensity()
            
            # Standard PSO updates
            for particle in self.pop:
                r1 = np.random.random(size=self.varSize)
                r2 = np.random.random(size=self.varSize)
                
                particle.velocity = (
                    self.w * particle.velocity +
                    self.c1 * r1 * (particle.best_position - particle.position) +
                    self.c2 * r2 * (self.gbest.position - particle.position)
                )
                
                particle.velocity = np.clip(particle.velocity, self.velMin, self.velMax)
                
                new_position_float = particle.position + particle.velocity
                particle.position = clamp_particle_position_chaotic(
                    new_position_float, 
                    particle, 
                    self.num_inputs, 
                    self.num_rows, 
                    self.num_outputs,
                    chaos_intensity=self.chaos_intensity
                )
                
                # Mutate if stuck
                if particle.stagnation_count > self.stagnation_threshold:
                    self.mutate_particle(particle)
                    particle.stagnation_count = 0
                
                # Evaluate
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
            
            # Apply hill climbing (HYBRID PART!)
            hill_climb_improvements = self.apply_hill_climbing(iteration)
            
            # Update global stagnation
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
            'diversity': self.get_diversity_metric(),
            'total_hill_climbs': self.total_hill_climbs,
            'successful_hill_climbs': self.successful_hill_climbs,
            'hill_climb_success_rate': self.successful_hill_climbs / max(1, self.total_hill_climbs)
        }
