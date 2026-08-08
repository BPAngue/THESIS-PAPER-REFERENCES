import numpy as np
import math
import copy
import problem_def as problem_def
from dataclasses import dataclass
from typing import List, Tuple

# Import robust utilities
try:
    from clamp_utils import clamp_particle_position, create_random_particle
except ImportError:
    # Fallback to pso_solver if clamp_utils not available
    from pso_solver import clamp_particle_position, create_random_particle

# --------------------------------------
# Multi-Objective Particle Class
# --------------------------------------
class MOParticle:
    """Multi-objective particle with two objectives: correctness and simplicity"""
    def __init__(self, varsize):
        self.position = create_random_particle(varsize)
        self.velocity = np.random.uniform(-1, 1, size=varsize)
        self.best_position = self.position.copy()
        
        # Two objectives
        self.obj1 = -math.inf  # Correctness (maximize)
        self.obj2 = math.inf   # Gates used (minimize)
        self.best_obj1 = -math.inf
        self.best_obj2 = math.inf
        
        # Additional metrics
        self.num_equal_tt = 0
        self.num_gates = 0
        self.dominated = False
        self.crowding_distance = 0.0
        self.grid_index = -1

    def dominates(self, other):
        """Check if this particle dominates another (Pareto dominance)"""
        better_in_one = (self.obj1 > other.obj1) or (self.obj2 < other.obj2)
        not_worse_in_any = (self.obj1 >= other.obj1) and (self.obj2 <= other.obj2)
        return better_in_one and not_worse_in_any

# --------------------------------------
# Archive Management
# --------------------------------------
class ParetoArchive:
    """Maintains a bounded archive of non-dominated solutions"""
    def __init__(self, max_size, ngrid=20):
        self.max_size = max_size
        self.ngrid = ngrid
        self.archive: List[MOParticle] = []
        self.grid_limits = None
        
    def add(self, particle: MOParticle):
        """Add particle to archive and remove dominated solutions"""
        # Check if new particle is dominated by archive
        for arch_particle in self.archive:
            if arch_particle.dominates(particle):
                return  # Don't add dominated particle
        
        # Remove particles dominated by new particle
        self.archive = [p for p in self.archive if not particle.dominates(p)]
        
        # Add new particle
        new_particle = copy.deepcopy(particle)
        self.archive.append(new_particle)
        
        # Maintain size limit using crowding distance
        if len(self.archive) > self.max_size:
            self._prune_by_crowding()
    
    def _prune_by_crowding(self):
        """Remove particles with smallest crowding distance"""
        self._calculate_crowding_distances()
        self.archive.sort(key=lambda p: p.crowding_distance)
        excess = len(self.archive) - self.max_size
        self.archive = self.archive[excess:]
    
    def _calculate_crowding_distances(self):
        """Calculate crowding distance for each particle in archive"""
        n = len(self.archive)
        if n <= 2:
            for p in self.archive:
                p.crowding_distance = math.inf
            return
        
        # Initialize distances
        for p in self.archive:
            p.crowding_distance = 0.0
        
        # For each objective
        for obj_idx in range(2):
            # Sort by objective
            if obj_idx == 0:
                self.archive.sort(key=lambda p: p.obj1)
                obj_range = self.archive[-1].obj1 - self.archive[0].obj1
            else:
                self.archive.sort(key=lambda p: p.obj2)
                obj_range = self.archive[-1].obj2 - self.archive[0].obj2
            
            # Boundary points get infinite distance
            self.archive[0].crowding_distance = math.inf
            self.archive[-1].crowding_distance = math.inf
            
            # Calculate distances for intermediate points
            if obj_range > 0:
                for i in range(1, n - 1):
                    if obj_idx == 0:
                        distance = (self.archive[i + 1].obj1 - self.archive[i - 1].obj1) / obj_range
                    else:
                        distance = (self.archive[i + 1].obj2 - self.archive[i - 1].obj2) / obj_range
                    self.archive[i].crowding_distance += distance
    
    def update_grid(self):
        """Update hypercube grid for leader selection"""
        if len(self.archive) == 0:
            return
        
        # Calculate grid limits
        obj1_vals = [p.obj1 for p in self.archive]
        obj2_vals = [p.obj2 for p in self.archive]
        
        obj1_min, obj1_max = min(obj1_vals), max(obj1_vals)
        obj2_min, obj2_max = min(obj2_vals), max(obj2_vals)
        
        # Avoid division by zero
        if obj1_max - obj1_min < 1e-10:
            obj1_max = obj1_min + 1.0
        if obj2_max - obj2_min < 1e-10:
            obj2_max = obj2_min + 1.0
        
        self.grid_limits = {
            'obj1': np.linspace(obj1_min, obj1_max, self.ngrid + 1),
            'obj2': np.linspace(obj2_min, obj2_max, self.ngrid + 1)
        }
        
        # Assign grid indices
        grid_counts = {}
        for particle in self.archive:
            idx1 = np.digitize(particle.obj1, self.grid_limits['obj1']) - 1
            idx2 = np.digitize(particle.obj2, self.grid_limits['obj2']) - 1
            idx1 = max(0, min(self.ngrid - 1, idx1))
            idx2 = max(0, min(self.ngrid - 1, idx2))
            particle.grid_index = idx1 * self.ngrid + idx2
            grid_counts[particle.grid_index] = grid_counts.get(particle.grid_index, 0) + 1
        
        # Calculate quality (inverse of density)
        for particle in self.archive:
            particle.quality = 10.0 / grid_counts[particle.grid_index]
    
    def select_leader(self):
        """Select leader using roulette wheel based on grid quality"""
        if len(self.archive) == 0:
            return None
        
        self.update_grid()
        
        # Roulette wheel selection
        qualities = [p.quality for p in self.archive]
        total_quality = sum(qualities)
        
        if total_quality == 0:
            return self.archive[np.random.randint(len(self.archive))]
        
        probs = [q / total_quality for q in qualities]
        selected_idx = np.random.choice(len(self.archive), p=probs)
        return self.archive[selected_idx]
    
    def get_best_for_objective(self, objective='correctness'):
        """Get best particle for specific objective"""
        if len(self.archive) == 0:
            return None
        
        if objective == 'correctness':
            return max(self.archive, key=lambda p: p.obj1)
        else:  # simplicity
            return min(self.archive, key=lambda p: p.obj2)

# --------------------------------------
# Multi-Objective PSO Solver
# --------------------------------------
class MOPSOSolver:
    def __init__(self, nPop, varSize, params):
        self.nPop = nPop
        self.varSize = varSize
        
        # PSO parameters with adaptive learning factors
        self.w = params['w']
        self.w_damp = params['w_damp']
        self.c1_start = params.get('c1_start', 2.05)
        self.c2_start = params.get('c2_start', 2.05)
        self.velMax = params['velMax']
        self.velMin = params['velMin']
        
        # Mutation parameters
        self.mutation_rate = params.get('mutation_rate', 0.5)
        
        # Initialize population
        self.pop = [MOParticle(varSize) for _ in range(nPop)]
        
        # Archive for non-dominated solutions
        self.archive = ParetoArchive(max_size=params.get('archive_size', 100))
        
        # Evaluate initial population
        self.evaluate_population()
    
    def evaluate_population(self):
        """Evaluate all particles and update archive"""
        for particle in self.pop:
            self._evaluate_particle(particle)
            
            # Update personal best
            if particle.dominates_personal_best():
                particle.best_position = particle.position.copy()
                particle.best_obj1 = particle.obj1
                particle.best_obj2 = particle.obj2
            
            # Add to archive
            self.archive.add(particle)
    
    def _evaluate_particle(self, particle):
        """Evaluate a single particle's objectives"""
        fitness, num_equal, num_gates, _ = problem_def.fitness_function(particle.position)
        
        # Objective 1: Maximize correctness (number of correct outputs)
        particle.obj1 = num_equal
        
        # Objective 2: Minimize complexity (number of gates)
        particle.obj2 = num_gates
        
        particle.num_equal_tt = num_equal
        particle.num_gates = num_gates
    
    def run_iterations(self, num_iterations, current_gen=0, max_gen=1000):
        """Run MOPSO for specified iterations"""
        for it in range(num_iterations):
            gen = current_gen + it
            
            # Adaptive learning factors (triangular function from MOIPSO)
            self.c1 = self.c1_start * abs(np.cos(2 * np.random.rand() * np.pi))
            self.c2 = self.c2_start * abs(np.sin(2 * np.random.rand() * np.pi))
            
            for particle in self.pop:
                # Select leader from archive
                leader = self.archive.select_leader()
                if leader is None:
                    continue
                
                # Update velocity
                r1 = np.random.random(self.varSize)
                r2 = np.random.random(self.varSize)
                
                particle.velocity = (self.w * particle.velocity +
                                   self.c1 * r1 * (particle.best_position - particle.position) +
                                   self.c2 * r2 * (leader.position - particle.position))
                
                particle.velocity = np.clip(particle.velocity, self.velMin, self.velMax)
                
                # Update position
                particle.position = particle.position + particle.velocity
                particle.position = self._clamp_position(particle.position)
                
                # Adaptive Gaussian mutation (from MOIPSO)
                if np.random.rand() < self.mutation_rate:
                    particle.position = self._mutate(particle.position, gen, max_gen)
                
                # Evaluate
                self._evaluate_particle(particle)
                
                # Update personal best
                if particle.dominates_personal_best():
                    particle.best_position = particle.position.copy()
                    particle.best_obj1 = particle.obj1
                    particle.best_obj2 = particle.obj2
                
                # Add to archive
                self.archive.add(particle)
            
            # Dampen inertia
            self.w = self.w * self.w_damp
    
    def _clamp_position(self, position):
        """Clamp position to valid ranges using robust clamping"""
        return clamp_particle_position(position)
    
    def _mutate(self, position, gen, max_gen):
        """Adaptive Gaussian mutation"""
        num_inputs = problem_def.truth_table.num_inputs
        num_rows = problem_def.num_rows
        
        # Adaptive sigma
        sigma_factor = 0.1 * (1 - gen / max_gen)
        
        mutated = position.copy()
        idx = 0
        
        # Mutate circuit matrix
        for i in range(num_rows):
            input_high = num_inputs + i + 1
            
            # Input 1
            if np.random.rand() < 0.3:
                dx = int(sigma_factor * np.random.randn() * input_high)
                mutated[idx] = max(1, min(input_high, mutated[idx] + dx))
            idx += 1
            
            # Gate type
            if np.random.rand() < 0.3:
                mutated[idx] = np.random.randint(0, len(problem_def.GateType))
            idx += 1
            
            # Input 2
            if np.random.rand() < 0.3:
                dx = int(sigma_factor * np.random.randn() * input_high)
                mutated[idx] = max(1, min(input_high, mutated[idx] + dx))
            idx += 1
        
        # Mutate outputs
        output_high = num_inputs + num_rows + 1
        for _ in range(problem_def.truth_table.num_outputs):
            if np.random.rand() < 0.3:
                dx = int(sigma_factor * np.random.randn() * output_high)
                mutated[idx] = max(1, min(output_high, mutated[idx] + dx))
            idx += 1
        
        return mutated
    
    def get_best_solution(self):
        """Get best solution prioritizing correctness first, then simplicity"""
        if len(self.archive.archive) == 0:
            return None
        
        # Find all solutions with maximum correctness
        max_correctness = max(p.obj1 for p in self.archive.archive)
        best_correct = [p for p in self.archive.archive if p.obj1 == max_correctness]
        
        # Among those, find the simplest (minimum gates)
        return min(best_correct, key=lambda p: p.obj2)

# Helper method for MOParticle
def dominates_personal_best(self):
    """Check if current position dominates personal best"""
    better_in_one = (self.obj1 > self.best_obj1) or (self.obj2 < self.best_obj2)
    not_worse_in_any = (self.obj1 >= self.best_obj1) and (self.obj2 <= self.best_obj2)
    return better_in_one and not_worse_in_any

MOParticle.dominates_personal_best = dominates_personal_best