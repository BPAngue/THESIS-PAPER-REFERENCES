import numpy as np
import math
import copy
import problem_def_patched as problem_def
from pso_solver import Particle, clamp_particle_position 


class HybridPSOGWOSolver:
    def __init__(self, nPop, varSize, pso_params):
        self.nPop = nPop
        self.varSize = varSize
        # Extract PSO parameters
        self.w = pso_params['w']
        self.c1 = pso_params['c1']  # Personal Best weight
        self.c2 = pso_params['c2']  # GWO Guide weight (replaces social weight)
        self.velMax = pso_params['velMax']
        self.velMin = pso_params['velMin']
        self.w_damp = pso_params['w_damp']

        # Initialize Population
        self.pop = [Particle(varSize) for _ in range(nPop)]
        
        # GWO Leaders (Alpha, Beta, Delta)
        self.alpha = Particle(varSize)
        self.beta = Particle(varSize)
        self.delta = Particle(varSize)
        self.alpha.fitness = -math.inf
        self.beta.fitness = -math.inf
        self.delta.fitness = -math.inf

        self.evaluate_population()

    def evaluate_population(self):
        """Evaluates fitness and sorts the population to find the GWO leaders."""
        for particle in self.pop:
            (particle.fitness, 
             particle.num_equal_tt, 
             particle.num_gates, 
             particle.num_no_gates) = problem_def.fitness_function(particle.position)

            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()

        # --- GWO LOGIC: Sort to find Leaders ---
        sorted_pop = sorted(self.pop, key=lambda x: x.fitness, reverse=True)

        if sorted_pop[0].fitness > self.alpha.fitness:
            self.alpha = copy.deepcopy(sorted_pop[0])
        if sorted_pop[1].fitness > self.beta.fitness:
            self.beta = copy.deepcopy(sorted_pop[1])
        if sorted_pop[2].fitness > self.delta.fitness:
            self.delta = copy.deepcopy(sorted_pop[2])

    def run_iterations(self, num_iterations):
        """Runs the Hybrid optimization."""
        for it in range(num_iterations):
            for particle in self.pop:
                
                r1 = np.random.random(size=self.varSize)
                r2 = np.random.random(size=self.varSize)
                
                # --- CORRECTED GWO LOGIC (Stochastic Crossover) ---
                # CRITICAL FIX: Do NOT average integers. Pick genes from leaders.
                source_mask = np.random.randint(0, 3, size=self.varSize)
                guide_position = np.where(source_mask == 0, self.alpha.position,
                                 np.where(source_mask == 1, self.beta.position,
                                                         self.delta.position))
                
                # Standard PSO Velocity Equation with GWO Guide
                new_velocity = (self.w * particle.velocity) + \
                               (self.c1 * r1 * (particle.best_position - particle.position)) + \
                               (self.c2 * r2 * (guide_position - particle.position))

                particle.velocity = np.clip(new_velocity, self.velMin, self.velMax)
                particle.position = particle.position + particle.velocity
                particle.position = clamp_particle_position(particle.position)

            self.evaluate_population()
            self.w = self.w * self.w_damp
            
    def get_gbest(self):
        return copy.deepcopy(self.alpha)

    def force_gbest_replacement(self, new_gbest_particle):
        if new_gbest_particle.fitness > self.alpha.fitness:
            self.alpha = copy.deepcopy(new_gbest_particle)
        elif new_gbest_particle.fitness > self.beta.fitness:
            self.beta = copy.deepcopy(new_gbest_particle)
        elif new_gbest_particle.fitness > self.delta.fitness:
            self.delta = copy.deepcopy(new_gbest_particle)