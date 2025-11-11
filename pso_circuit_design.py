import problem2
import numpy as np
import math
import sys
import os
import copy
from datetime import datetime

# --------------------------------------
# Logging Setup
# --------------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(results_dir, f"pso_log_{timestamp}.txt")

# Custom print function that writes to both console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed for real-time console output
        self.terminal.flush()
        self.log.flush()

# Redirect all prints
sys.stdout = Logger(log_filename)
print(f"[LOGGING ENABLED] Output is being saved to {log_filename}\n")
print("[PSO ALGORITHM] Algorithm is currently running")

# --------------------------------------
# Problem and PSO Parameters
# --------------------------------------
num_rows = problem2.num_rows
num_cols = problem2.num_cols
nVar = num_rows * num_cols * 3    # number of decision variables
varSize = nVar                    # size of decision variable matrix
varMin = 0                        # lower bound of variables
varMax = 20                       # upper bound of variables

maxIt = 100    # maximum number of iterations
nPop = 50     # population size
constriction_coefficient = True

if constriction_coefficient == False:
    w = 1          # inertia weight
    w_damp = 0.99  # inertia weight damping ratio
    c1 = 2         # personal learning coefficient
    c2 = 2         # global learning coefficient
else:
    # with constriction coefficient
    phi1 = 2.05
    phi2 = 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi1)))
    w = chi
    w_damp = 1
    c1 = chi * phi1
    c2 = chi * phi2

# velocity limits
velMax = 0.2 * (varMax - varMin)
velMin = -velMax

# --------------------------------------
# Particle Class
# --------------------------------------
class Particle:
    def __init__(self, varsize):
        self.position: np.ndarray = np.random.randint(varMin, varMax, size=varsize)
        self.velocity = np.random.uniform(velMin, velMax, size=varsize)
        self.best_position: np.ndarray = self.position.copy()
        self.fitness = -math.inf
        self.best_fitness = -math.inf
        self.num_equal_tt = 0
        self.num_gates = 0
        self.num_no_gates = 0
        self.prev_static_fitness = 0.0
        self.prev_derivative = 0.0

# --------------------------------------
# Fitness Function Wrapper
# --------------------------------------
def fitnessFunction(position, num_equal_tt, num_gates, num_no_gates, prev_static_fitness, previous_derivative):
    return problem2.problem(position, num_equal_tt, num_gates, num_no_gates, prev_static_fitness, previous_derivative)

# --------------------------------------
# Initialize Population
# --------------------------------------
pop = [Particle(varSize) for _ in range(nPop)]
global_best_particle = Particle(varSize)
global_best_particle.fitness = -math.inf

for particle in pop:
    particle.fitness, particle.prev_static_fitness, particle.prev_derivative, particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
        particle.position, 
        particle.num_equal_tt, 
        particle.num_gates, 
        particle.num_no_gates, 
        particle.prev_static_fitness, 
        particle.prev_derivative
    )

    particle.best_fitness = particle.fitness
    particle.best_position = particle.position.copy()

    if particle.best_fitness > global_best_particle.fitness:
        global_best_particle = copy.deepcopy(particle)

# --------------------------------------
# PSO Main Loop
# --------------------------------------
for itr in range(maxIt):
    for i, particle in enumerate(pop): # loop through the swarm (pop) while also keeping track of the index of each particle.
        # update velocity
        r1 = np.random.random(size=varSize)
        r2 = np.random.random(size=varSize)
        particle.velocity = (w * particle.velocity) + (c1 * r1 * (particle.best_position - particle.position)) + (c2 * r2 * (global_best_particle.position - particle.position))

        # apply velocity limits
        particle.velocity = np.clip(particle.velocity, velMin, velMax)

        # update position
        particle.position = particle.position + particle.velocity

        # apply clamping to limit position within boundaries (try -> to be removed if not working)
        particle.position = np.clip(particle.position, varMin, varMax)
        particle.position = np.round(particle.position).astype(int)
        
        # update fitness, prev_static_fitness, prev_dynamic_fitness, num_equal_tt, num_gates, num_no_gates
        particle.fitness, particle.prev_static_fitness, particle.prev_derivative, particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
            particle.position, 
            particle.num_equal_tt, 
            particle.num_gates, 
            particle.num_no_gates, 
            particle.prev_static_fitness, 
            particle.prev_derivative
        )

        # update personal best
        if particle.fitness > particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = particle.position.copy()

        # update global best
        if particle.fitness > global_best_particle.fitness:
            global_best_particle = copy.deepcopy(particle)

# --------------------------------------
# Print PSO Results
# --------------------------------------
final_formulas = problem2.get_circuit_formula(global_best_particle.position)
input1, input2, gate_type = problem2.decode_matrix(global_best_particle.position)
print("Particle Position:")
print(f"Input1: {input1}")
print(f"Input2: {input2}")
print(f"Gate:   {gate_type}")
print(f"Cost: {global_best_particle.fitness}")
print(f"Correct outputs: {global_best_particle.num_equal_tt}/{problem2.truth_table.total_outputs}")
print(f"Gates used: {global_best_particle.num_gates}")
print(f"Wires/Simplified: {global_best_particle.num_no_gates}")

print("[PSO ALGORITHM] Algorithm finished!")