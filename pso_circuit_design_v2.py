import problem3
import numpy as np
import math
import sys
import os
import copy
from datetime import datetime
import time

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
        self.terminal.flush()
        self.log.flush()

# Redirect all prints
sys.stdout = Logger(log_filename)
print(f"[LOGGING ENABLED] Output is being saved to {log_filename}\n")
print("[PSO ALGORITHM] Algorithm is currently running")

_perf_start = time.perf_counter()

# --------------------------------------
# Problem and PSO Parameters
# --------------------------------------
# Load problem definitions from problem3.py
num_inputs = problem3.truth_table.num_inputs
num_outputs = problem3.truth_table.num_outputs
num_rows = problem3.num_rows  # Number of internal gates

# --- NEW PARTICLE DEFINITION ---
# A particle = (num_rows * 3) genes for circuit_matrix + (num_outputs) genes for output_array
nVar = (num_rows * 3) + num_outputs
varSize = nVar

# Heuristic bounds for velocity calculation.
# A particle's value can be a pointer up to (num_inputs + num_rows)
varMin = 0
varMax = num_inputs + num_rows
# --- END NEW PARTICLE DEFINITION ---

maxIt = 100
nPop = 50
constriction_coefficient = True

if not constriction_coefficient:
    w = 1
    w_damp = 0.99
    c1 = 2
    c2 = 2
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
# --- NEW ENCODING HELPER FUNCTIONS ---
# --------------------------------------

def create_random_particle(varsize):
    """Creates a 1D particle vector with valid, random integers for each gene."""
    particle = np.zeros(varsize, dtype=int)
    idx = 0
    gate_type_high = len(problem3.GateType)

    # 1. Create Circuit Matrix genes (num_rows * 3)
    for i in range(num_rows):
        # Set input_high_exclusive for row 'i'
        # Can point to primary inputs (1...num_inputs) or previous gates (num_inputs+1...num_inputs+i)
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

    # 2. Create Output Array genes (num_outputs)
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
    gate_type_high = len(problem3.GateType) - 1 # Inclusive
    
    # 1. Clamp Circuit Matrix genes
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        
        # Clamp Input 1 (Col 0)
        x[idx] = max(1, min(x[idx], input_high_inclusive))
        idx += 1
        
        # Clamp Gate Type (Col 1)
        x[idx] = max(0, min(x[idx], gate_type_high))
        idx += 1
        
        # Clamp Input 2 (Col 2)
        x[idx] = max(1, min(x[idx], input_high_inclusive))
        idx += 1

    # 2. Clamp Output Array genes
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        x[idx] = max(1, min(x[idx], output_high_inclusive))
        idx += 1
        
    return x

# --------------------------------------
# Particle Class (Updated)
# --------------------------------------
class Particle:
    def __init__(self, varsize):
        # Initialize with a VALID, random particle
        self.position: np.ndarray = create_random_particle(varsize)
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
def fitnessFunction(position, prev_static_fitness, previous_derivative):
    # Pass-through to the problem file
    return problem3.problem(position, prev_static_fitness, previous_derivative)

# --------------------------------------
# Initialize Population
# --------------------------------------
pop = [Particle(varSize) for _ in range(nPop)]
global_best_particle = Particle(varSize)
global_best_particle.fitness = -math.inf

for particle in pop:
    # Note: num_equal_tt, num_gates, etc. are returned by the fitness function
    particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
    particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
        particle.position, 
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
print(f"Starting PSO loop for {maxIt} iterations...")
for itr in range(maxIt):
    for i, particle in enumerate(pop):
        # update velocity
        r1 = np.random.random(size=varSize)
        r2 = np.random.random(size=varSize)
        particle.velocity = (w * particle.velocity) + \
                            (c1 * r1 * (particle.best_position - particle.position)) + \
                            (c2 * r2 * (global_best_particle.position - particle.position))

        # apply velocity limits
        particle.velocity = np.clip(particle.velocity, velMin, velMax)

        # update position
        particle.position = particle.position + particle.velocity

        # --- APPLY NEW CLAMPING ---
        # This is the most important change in the loop
        particle.position = clamp_particle_position(particle.position)
        # --- END NEW CLAMPING ---
        
        # update fitness
        particle.fitness, particle.prev_static_fitness, particle.prev_derivative, \
        particle.num_equal_tt, particle.num_gates, particle.num_no_gates = fitnessFunction(
            particle.position, 
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
    
    if (itr + 1) % 10 == 0:
        print(f"Iteration {itr + 1}/{maxIt}, Best Fitness: {global_best_particle.best_fitness:.2f}")

# --------------------------------------
# Print PSO Results (Updated)
# --------------------------------------
print("\n... PSO loop finished ...\n")
# Get formulas and decoded matrices for the final report
final_formulas = problem3.get_circuit_formula(global_best_particle.position)
circuit_matrix, output_array = problem3.decode_particle(global_best_particle.position)

print("--- FINAL BEST PARTICLE ---")
print("Circuit Matrix ([In1, Gate, In2]):")
print(circuit_matrix)
print("\nOutput Array (Pointers):")
print(output_array)
print("\n--- STATS ---")
print(f"Cost (Fitness): {global_best_particle.fitness:.4f}")
print(f"Correct outputs: {global_best_particle.num_equal_tt}/{problem3.truth_table.total_outputs}")
print(f"Active Gates Used: {global_best_particle.num_gates}")
print(f"Unused Gates (Simplicity): {global_best_particle.num_no_gates}")

print("\n[PSO ALGORITHM] Algorithm finished!")

_perf_end = time.perf_counter()
elapsed_seconds = _perf_end - _perf_start
print(f"[TIMER] Elapsed seconds: {elapsed_seconds:.6f}")