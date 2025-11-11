import problem2
import numpy as np
import math
import copy
import sys
from datetime import datetime

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_filename = f"pso_log_{timestamp}.txt"

# # Custom print function that writes to both console and file
# class Logger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w", encoding="utf-8")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         # Needed for real-time console output
#         self.terminal.flush()
#         self.log.flush()

# # Redirect all prints
# sys.stdout = Logger(log_filename)

# print(f"[LOGGING ENABLED] Output is being saved to {log_filename}\n")

# Problem Definition and PSO parameters
def costFunction(position, num_equal_tt, num_gates, num_no_gates, prev_cost, d_cost):
    return problem2.problem(position, num_equal_tt, num_gates, num_no_gates, prev_cost, d_cost)

num_rows = problem2.num_rows
num_cols = problem2.num_cols

nVar = num_rows * num_cols * 3    # number of decision variables
varSize = nVar                    # size of decision variable matrix
varMin = 0                        # lower bound of variables
varMax = 20                       # upper bound of variables

maxIt = 100    # maximum number of iterations
nPop = 50      # population size

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

# Initialize a particle
class Particle:
    pass

empty_particle = Particle() # create an instance of class particle

# dynamically attach attributes to the instance of class particle
empty_particle.position = np.ndarray   # create an empty n-dimensional array for the candidate solution
empty_particle.cost = None             # assign empty cost (how good or bad the position is)
empty_particle.velocity = np.ndarray   # velocity is an n-dimensional array since each particle's position is an n element vector. The velocity contains a separate speed component for every coordinate direction.
empty_particle.best_position = np.ndarray
empty_particle.best_cost = None
empty_particle.num_equal_tt = None     # assign empty (sets the number of equal outputs from the reference truth table)
empty_particle.num_gates = None        # assign empty (sets the number of gates used)
empty_particle.num_no_gates = None     # assign empty (sets the number of wires in solution)
empty_particle.prev_cost = None        # assign empty (sets the previous cost) (for dynamic fitness)
empty_particle.d_cost = None           # assign empty (sets the fractional derivative term) (for dynamic fitness)

pop = []    # list containing the population of particles

# create a population of empty particles
for i in range(nPop):
    pop.append(copy.deepcopy(empty_particle))

global_best_particle = copy.deepcopy(empty_particle) # create global best particle variable templated from the class particle
global_best_particle.cost = -math.inf                 # set the global best particles cost to infinity so that any real particle encountered later will have a lower cost and replace it.

# Initialize the particles
for particle in pop:
    particle.num_equal_tt = 0
    particle.num_gates = 0
    particle.num_no_gates = 0
    particle.prev_cost = 0.0
    particle.d_cost = 0.0

    particle.position = np.random.randint(varMin, varMax, size=varSize)
    particle.velocity = np.random.uniform(velMin, velMax, size=varSize)

    particle.cost, particle.prev_cost, particle.d_cost, particle.num_equal_tt, particle.num_gates, particle.num_no_gates = costFunction(
        particle.position, 
        particle.num_equal_tt, 
        particle.num_gates, 
        particle.num_no_gates, 
        particle.prev_cost, 
        particle.d_cost
    )

    particle.best_position = particle.position.copy()
    particle.best_cost = particle.cost 

    if particle.best_cost > global_best_particle.cost:
        global_best_particle = copy.copy(particle)

# PSO Main Loop
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
        
        # update cost, prev_cost, d_cost, num_equal_tt, num_gates, num_no_gates
        particle.cost, particle.prev_cost, particle.d_cost, particle.num_equal_tt, particle.num_gates, particle.num_no_gates = costFunction(
            particle.position, 
            particle.num_equal_tt, 
            particle.num_gates, 
            particle.num_no_gates, 
            particle.prev_cost, 
            particle.d_cost
        ) # update the cost of the new particle position

        # update personal best
        if particle.cost > particle.best_cost:
            particle.best_cost = particle.cost
            particle.best_position = particle.position.copy()

        # update global best
        if particle.cost > global_best_particle.cost:
            global_best_particle = copy.copy(particle)

final_formulas = problem2.get_circuit_formula(global_best_particle.position)
print(f"Cost: {global_best_particle.cost}")
print(f"Correct outputs: {global_best_particle.num_equal_tt}")
print(f"Gates used: {global_best_particle.num_gates}")
print(f"Wires/Simplified: {global_best_particle.num_no_gates}")