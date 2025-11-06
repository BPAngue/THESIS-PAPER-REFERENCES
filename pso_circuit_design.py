import problem2
import numpy as np
import math
import copy

# Problem Definition and PSO parameters
def costFunction(x):
    return problem2.problem(x)

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

pop = []    # list containing the population of particles

# create a population of empty particles
for i in range(nPop):
    pop.append(copy.deepcopy(empty_particle))

global_best_particle = copy.deepcopy(empty_particle) # create global best particle variable templated from the class particle
global_best_particle.cost = math.inf                 # set the global best particles cost to infinity so that any real particle encountered later will have a lower cost and replace it.

# Initialize the particles
for particle in pop:
    particle.position = np.random.randint(varMin, varMax, size=varSize)
    particle.velocity = np.random.uniform(velMin, velMax, size=varSize)
    particle.cost = costFunction(particle.position)
    particle.best_position = particle.position.copy()
    particle.best_cost = particle.cost 