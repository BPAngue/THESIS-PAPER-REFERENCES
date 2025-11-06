import numpy as np
import math
import copy
from problem import problem
import matplotlib.pyplot as plt
import sys
import datetime
import os

# create directory for results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# unique timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_filename = os.path.join(results_dir, f"{timestamp}_pso_debug_log.txt")
fig_linear_filename = os.path.join(results_dir, f"{timestamp}_pso_cost_linear.png")
fig_semilog_filename = os.path.join(results_dir, f"{timestamp}_pso_cost_semilog.png")
fig_swarm_filename = os.path.join(results_dir, f"{timestamp}_pso_swarm.png")


# for exporting printed debug logs into text file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            try:
                f.write(text)
                f.flush()
            except ValueError:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass

with open(log_filename, "w") as log:
    sys_stdout_backup = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log)
    try:
        # Problem definition
        def costFunction(x):
            return problem(x)

        nVar = 2            # number of decision variables
        varSize = nVar      # size of decision variable matrix
        varMin = -20        # lower bound of variables
        varMax = 20         # upper bound of variables

        # PSO Parameters
        maxIt = 100         # maximum number of iterations
        nPop = 50           # population size (swarm size)

        constriction_coefficient = True

        if constriction_coefficient == False:
            w = 1                  # inertia weight
            w_damp = 0.99          # inertia weight damping ratio
            c1 = 2                 # personal learning coefficient
            c2 = 2                 # global learning coefficient
        else:
            # with constriction coefficient
            phi1 = 2.05
            phi2 = 2.05
            phi = phi1 + phi2
            chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi)))
            w = chi
            w_damp = 1
            c1 = chi * phi1
            c2 = chi * phi2

        # Velocity Limits
        velMax = 0.1 * (varMax-varMin)
        velMin = -velMax

        # Initialization
        class Particle:
            pass

        empty_particle = Particle() # create an instance of class particle

        # dynamically attach attributes to the instance of class particle
        empty_particle.position = np.array([]) # create an empty array for the candidate solution
        empty_particle.cost = None             # assign empty cost (how good or bad the position is)
        empty_particle.velocity = np.array([]) # velocity is an array since each particle's position is an n element vector. The velocity contains a separate speed component for every coordinate direction.
        empty_particle.best_position = np.array([])
        empty_particle.best_cost = None 

        pop = np.array([])    # the population of particles

        # create a population of empty particles
        for i in range(nPop):
            pop = np.append(pop, copy.deepcopy(empty_particle))

        global_best_particle = copy.deepcopy(empty_particle) # create global best particle variable templated from the class particle
        global_best_particle.cost = math.inf             # set the global best particles cost to infinity so that any real particle encountered later will have a lower cost and replace it.

        # initialize the particles
        print("-"*100)
        print("PARTICLE SWARM OPTIMIZATION PARTICLE INITIALIZATION")
        print("-"*100)
        i = 1 # for debugging
        for particle in pop:
            particle.position = np.random.uniform(low=varMin, high=varMax, size=varSize) # tells NumPy to generate n=size random numbers uniformly distributed between varMin and varMax for each particle
            particle.velocity = np.zeros(shape=varSize)                                  # sets the initial velocity vector to zero for each particle
            particle.cost = costFunction(particle.position)                              # this is the cost function similar to the evaluation function/fitness function
            particle.best_position = particle.position.copy()                            # at initialization each particle only knows its starting position, so its personal best must start here
            particle.best_cost = particle.cost                                           # at initialization each particle only knows its starting cost, so its personal best cost must start here
            
            # for debugging
            print(f"Particle {i}")
            print("   Position:", particle.position)
            print("   Velocity:", particle.velocity)
            print("   Cost:", particle.cost)
            print("   Best Position:", particle.best_position)
            print("   Best Cost:", particle.best_cost)
            print("-"*50)
            if particle.best_cost < global_best_particle.cost:
                global_best_particle = copy.copy(particle)                           # for every randomly initialized particle update the global_best_particle whenever the cost of the particle is less than the current cost of the global best particle

                # for debugging
                print(f"Updated the global best particle at {i}")
                print("   Position:", global_best_particle.position)
                print("   Velocity:", global_best_particle.velocity)
                print("   Cost:", global_best_particle.cost)
                print("   Best Position:", global_best_particle.best_position)
                print("   Best Cost:", global_best_particle.best_cost)
                print("-"*50)
            
            i+=1

        # for debugging
        print("Final global best particle after initialization")
        print("   Position:", global_best_particle.position)
        print("   Velocity:", global_best_particle.velocity)
        print("   Cost:", global_best_particle.cost)
        print("   Best Position:", global_best_particle.best_position)
        print("   Best Cost:", global_best_particle.best_cost)
        print("-"*50)

        best_cost_list = []     # this list is used for plotting the best costs at the end (does not affect the algorithm)
        best_cost_list.append(global_best_particle.cost)

        # live swarm view (first two dimensions) (does not affect the algorithm)
        plt.ion()  # turn on interactive drawing
        fig_swarm, ax_swarm = plt.subplots()
        ax_swarm.set_title('PSO swarm (dims 1 & 2)')
        ax_swarm.set_xlabel('x1')
        ax_swarm.set_ylabel('x2')
        ax_swarm.set_xlim(varMin, varMax)
        ax_swarm.set_ylim(varMin, varMax)
        ax_swarm.set_aspect('equal', adjustable='box')

        # scatter for particles + star for global best
        scat = ax_swarm.scatter([], [], s=25)                                          # particles
        gb_star, = ax_swarm.plot([], [], marker='*', markersize=12, linestyle='None')  # global best
        hud = ax_swarm.text(0.02, 0.98, '', transform=ax_swarm.transAxes, va='top')    # iter + best cost (text shown in the figure)

        # PSO Main Loop
        print("-"*100)
        print("PARTICLE SWARM OPTIMIZATION ALGORITHM")
        print("-"*100)
        for itr in range(maxIt):
            for i, particle in enumerate(pop): # loop through the swarm (pop) while also keeping track of the index of each particle.
                # for debugging
                print(f"Iteration {itr+1}")
                print()

                # update velocity
                r1 = np.random.random(size=varSize) # creates an array with size=varSize of random numbers between 0 and 1
                r2 = np.random.random(size=varSize)
                particle.velocity = (w * particle.velocity) + (c1 * r1 * (particle.best_position - particle.position)) + (c2 * r2 * (global_best_particle.position - particle.position))

                # apply velocity limits
                particle.velocity = np.array([max(p, velMin) for p in particle.velocity]) # makes sure every component of the particle's velocity stays within [velMin, velMax]
                particle.velocity = np.array([min(p, velMax) for p in particle.velocity])

                # update position
                particle.position = particle.position + particle.velocity

                # apply clamping to limit position within boundaries (try -> to be removed if not working)
                particle.position = np.clip(particle.position, varMin, varMax)

                # update cost
                particle.cost = costFunction(particle.position) # update the cost of the new particle position

                # update personal best
                if particle.cost < particle.best_cost:
                    particle.best_cost = particle.cost
                    particle.best_position = particle.position.copy()

                # for debugging
                print(f"Particle {i+1}")
                print("   Position:", particle.position)
                print("   Velocity:", particle.velocity)
                print("   Cost:", particle.cost)
                print("   Best Position:", particle.best_position)
                print("   Best Cost:", particle.best_cost)
                print("-"*50)

                # update global best
                if particle.cost < global_best_particle.cost:
                    global_best_particle = copy.copy(particle)
                    # for debugging
                    print(f"UPDATED THE GLOBAL BEST PARTICLE AT {i+1}")
                    print("   Position:", global_best_particle.position)
                    print("   Velocity:", global_best_particle.velocity)
                    print("   Cost:", global_best_particle.cost)
                    print("   Best Position:", global_best_particle.best_position)
                    print("   Best Cost:", global_best_particle.best_cost)
                    print("-"*50)

            # update live swarm view (does not affect the algorithm)
            positions_2d = np.vstack([p.position[:2] for p in pop])  # take dims 1 & 2
            scat.set_offsets(positions_2d)
            gb_star.set_data(
                [global_best_particle.position[0]],
                [global_best_particle.position[1]]
            )
            hud.set_text(f'iter: {itr+1}/{maxIt}\nbest: {global_best_particle.cost:.6e}')
            plt.pause(0.01)  # this draws one animation frame

            # appends best costs and prints it for every iteration
            best_cost_list.append(global_best_particle.cost)
            print()
            print("-"*100)
            print(f"Iteration {itr+1}: Best Cost = {global_best_particle.cost:.6f}")
            print(f"Best Position = {global_best_particle.position}")
            print("-"*100)
            print()
            w = w * w_damp   # gradually reduces the inertia weight over time — it’s the “cooling schedule” for the swarm’s momentum.

        plt.ioff() # stop interactive updates (animation is over) (does not affect the algorithm)

        # Print the figures
        # ---- Figure: linear cost ----
        fig_cost_lin, ax_lin = plt.subplots()
        ax_lin.plot(best_cost_list)
        ax_lin.set_title('Cost')
        ax_lin.set_xlabel('Iterations')
        ax_lin.set_ylabel('Cost (Linear)')

        # ---- Figure: semilog cost ----
        fig_cost_log, ax_log = plt.subplots()
        ax_log.semilogy(best_cost_list)
        ax_log.set_title('Cost')
        ax_log.set_xlabel('Iterations')
        ax_log.set_ylabel('Cost (Semilog)')

        # Save the figures as png files automatically
        fig_cost_lin.savefig(fig_linear_filename, dpi=300, bbox_inches='tight')
        fig_cost_log.savefig(fig_semilog_filename, dpi=300, bbox_inches='tight')
        fig_swarm.savefig(fig_swarm_filename, dpi=300, bbox_inches='tight')

        plt.show()

        print()
        print("-"*100)
        print("FINAL RESULT OF THE OPTIMIZATION")
        print("   Best Position:", global_best_particle.position)
        print("   Best Cost:", global_best_particle.cost)
        print("-"*100)
        print()

        print("-"*100)
        print('FINISHED')
        print("-"*100)
    finally:
        sys.stdout = sys_stdout_backup