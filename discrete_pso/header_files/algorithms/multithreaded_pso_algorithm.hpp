#pragma once
#include "../circuit_types.hpp"
#include "../generator.hpp"
#include "../evaluator.hpp"
#include "../reader.hpp"
#include "pso_algorithm.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <barrier>
#include <iostream>
#include <memory>
#include <atomic>

namespace LogicCircuitSynthesis {

    // Shared state between swarms during single-evolutionary phase
    struct GlobalBestState {
        std::vector<int> gbest_pos;           // Global best position
        std::vector<int> gbest_genotype_data; // Flattened genotype data
        int gbest_fitness = 99999999;
        std::mutex mutex;
    };

    // Swarm context for individual thread
    struct SwarmContext {
        std::vector<Particle> particles;
        std::vector<int> swarm_best_pos;     // Best position in this swarm
        int swarm_best_fitness = 99999999;
        int swarm_id;
        DiscretePSO* pso_engine;
        Evaluator* evaluator;
        const PluReader* benchmark;
        GlobalBestState* global_state;
        std::barrier<>* multi_evo_barrier;   // Barrier for multi-evolutionary phase sync
        std::barrier<>* phase_transition_barrier; // Barrier for transitioning to single-evo phase
        int iterations_per_phase;
        bool use_global_best = false;        // Flag to use global best in velocity update
    };

    class MultithreadedDiscretePSO {
    private:
        int num_swarms;
        int swarm_size;
        int max_iterations;
        int iterations_per_phase;
        double phi1;
        double phi2;
        double vMax;
        double mutation_rate;

        int num_inputs;
        int num_outputs;
        int num_nodes;
        int levels_back;
        int num_functions;
        int total_dimensions;

        std::mt19937 rng;

    public:
        MultithreadedDiscretePSO(int n_swarms, int s_size, int iters, int iters_per_phase,
                                  double p_phi1, double p_phi2, double v_max, double mut_rate,
                                  int in, int out, int nodes, int l_back, int funcs)
                : num_swarms(n_swarms), swarm_size(s_size), max_iterations(iters),
                  iterations_per_phase(iters_per_phase), phi1(p_phi1), phi2(p_phi2),
                  vMax(v_max), mutation_rate(mut_rate), num_inputs(in), num_outputs(out),
                  num_nodes(nodes), levels_back(l_back), num_functions(funcs) {

            total_dimensions = (num_nodes * 3) + num_outputs;

            std::random_device rd;
            rng.seed(rd());
        }

        // Worker function for individual swarm thread
        void swarm_worker(SwarmContext& context) {
            int total_completed_iterations = 0;

            std::cout << "[Swarm " << context.swarm_id << "] Starting multi-evolutionary phase...\n";

            // ============================================================================
            // MULTI-EVOLUTIONARY PHASE: K independent swarms evolve in parallel
            // ============================================================================
            
            // Step 1-2: Initialize and evaluate this swarm's particles
            std::cout << "[Swarm " << context.swarm_id << "] Step 1-2: Initializing and evaluating particles...\n";
            for (int i = 0; i < (int)context.particles.size(); i++) {
                context.particles[i].current_fitness = context.evaluator->calculate_fitness(
                    context.particles[i].genotype, *context.benchmark);
                
                if (context.particles[i].current_fitness < context.particles[i].pbest_fitness) {
                    context.particles[i].pbest_fitness = context.particles[i].current_fitness;
                    context.particles[i].pbest_pos = context.particles[i].current_pos;
                }
            }

            // Step 3: Determine particle-best and swarm-best for this swarm
            update_swarm_best(context);
            std::cout << "[Swarm " << context.swarm_id << "] Step 3: Swarm best fitness = " 
                      << context.swarm_best_fitness << "\n";

            // Multi-evolutionary iterations (Steps 4-5)
            for (int phase_iter = 0; phase_iter < iterations_per_phase; phase_iter++) {
                // Step 4-5: Update velocity and position, apply mutation
                evolve_swarm(context, false); // false = don't use global best yet
                total_completed_iterations++;

                if (phase_iter % 100 == 0) {
                    std::cout << "[Swarm " << context.swarm_id << "] Multi-evo phase iteration " 
                              << phase_iter << " | Best: " << context.swarm_best_fitness << "\n";
                }
            }

            // Synchronization point: Wait for all swarms to complete multi-evolutionary phase
            std::cout << "[Swarm " << context.swarm_id << "] Reaching barrier: multi-evolutionary sync...\n";
            context.multi_evo_barrier->arrive_and_wait();

            // ============================================================================
            // SINGLE-EVOLUTIONARY PHASE: Swarms exchange information
            // ============================================================================

            std::cout << "[Swarm " << context.swarm_id << "] Starting single-evolutionary phase...\n";

            // Step 6: Global-best is computed by main thread after barrier
            // (All swarm-best values are now available)

            // Step 7: Particles in all swarms use global-best reference
            context.use_global_best = true;

            // Single-evolutionary iterations (same steps but using global-best)
            for (int phase_iter = 0; phase_iter < iterations_per_phase; phase_iter++) {
                // Update with global best reference
                evolve_swarm(context, true); // true = use global best
                total_completed_iterations++;

                if (phase_iter % 100 == 0) {
                    std::cout << "[Swarm " << context.swarm_id << "] Single-evo phase iteration " 
                              << phase_iter << " | Local Best: " << context.swarm_best_fitness << "\n";
                }
            }

            // Synchronization point: Wait for single-evolutionary phase to complete
            std::cout << "[Swarm " << context.swarm_id << "] Reaching barrier: phase transition sync...\n";
            context.phase_transition_barrier->arrive_and_wait();

            std::cout << "[Swarm " << context.swarm_id << "] Completed. Total iterations: " 
                      << total_completed_iterations << "\n";
        }

        // Evolve one iteration for a swarm
        void evolve_swarm(SwarmContext& context, bool use_global_best) {
            int n = context.particles.size();

            // A. Compute fitness and update Pbest
            for (int i = 0; i < n; i++) {
                context.particles[i].current_fitness = context.evaluator->calculate_fitness(
                    context.particles[i].genotype, *context.benchmark);
                
                if (context.particles[i].current_fitness < context.particles[i].pbest_fitness) {
                    context.particles[i].pbest_fitness = context.particles[i].current_fitness;
                    context.particles[i].pbest_pos = context.particles[i].current_pos;
                }
            }

            // B. Update swarm best
            update_swarm_best(context);

            // C & D: Velocity update and position update
            if (use_global_best) {
                update_particles_with_global_best(context);
            } else {
                update_particles_with_neighborhood(context);
            }

            // E. Apply mutation
            for (int i = 0; i < n; i++) {
                context.pso_engine->apply_discrete_mutation(context.particles[i]);
            }
        }

        // Update swarm best for a given context
        void update_swarm_best(SwarmContext& context) {
            context.swarm_best_fitness = 99999999;
            int best_idx = 0;

            for (int i = 0; i < (int)context.particles.size(); i++) {
                if (context.particles[i].pbest_fitness < context.swarm_best_fitness) {
                    context.swarm_best_fitness = context.particles[i].pbest_fitness;
                    context.swarm_best_pos = context.particles[i].pbest_pos;
                    best_idx = i;
                }
            }
        }

        // Neighborhood-based velocity update (multi-evolutionary phase)
        void update_particles_with_neighborhood(SwarmContext& context) {
            int n = context.particles.size();

            // Compute neighborhood best for each particle
            std::vector<std::vector<int>> Nbest(n);
            for (int i = 0; i < n; i++) {
                int left = (i - 1 + n) % n;
                int right = (i + 1) % n;

                int best_idx = i;
                if (context.particles[left].pbest_fitness < context.particles[best_idx].pbest_fitness)
                    best_idx = left;
                if (context.particles[right].pbest_fitness < context.particles[best_idx].pbest_fitness)
                    best_idx = right;

                Nbest[i] = context.particles[best_idx].pbest_pos;
            }

            // Update velocity and position
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < total_dimensions; d++) {
                    double r1 = context.pso_engine->get_random_double();
                    double r2 = context.pso_engine->get_random_double();

                    context.particles[i].velocity[d] += 
                        (phi1 * r1 * (context.particles[i].pbest_pos[d] - context.particles[i].current_pos[d])) +
                        (phi2 * r2 * (Nbest[i][d] - context.particles[i].current_pos[d]));
                    
                    context.particles[i].velocity[d] = 
                        std::clamp(context.particles[i].velocity[d], -vMax, vMax);

                    double prob = context.pso_engine->sigmoid(context.particles[i].velocity[d]);

                    if (context.pso_engine->flip(prob)) {
                        context.particles[i].current_pos[d] = Nbest[i][d];
                    } else if (context.pso_engine->flip(1.0 - prob)) {
                        context.particles[i].current_pos[d] = context.particles[i].pbest_pos[d];
                    }
                }
                context.pso_engine->unflatten_genotype(context.particles[i]);
            }
        }

        // Global-best based velocity update (single-evolutionary phase)
        void update_particles_with_global_best(SwarmContext& context) {
            int n = context.particles.size();

            // Read global best (thread-safe with lock)
            std::vector<int> gbest_pos;
            {
                std::lock_guard<std::mutex> lock(context.global_state->mutex);
                gbest_pos = context.global_state->gbest_pos;
            }

            // Update velocity and position using global best
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < total_dimensions; d++) {
                    double r1 = context.pso_engine->get_random_double();
                    double r2 = context.pso_engine->get_random_double();
                    double r3 = context.pso_engine->get_random_double();

                    // Equation (11) from the paper: includes global best
                    context.particles[i].velocity[d] += 
                        (phi1 * r1 * (context.particles[i].pbest_pos[d] - context.particles[i].current_pos[d])) +
                        (phi2 * r2 * (gbest_pos[d] - context.particles[i].current_pos[d])) +
                        (0.5 * r3 * (gbest_pos[d] - context.particles[i].current_pos[d]));
                    
                    context.particles[i].velocity[d] = 
                        std::clamp(context.particles[i].velocity[d], -vMax, vMax);

                    double prob = context.pso_engine->sigmoid(context.particles[i].velocity[d]);

                    if (context.pso_engine->flip(prob)) {
                        context.particles[i].current_pos[d] = gbest_pos[d];
                    } else if (context.pso_engine->flip(1.0 - prob)) {
                        context.particles[i].current_pos[d] = context.particles[i].pbest_pos[d];
                    }
                }
                context.pso_engine->unflatten_genotype(context.particles[i]);
            }
        }

        // Main optimization entry point
        Genotype optimize(Generator& generator, Evaluator& evaluator, 
                         const PluReader& benchmark, int num_threads = 0) {
            if (num_threads == 0) {
                num_threads = std::thread::hardware_concurrency();
            }
            num_swarms = num_threads;

            std::cout << "============================================================\n";
            std::cout << "Multithreaded PSO Initialization\n";
            std::cout << "Number of Swarms (Threads): " << num_swarms << "\n";
            std::cout << "Swarm Size: " << swarm_size << "\n";
            std::cout << "Iterations per Phase: " << iterations_per_phase << "\n";
            std::cout << "Total Iterations: " << max_iterations << "\n";
            std::cout << "============================================================\n";

            // Global best state
            GlobalBestState global_state;
            global_state.gbest_pos.resize(total_dimensions);
            global_state.gbest_genotype_data.resize(total_dimensions);

            // Create PSO engine and swarm contexts
            DiscretePSO pso_engine(swarm_size, max_iterations, phi1, phi2, vMax, mutation_rate,
                                   num_inputs, num_outputs, num_nodes, levels_back, num_functions);

            std::vector<SwarmContext> swarm_contexts;
            std::barrier<> multi_evo_barrier(num_swarms);
            std::barrier<> phase_transition_barrier(num_swarms);

            Genotype global_best_genotype;
            int global_best_fitness = 99999999;

            // Initialize all swarms
            std::cout << "\nInitializing " << num_swarms << " independent swarms...\n";
            for (int s = 0; s < num_swarms; s++) {
                SwarmContext ctx;
                ctx.swarm_id = s;
                ctx.particles.resize(swarm_size);
                ctx.swarm_best_pos.resize(total_dimensions);
                ctx.pso_engine = &pso_engine;
                ctx.evaluator = &evaluator;
                ctx.benchmark = &benchmark;
                ctx.global_state = &global_state;
                ctx.multi_evo_barrier = &multi_evo_barrier;
                ctx.phase_transition_barrier = &phase_transition_barrier;
                ctx.iterations_per_phase = iterations_per_phase;

                // Step 1: Randomly generate particles for this swarm
                for (int i = 0; i < swarm_size; i++) {
                    ctx.particles[i].genotype = generator.generate_random_individual();
                    ctx.particles[i].current_pos = pso_engine.flatten_genotype(ctx.particles[i].genotype);
                    ctx.particles[i].pbest_pos = ctx.particles[i].current_pos;
                    ctx.particles[i].velocity.resize(total_dimensions, 0.0);
                    ctx.particles[i].pbest_fitness = 99999999;
                }

                swarm_contexts.push_back(ctx);
            }

            // Main iteration loop
            for (int main_iter = 0; main_iter < max_iterations; main_iter += (2 * iterations_per_phase)) {
                std::cout << "\n=== Main Iteration " << (main_iter / (2 * iterations_per_phase) + 1) << " ===\n";

                // Create and launch threads for all swarms
                std::vector<std::thread> threads;
                for (int s = 0; s < num_swarms; s++) {
                    threads.emplace_back(&MultithreadedDiscretePSO::swarm_worker, this, 
                                        std::ref(swarm_contexts[s]));
                }

                // Wait for all threads to complete multi-evolutionary phase and sync
                for (auto& t : threads) {
                    t.join();
                }

                // Main thread computes global best (Step 6)
                std::cout << "\n[Main] Step 6: Computing global-best from all swarms...\n";
                {
                    std::lock_guard<std::mutex> lock(global_state.mutex);
                    global_state.gbest_fitness = 99999999;

                    for (int s = 0; s < num_swarms; s++) {
                        if (swarm_contexts[s].swarm_best_fitness < global_state.gbest_fitness) {
                            global_state.gbest_fitness = swarm_contexts[s].swarm_best_fitness;
                            global_state.gbest_pos = swarm_contexts[s].swarm_best_pos;
                        }
                    }
                }

                std::cout << "[Main] Global Best Fitness: " << global_state.gbest_fitness << "\n";

                // Update global best for early termination check
                if (global_state.gbest_fitness < global_best_fitness) {
                    global_best_fitness = global_state.gbest_fitness;
                    // Reconstruct genotype from position
                    // (Implementation depends on how genotype is structured)
                }

                if (global_best_fitness == 0) {
                    std::cout << "\n*** SUCCESS: Circuit synthesized at main iteration " 
                              << (main_iter / (2 * iterations_per_phase) + 1) << " ***\n";
                    break;
                }
            }

            std::cout << "\n=== Optimization Complete ===\n";
            std::cout << "Final Global Best Fitness: " << global_best_fitness << "\n";

            return global_best_genotype;
        }
    };
}
