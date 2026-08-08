#pragma once
#include "../circuit_types.hpp"
#include "../generator.hpp"
#include "../evaluator.hpp"
#include "../reader.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace LogicCircuitSynthesis {

    // Particle structure adapted for Discrete PSO
    struct Particle {
        Genotype genotype;                  // The actual valid discrete CGP circuit
        std::vector<int> current_pos;       // Flattened array of current integers (X)
        std::vector<int> pbest_pos;         // Flattened array of best integers (Pbest)
        std::vector<double> velocity;       // Continuous velocity acting as probability weighting
        
        int current_fitness;
        int pbest_fitness;
    };

    class DiscretePSO {
    private:
        int swarm_size;
        int max_iterations;
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
        DiscretePSO(int s_size, int iters, double p_phi1, double p_phi2, double v_max, double mut_rate,
                    int in, int out, int nodes, int l_back, int funcs)
                : swarm_size(s_size), max_iterations(iters), phi1(p_phi1), phi2(p_phi2),
                    vMax(v_max), mutation_rate(mut_rate), num_inputs(in), num_outputs(out),
                    num_nodes(nodes), levels_back(l_back), num_functions(funcs) {

            // Dimensions: 3 integers per node (func, in1, in2) + 1 integer per output
            total_dimensions = (num_nodes * 3) + num_outputs;

            std::random_device rd;
            rng.seed(rd());
        }

        double get_random_double() {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            return dist(rng);
        }

        int get_random_int(int min, int max) {
            std::uniform_int_distribution<int> dist(min, max);
            return dist(rng);
        }

        // Standard Sigmoid function to normalize velocity between 0 and 1
        double sigmoid(double v) {
            return 1.0 / (1.0 + std::exp(-v));
        }

        bool flip(double probability) {
            return get_random_double() < probability;
        }

        // Flattens a Genotype into a 1D array of integers for easy math
        std::vector<int> flatten_genotype(const Genotype& g) {
            std::vector<int> vec(total_dimensions);
            for (int i = 0; i < num_nodes; i++) {
                vec[i * 3 + 0] = g.logic_nodes[i].function_idx;
                vec[i * 3 + 1] = g.logic_nodes[i].input_1_idx;
                vec[i * 3 + 2] = g.logic_nodes[i].input_2_idx;
            }
            for (int o = 0; o < num_outputs; o++) {
                vec[num_nodes * 3 + o] = g.output_genes[o];
            }
            return vec;
        }

        // Translates the flat integer array back into the structural Genotype
        void unflatten_genotype(Particle& p) {
            for (int i = 0; i < num_nodes; i++) {
                p.genotype.logic_nodes[i].function_idx = p.current_pos[i * 3 + 0];
                p.genotype.logic_nodes[i].input_1_idx  = p.current_pos[i * 3 + 1];
                p.genotype.logic_nodes[i].input_2_idx  = p.current_pos[i * 3 + 2];
            }
            for (int o = 0; o < num_outputs; o++) {
                p.genotype.output_genes[o] = p.current_pos[num_nodes * 3 + o];
            }
        }

        // Applies uniform mutation keeping strict CGP connectivity limits
        void apply_discrete_mutation(Particle& p) {
            for (int i = 0; i < num_nodes; i++) {
                int min_conn = (i >= levels_back) ? (num_inputs + i - levels_back) : 0;
                int max_conn = num_inputs + i - 1;

                if (flip(mutation_rate)) p.current_pos[i * 3 + 0] = get_random_int(0, num_functions - 1);
                if (flip(mutation_rate)) p.current_pos[i * 3 + 1] = get_random_int(min_conn, max_conn);
                if (flip(mutation_rate)) p.current_pos[i * 3 + 2] = get_random_int(min_conn, max_conn);
            }

            int max_total_idx = num_inputs + num_nodes - 1;
            for (int o = 0; o < num_outputs; o++) {
                if (flip(mutation_rate)) {
                    p.current_pos[num_nodes * 3 + o] = get_random_int(0, max_total_idx);
                }
            }
            unflatten_genotype(p); // Sync changes to the genotype
        }

        // One iteration of the PSO loop (steps A-D).
        // Operates on an externally-owned swarm passed by reference, so both
        // optimize() and PGPHEA can share the exact same step logic with no duplication.
        void step(std::vector<Particle>& P, Evaluator& evaluator, const PluReader& benchmark) {
            int n = (int)P.size();

            // A. Compute fitness and update Pbest
            for (int i = 0; i < n; i++) {
                P[i].current_fitness = evaluator.calculate_fitness(P[i].genotype, benchmark);
                
                if (P[i].current_fitness < P[i].pbest_fitness) {
                    P[i].pbest_fitness = P[i].current_fitness;
                    P[i].pbest_pos = P[i].current_pos;
                }
            }

            // B. Select topological neighborhood best (Nbest)
            std::vector<std::vector<int>> Nbest(n);
            for(int i = 0; i < n; i++) {
                int left = (i - 1 + n) % n;
                int right = (i + 1) % n;

                int best_idx = i;
                if (P[left].pbest_fitness < P[best_idx].pbest_fitness) best_idx = left;
                if (P[right].pbest_fitness < P[best_idx].pbest_fitness) best_idx = right;

                Nbest[i] = P[best_idx].pbest_pos;
            }

            // C. Compute velocity and update Discrete Position (INTEGERB rule from pseudocode)
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < total_dimensions; d++) {
                    double r1 = get_random_double();
                    double r2 = get_random_double();

                    // Accumulate velocity based on distance to Pbest and Nbest
                    P[i].velocity[d] += (phi1 * r1 * (P[i].pbest_pos[d] - P[i].current_pos[d])) +
                                        (phi2 * r2 * (Nbest[i][d] - P[i].current_pos[d]));
                    
                    // Clamp velocity
                    P[i].velocity[d] = std::clamp(P[i].velocity[d], -vMax, vMax);

                    // Normalize to a probability using Sigmoid
                    double prob = sigmoid(P[i].velocity[d]);

                    // The Discrete Decision tree
                    if (flip(prob)) {
                        P[i].current_pos[d] = Nbest[i][d]; // Steal gene from neighbor
                    } else if (flip(1.0 - prob)) {
                        P[i].current_pos[d] = P[i].pbest_pos[d]; // Revert to personal best
                    }
                    // Else: retain current gene P[i].current_pos[d]
                }
                
                // Push the newly constructed flat array back into the structured Genotype
                unflatten_genotype(P[i]);
            }

            // D. Apply standard CGP mutation
            for (int i = 0; i < n; i++) {
                apply_discrete_mutation(P[i]);
            }
        } 

        // The core algorithm execution
        Genotype optimize(Generator& generator, Evaluator& evaluator, const PluReader& benchmark) {
            std::vector<Particle> P(swarm_size);
            
            Genotype global_best_genotype;
            int global_best_fitness = 99999999; 

            // 1. Initialization
            std::cout << "Initializing swarm...\n";  
            for (int i = 0; i < swarm_size; i++) {
                P[i].genotype = generator.generate_random_individual();
                P[i].current_pos = flatten_genotype(P[i].genotype);
                P[i].pbest_pos = P[i].current_pos;
                P[i].velocity = std::vector<double>(total_dimensions, 0.0);
                P[i].pbest_fitness = 99999999; 
                std::cout << "Particle " << i + 1 << " initialized.\n";
            }

            for (int iter = 1; iter <= max_iterations; iter++) {
                
                // A. Compute fitness & update Pbest
                for (int i = 0; i < swarm_size; i++) {
                    P[i].current_fitness = evaluator.calculate_fitness(P[i].genotype, benchmark);
                    
                    if (P[i].current_fitness < P[i].pbest_fitness) {
                        P[i].pbest_fitness = P[i].current_fitness;
                        P[i].pbest_pos = P[i].current_pos;
                    }

                    if (P[i].current_fitness < global_best_fitness) {
                        global_best_fitness = P[i].current_fitness;
                        global_best_genotype = P[i].genotype;
                    }
                }

                if (global_best_fitness == 0) {
                    std::cout << "\nSUCCESS: Swarm synthesized the circuit at iteration " << iter << "!\n";
                    break;
                }

                // B. Select topological neighborhood best (Nbest)
                std::vector<std::vector<int>> Nbest(swarm_size);
                for (int i = 0; i < swarm_size; i++) {
                    int left = (i - 1 + swarm_size) % swarm_size;
                    int right = (i + 1) % swarm_size;
                    
                    int best_idx = i;
                    if (P[left].pbest_fitness < P[best_idx].pbest_fitness) best_idx = left;
                    if (P[right].pbest_fitness < P[best_idx].pbest_fitness) best_idx = right;
                    
                    Nbest[i] = P[best_idx].pbest_pos;
                }

                // C. Compute velocity & update Discrete Position (INTEGERB rule from pseudocode)
                for (int i = 0; i < swarm_size; i++) {
                    for (int d = 0; d < total_dimensions; d++) {
                        double r1 = get_random_double();
                        double r2 = get_random_double();

                        // Accumulate velocity based on distance to Pbest and Nbest
                        P[i].velocity[d] += (phi1 * r1 * (P[i].pbest_pos[d] - P[i].current_pos[d])) +
                                            (phi2 * r2 * (Nbest[i][d] - P[i].current_pos[d]));
                        
                        // Clamp velocity
                        P[i].velocity[d] = std::clamp(P[i].velocity[d], -vMax, vMax);

                        // Normalize to a probability using Sigmoid
                        double prob = sigmoid(P[i].velocity[d]);

                        // The Discrete Decision tree
                        if (flip(prob)) {
                            P[i].current_pos[d] = Nbest[i][d]; // Steal gene from neighbor
                        } else if (flip(1.0 - prob)) {
                            P[i].current_pos[d] = P[i].pbest_pos[d]; // Revert to personal best
                        }
                        // Else: retain current gene P[i].current_pos[d]
                    }
                    
                    // Push the newly constructed flat array back into the structured Genotype
                    unflatten_genotype(P[i]);
                }

                // D. Apply standard CGP mutation
                for (int i = 0; i < swarm_size; i++) {
                    apply_discrete_mutation(P[i]);
                }

                // if (iter % 10000 == 0) {
                std::cout << "Iteration " << iter << " | Global Best Error: " << global_best_fitness << "\n";
                // }
            }

            return global_best_genotype;
        }
    };
}