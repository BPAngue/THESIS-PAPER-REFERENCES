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

    // struct for an individual in a population
    struct Individual {
        Genotype genotype;
        int fitness;
        int hamming_distance;
    };
    
    class CGPSAAlgorithm {
    private:
        int num_inputs;
        int num_outputs;
        int max_generations;
        int max_nodes;
        int num_gate_types;
        int max_bits;

        int levels_back;
        int total_dimensions;
        int lambda;
        double mutation_rate;

        double temperature_max;
        double temperature_min;
        double cooling_rate;

        FunctionSet function_set;
        const int* set_indices;
        int set_size;

        std::mt19937 rng;

    public:
        CGPSAAlgorithm(int in, int out, int max_gen, int nodes, int g_types, int l_back, int t_dimensions,
                       int lamb, double mut_rate, double t_max, double t_min, double c_rate, FunctionSet fs)
                    : num_inputs(in), num_outputs(out), max_generations(max_gen), max_nodes(nodes), num_gate_types(g_types),
                      levels_back(l_back), total_dimensions(t_dimensions), lambda(lamb), mutation_rate(mut_rate),
                      temperature_max(t_max), temperature_min(t_min), cooling_rate(c_rate), function_set(fs) {
            
            set_indices = get_function_set(function_set, set_size);
            max_bits = num_outputs * (1u << num_inputs);
            std::cout << "Total_dimensions: " << total_dimensions << "\n"; 
            
            std::random_device rd;
            rng.seed(rd());
        }

        // random double
        double get_random_double() {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            return dist(rng);
        }

        // returns transistor count
        int transistor_count(int global_gate_id) {
            switch(global_gate_id) {
                case NULL_GATE: return 0;
                case NOR:       return 4;
                case INHb:      return 8;   // (not a) AND b  =  NOT(2) + AND(6)
                case NOTa:      return 2;
                case INHa:      return 8;   // a AND (not b)  =  NOT(2) + AND(6)
                case NOTb:      return 2;
                case XOR:       return 9;
                case NAND:      return 4;
                case AND:       return 6;
                case XNOR:      return 9;
                case BUFb:      return 2;
                case IMPb:      return 8;   // (not a) OR b   =  NOT(2) + OR(6)
                case BUFa:      return 2;
                case IMPa:      return 8;   // a OR (not b)   =  NOT(2) + OR(6)
                case OR:        return 6;
                case ID:        return 0;
                default:        return 0;
            }
        }

        // count active transistors
        int count_active_transistors(const Genotype& ind) {
            std::vector<int> active_nodes = GetActiveNodes(ind, num_inputs, max_nodes, num_outputs);

            int total = 0;
            for (int node_address : active_nodes) {
                int n_idx = node_address - num_inputs;
                int global_gate_id = set_indices[ind.logic_nodes[n_idx].function_idx];
                total += transistor_count(global_gate_id);
            }

            return total;
        }

        // Fitness function used in the paper
        int calculate_paper_fitness(int hamming_distance, int num_transistors) {
            int b = max_bits - hamming_distance; // number of correct in the truth table

            if (b < max_bits) {
                return b * 100;
            } else {
                return b * 100 + (100 - num_transistors / 10);
            }
        }

        // point mutation
        Genotype mutate(Genotype& parent, Generator& generator) {
            Genotype offspring = parent;

            for (int g = 0; g < max_nodes; g++) {
                // mutate function gene
                if (get_random_double() < mutation_rate) {
                    offspring.logic_nodes[g].function_idx = generator.get_random_int(0, num_gate_types - 1);
                }

                // Determine valid connection range for this node
                int min_conn = (g >= levels_back) ? (num_inputs + g - levels_back) : 0;
                int max_conn = num_inputs + g - 1;

                // mutate input 1
                if (get_random_double() < mutation_rate) {
                    offspring.logic_nodes[g].input_1_idx = generator.get_random_int(min_conn, max_conn);
                }

                // mutate input 2
                if (get_random_double() < mutation_rate) {
                    offspring.logic_nodes[g].input_2_idx = generator.get_random_int(min_conn, max_conn);
                }
            }

            // mutate output genes
            int max_total_idx = num_inputs + max_nodes - 1;
            for (int o = 0; o < num_outputs; o++) {
                if (get_random_double() < mutation_rate) {
                    offspring.output_genes[o] = generator.get_random_int(0, max_total_idx);
                }
            }

            return offspring;
        }

        // The core algorithm execution
        Genotype optimize(Generator& generator, Evaluator& evaluator, const PluReader& benchmark) {

            Individual S; // best individual ever found (elitist save)
            Individual R; // the current working parent
            Individual N; // best child 

            S.fitness = -99999999;
            R.fitness = -99999999;
            S.hamming_distance = 99999999;
            R.hamming_distance = 99999999;

            // 1. Initialize population
            std::vector<Individual> population(1 + lambda);
            for (int i = 0; i < 1 + lambda; i++) {
                population[i].genotype = generator.generate_random_individual(); 
            }

            // 2. Initialize temperature
            double T = temperature_max;

            // 3. Initialize cooling rate (passed in as a parameter value)
            // 4. Compute fitness for all individuals and 5. the best becomes parent R and saved S;
            int best_idx = 0;
            for (int i = 0; i < 1 + lambda; i++) {
                population[i].hamming_distance = evaluator.calculate_fitness(population[i].genotype, benchmark);
                int transistors = count_active_transistors(population[i].genotype);
                population[i].fitness = calculate_paper_fitness(population[i].hamming_distance, transistors);

                if (population[i].fitness > S.fitness) {
                    best_idx = i;
                    R.genotype = population[i].genotype;
                    R.fitness = population[i].fitness;
                    R.hamming_distance = population[i].hamming_distance;

                    S.genotype = population[i].genotype;
                    S.fitness = population[i].fitness;
                    S.hamming_distance = population[i].hamming_distance;
                }
            }

            // pull R out, shrink to lambda children
            population.erase(population.begin() + best_idx);
            population.resize(lambda);

            int generation = 0;
            while (generation != max_generations) {
                N.fitness = -99999999;
                N.hamming_distance = 99999999;

                // 7. find best child N
                for (int i = 0; i < lambda; i++) {
                    if (population[i].fitness > N.fitness) {
                        N.genotype = population[i].genotype;
                        N.fitness = population[i].fitness;
                        N.hamming_distance = population[i].hamming_distance;
                    }
                }

                // 8. Fitness difference = fitness(N) - fitness(R)
                int delta_f = N.fitness - R.fitness;

                // 9. u random number
                double u = get_random_double();

                if (N.fitness > S.fitness) {
                    S.genotype = N.genotype;
                    S.fitness = N.fitness;
                    S.hamming_distance = N.hamming_distance;
                }

                if (delta_f > 0 || std::exp((double)delta_f / T) >= u) {
                    R.genotype = N.genotype;
                    R.fitness = N.fitness;
                    R.hamming_distance = N.hamming_distance;
                }

                // 14. create lambda children by mutating the parent R and 15. new population consists of parent R and lambda children
                for (int i = 0; i < lambda; i++) {
                    population[i].genotype = mutate(R.genotype, generator);

                    // calculate fitness of each children
                    population[i].hamming_distance = evaluator.calculate_fitness(population[i].genotype, benchmark);
                    int transistors = count_active_transistors(population[i].genotype);
                    population[i].fitness = calculate_paper_fitness(population[i].hamming_distance, transistors);
                }

                // 15. lower temperature
                T = cooling_rate * T;

                if (T < temperature_min) {
                    T = temperature_max;
                }

                // print results
                std::cout << "Generation " << (generation + 1) << " | Best fitness: " << S.fitness << " | Best Hamming Distance: " << S.hamming_distance << "\n";

                generation++;
            }

            return S.genotype;
        }
    };
}