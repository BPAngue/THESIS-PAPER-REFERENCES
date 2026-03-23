#pragma once
#include "circuit_types.hpp"
#include <random>

namespace LogicCircuitSynthesis {
    // Generates a random candidate circuit solution
    class Generator {
    private:
        int num_inputs;
        int num_outputs;
        int num_nodes;
        int levels_back;
        int num_functions;
        std::mt19937 rng;

    public:
        Generator(int in, int out, int nodes, int l_back, int funcs) 
            : num_inputs(in), num_outputs(out), num_nodes(nodes), 
            levels_back(l_back), num_functions(funcs) {
            
            std::random_device rd;
            rng.seed(rd());
        }

        int get_random_int(int min, int max) {
            std::uniform_int_distribution<int> dist(min, max);
            return dist(rng);
        }

        Genotype generate_random_individual() {
            Genotype ind;
            ind.logic_nodes.resize(num_nodes);
            ind.output_genes.resize(num_outputs);

            for (int i = 0; i < num_nodes; i++) {
                int min_conn = (i >= levels_back) ? (num_inputs + i - levels_back) : 0;
                int max_conn = num_inputs + i - 1;

                ind.logic_nodes[i].function_idx = get_random_int(0, num_functions - 1);
                ind.logic_nodes[i].input_1_idx = get_random_int(min_conn, max_conn);
                ind.logic_nodes[i].input_2_idx = get_random_int(min_conn, max_conn);
            }

            int max_total_idx = num_inputs + num_nodes - 1;
            for (int o = 0; o < num_outputs; o++) {
                ind.output_genes[o] = get_random_int(0, max_total_idx);
            }

            return ind;
        }
    };
}