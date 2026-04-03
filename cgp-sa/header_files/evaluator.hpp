#pragma once
#include <vector>             // For state_buffer and handling genotype arrays
#include <string>             // For function_set_name return types
#include <stdexcept>          // For invalid_argument in get_function_set
#include "circuit_types.hpp"  // For Genotype, Node, GateType, and AllGates
#include "reader.hpp"         // For PluReader data access
#include "decode_fitness.hpp" // For CalculateFitness and DecodeCGP (Procedures 2.2 & 2.3)

namespace LogicCircuitSynthesis {
    class Evaluator {
    private:
        int num_inputs, num_nodes, num_outputs;
        const int* set_indices;
        int set_size;

    public:
        Evaluator(int in, int nodes, int out, FunctionSet fs = EXTENDED) 
            : num_inputs(in), num_nodes(nodes), num_outputs(out) {
            set_indices = get_function_set(fs, set_size);
        }

        int num_functions() const { 
            return set_size; 
        }

        int calculate_fitness(const Genotype& ind, const PluReader& plu_data) {
            return CalculateFitness(ind, plu_data, num_inputs, num_nodes, num_outputs, set_indices);
        }
    };
}