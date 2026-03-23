#pragma once
#include "get_active_nodes.hpp"
#include "reader.hpp"
#include "circuit_types.hpp"
#include <vector>
#include <cstdint> 

namespace LogicCircuitSynthesis {

    // Procedure 2.2: Decoding CGP to get the output
    // Only processes nodes stored in the active_nodes (NP) vector.
    inline void DecodeCGP(const Genotype& ind, const PluReader& plu_data, 
                          int pattern_idx, const std::vector<int>& active_nodes,
                          std::vector<uint32_t>& state_buffer, int num_inputs,
                          const int* set_indices) {
        
        // 1. Load primary inputs (Lines 2-4)
        for (int i = 0; i < num_inputs; i++) {
            state_buffer[i] = plu_data.inputs[i][pattern_idx];
        }

        // 2. Simulate only active nodes (Lines 5-13)
        for (int node_address : active_nodes) {
            int n_idx = node_address - num_inputs; // Convert address to node index
            const Node& node = ind.logic_nodes[n_idx];
            
            uint32_t in_a = state_buffer[node.input_1_idx];
            uint32_t in_b = state_buffer[node.input_2_idx];

            int gate_idx = set_indices[node.function_idx];
            state_buffer[node_address] = AllGates[gate_idx](in_a, in_b);
        }
    }

    // Procedure 2.3: Calculating the fitness of a CGP genotype
    inline int CalculateFitness(const Genotype& ind, const PluReader& plu_data,
                                int num_inputs, int num_nodes, int num_outputs,
                                const int* set_indices) {
        
        // 1. Find active nodes ONCE per genotype (Line 2)
        std::vector<int> active_nodes = GetActiveNodes(ind, num_inputs, num_nodes, num_outputs);
        
        int total_hamming_distance = 0;
        std::vector<uint32_t> state_buffer(num_inputs + num_nodes, 0);

        // 2. Loop through all fitness cases (Line 4)
        for (int p = 0; p < plu_data.num_patterns; p++) {
            
            // Call Procedure 2.2 (Line 5)
            DecodeCGP(ind, plu_data, p, active_nodes, state_buffer, num_inputs, set_indices);

            // 3. Evaluate and accumulate (Lines 6-7)
            for (int o = 0; o < num_outputs; o++) {
                uint32_t actual_out = state_buffer[ind.output_genes[o]];
                uint32_t target_out = plu_data.outputs[o][p];
                total_hamming_distance += __builtin_popcount(actual_out ^ target_out);
            }
        }

        return total_hamming_distance;
    }
}