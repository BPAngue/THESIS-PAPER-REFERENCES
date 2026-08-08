#pragma once
#include <vector>
#include "circuit_types.hpp"

namespace LogicCircuitSynthesis {
    
    // Procedure 2.1: Determining which nodes need to be processed
    // Returns a vector of indices of the nodes that are active
    inline std::vector<int> GetActiveNodes(const Genotype& ind, int num_inputs, int num_nodes, int num_outputs) {
        int M = num_inputs + num_nodes;
        std::vector<bool> NU(M, false); // "Node Used" array (Line 3)

        // 1. Mark nodes connected to primary outputs as TRUE (Line 6)
        for (int i = 0; i < num_outputs; i++) {
            NU[ind.output_genes[i]] = true;
        }

        // 2. Backwards trace: If a node is used, mark its inputs as used (Lines 8-18)
        // We iterate backwards from the last node to the first node.
        for (int i = num_nodes - 1; i >= 0; i--) {
            int current_node_address = num_inputs + i;

            if (NU[current_node_address]) {
                const Node& node = ind.logic_nodes[i];
                // Mark the sources of this node's inputs as active
                NU[node.input_1_idx] = true;
                NU[node.input_2_idx] = true;
            }
        }

        // 3. Store active node addresses in NP (Lines 20-25)
        std::vector<int> NP;
        for (int j = num_inputs; j < M; j++) {
            if (NU[j]) {
                NP.push_back(j);
            }
        }

        return NP; // returns the list of active node addresses (n_u is NP.size())
    }
}