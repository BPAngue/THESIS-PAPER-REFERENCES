#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "circuit_types.hpp"
#include "get_active_nodes.hpp"

namespace LogicCircuitSynthesis {

    // Maps the global GateType ID to its boolean algebraic string representation
    inline std::string format_gate_string(int global_gate_id, const std::string& a, const std::string& b) {
        switch (global_gate_id) {
            case NULL_GATE: return "0";
            case NOR:       return "~(" + a + " | " + b + ")";
            case INHb:      return "(" + b + " & ~" + a + ")";
            case NOTa:      return "~" + a;
            case INHa:      return "(" + a + " & ~" + b + ")";
            case NOTb:      return "~" + b;
            case XOR:       return "(" + a + " ^ " + b + ")";
            case NAND:      return "~(" + a + " & " + b + ")";
            case AND:       return "(" + a + " & " + b + ")";
            case XNOR:      return "~(" + a + " ^ " + b + ")";
            case BUFb:      return b;
            case IMPb:      return "(~" + a + " | " + b + ")";
            case BUFa:      return a;
            case IMPa:      return "(" + a + " | ~" + b + ")";
            case OR:        return "(" + a + " | " + b + ")";
            case ID:        return "1";
            default:        return "UNKNOWN";
        }
    }

    // Translates the active genotype into boolean formulas
    inline void PrintCircuitFormula(const Genotype& ind, int num_inputs, int num_nodes, int num_outputs, FunctionSet fs) {
        // 1. Get active nodes to avoid printing junk logic
        std::vector<int> active_nodes = GetActiveNodes(ind, num_inputs, num_nodes, num_outputs);
        
        int set_size;
        const int* set_indices = get_function_set(fs, set_size);

        // 2. Array to store the string representation of each node's output
        std::vector<std::string> formulas(num_inputs + num_nodes);

        // 3. Initialize primary input strings
        for (int i = 0; i < num_inputs; i++) {
            formulas[i] = "In[" + std::to_string(i) + "]";
        }

        // 4. Build the formula sequentially for active nodes
        for (int node_address : active_nodes) {
            int n_idx = node_address - num_inputs;
            const Node& node = ind.logic_nodes[n_idx];

            std::string in_a = formulas[node.input_1_idx];
            std::string in_b = formulas[node.input_2_idx];

            int global_gate_id = set_indices[node.function_idx];
            formulas[node_address] = format_gate_string(global_gate_id, in_a, in_b);
        }

        // 5. Print the formula linked to each output gene
        std::cout << "\n--- Final Circuit Formulas ---\n";
        for (int o = 0; o < num_outputs; o++) {
            int out_address = ind.output_genes[o];
            std::cout << "Output[" << o << "] = " << formulas[out_address] << "\n";
        }
        std::cout << "------------------------------\n";
    }
}