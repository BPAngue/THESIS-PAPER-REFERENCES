#pragma once
#include <vector>

// =============================================================================
// logiswarm_types.hpp — mirrors CGP++ template_types.h and Parameters constants
//
// CGP++ genome is a FLAT INTEGER ARRAY, not a struct.
// Layout: [func][conn1][conn2] per node, then [output0][output1]...
// This is the key structural difference from the original AI recreation.
// =============================================================================

namespace LogiSwarm {

    // Mirrors CGP++ GateType enum from GBFS paper Table 4
    enum GateType {
        NULL_GATE = 0,
        NOR       = 1,
        INHb      = 2,
        NOTa      = 3,
        INHa      = 4,
        NOTb      = 5,
        XOR       = 6,
        NAND      = 7,
        AND       = 8,
        XNOR      = 9,
        BUFb      = 10,
        IMPb      = 11,
        BUFa      = 12,
        IMPa      = 13,
        OR        = 14,
        ID        = 15
    };

    // Function set identifiers
    enum FunctionSet {
        REDUCED  = 0,  // {AND, OR, NAND, NOR}
        EXTENDED = 1,  // {BUFa, NOTa, AND, OR, XOR, NAND, NOR, XNOR}
        CRYPTO   = 2   // {AND, XOR, OR, XNOR, INHb}
    };

    // ==========================================================================
    // Genotype — mirrors CGP++ flat integer genome array
    //
    // CGP++ genome layout (from Species.h):
    //   position 0,1,2       = node 0: [func, conn1, conn2]
    //   position 3,4,5       = node 1: [func, conn1, conn2]
    //   ...
    //   position n*3,n*3+1,n*3+2 = node n: [func, conn1, conn2]
    //   position num_nodes*3 + o  = output gene o
    //
    // genome_size = num_nodes * (max_arity + 1) + num_outputs
    //             = num_nodes * 3 + num_outputs
    // ==========================================================================
    struct Genotype {
        std::vector<int> genome;  // flat array — mirrors CGP++ int genome[]
        int genome_size;
        int num_nodes;
        int num_outputs;
        int max_arity = 2;        // fixed at 2 for Boolean functions

        // Helper: get gene at position (mirrors CGP++ gene_at())
        int gene_at(int position) const {
            return genome[position];
        }

        // Helper: position from node number (mirrors CGP++ Species::position_from_node_number)
        int position_from_node_number(int node_num, int num_inputs) const {
            return (node_num - num_inputs) * (max_arity + 1);
        }
    };

} // namespace LogiSwarm
