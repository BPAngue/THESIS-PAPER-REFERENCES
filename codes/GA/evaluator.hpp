#pragma once
#include "logiswarm_types.hpp"
#include "reader.hpp"
#include <vector>
#include <map>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// =============================================================================
// evaluator.hpp — mirrors CGP++ Evaluator.h + LogicSynthesisProblem.h +
//                 BlackBoxProblem.h exactly
//
// Key CGP++ behaviors replicated:
//   1. genome is FLAT INT ARRAY — gene_at(position) used everywhere
//   2. position_from_node_number: (node_num - num_inputs) * (max_arity + 1)
//   3. node_value_map (std::map<int,uint32_t>) — cleared per pattern
//   4. decode_path uses visit_node recursively, then sorts active_nodes
//   5. evaluate_iterative: only active nodes, uses node_value_map.at()
//   6. num_bits = min(2^num_inputs, 32) — MAX_BITS cap
//   7. Hamming distance via manual get_bit loop (not __builtin_popcount)
//   8. Species::min_gene / max_gene logic replicated
// =============================================================================

namespace LogiSwarm {

    // -------------------------------------------------------------------------
    // Gate functions — uint32_t to match CGP++ unsigned int EVALUATION_TYPE
    // -------------------------------------------------------------------------
    inline uint32_t gate_NULL (uint32_t a, uint32_t b) { return 0; }
    inline uint32_t gate_NOR  (uint32_t a, uint32_t b) { return ~(a | b); }
    inline uint32_t gate_INHb (uint32_t a, uint32_t b) { return b & ~a; }
    inline uint32_t gate_NOTa (uint32_t a, uint32_t b) { return ~a; }
    inline uint32_t gate_INHa (uint32_t a, uint32_t b) { return a & ~b; }
    inline uint32_t gate_NOTb (uint32_t a, uint32_t b) { return ~b; }
    inline uint32_t gate_XOR  (uint32_t a, uint32_t b) { return a ^ b; }
    inline uint32_t gate_NAND (uint32_t a, uint32_t b) { return ~(a & b); }
    inline uint32_t gate_AND  (uint32_t a, uint32_t b) { return a & b; }
    inline uint32_t gate_XNOR (uint32_t a, uint32_t b) { return ~(a ^ b); }
    inline uint32_t gate_BUFb (uint32_t a, uint32_t b) { return b; }
    inline uint32_t gate_IMPb (uint32_t a, uint32_t b) { return ~a | b; }
    inline uint32_t gate_BUFa (uint32_t a, uint32_t b) { return a; }
    inline uint32_t gate_IMPa (uint32_t a, uint32_t b) { return a | ~b; }
    inline uint32_t gate_OR   (uint32_t a, uint32_t b) { return a | b; }
    inline uint32_t gate_ID   (uint32_t a, uint32_t b) { return ~0u; }

    typedef uint32_t (*GateFunction)(uint32_t, uint32_t);

    inline const GateFunction AllGates[] = {
        gate_NULL, gate_NOR,  gate_INHb, gate_NOTa,
        gate_INHa, gate_NOTb, gate_XOR,  gate_NAND,
        gate_AND,  gate_XNOR, gate_BUFb, gate_IMPb,
        gate_BUFa, gate_IMPa, gate_OR,   gate_ID
    };

    // -------------------------------------------------------------------------
    // Function sets — mirrors CGP++ BooleanFunctions.h
    // -------------------------------------------------------------------------
    inline const int FunctionSetReduced[]  = { GateType::AND,  GateType::OR,
                                                GateType::NAND, GateType::NOR };

    inline const int FunctionSetExtended[] = { GateType::BUFa, GateType::NOTa,
                                                GateType::AND,  GateType::OR,
                                                GateType::XOR,  GateType::NAND,
                                                GateType::NOR,  GateType::XNOR };

    inline const int FunctionSetCrypto[]   = { GateType::AND,  GateType::XOR,
                                                GateType::OR,   GateType::XNOR,
                                                GateType::INHb };

    inline const int* get_function_set(FunctionSet fs, int& size) {
        switch (fs) {
            case REDUCED:  size = 4; return FunctionSetReduced;
            case EXTENDED: size = 8; return FunctionSetExtended;
            case CRYPTO:   size = 5; return FunctionSetCrypto;
            default: throw std::invalid_argument("Unknown FunctionSet");
        }
    }

    inline std::string function_set_name(FunctionSet fs) {
        switch (fs) {
            case REDUCED:  return "Reduced  {AND, OR, NAND, NOR}";
            case EXTENDED: return "Extended {BUFa, NOTa, AND, OR, XOR, NAND, NOR, XNOR}";
            case CRYPTO:   return "Crypto   {AND, XOR, OR, XNOR, INHb}";
            default:       return "Unknown";
        }
    }

    // =========================================================================
    // Evaluator
    // =========================================================================
    class Evaluator {
    private:
        int num_inputs;
        int num_nodes;
        int num_outputs;
        int max_arity;
        int genome_size;
        int levels_back_;
        int num_functions_;
        FunctionSet active_set;
        const int* set_indices;
        int set_size;

        int num_bits;
        const int MAX_BITS = 32;

        // mirrors CGP++ node_value_map and node_number_map
        std::map<int, uint32_t> node_value_map;
        std::map<int, int>      node_number_map;

        // -----------------------------------------------------------------------
        // gene_at — mirrors CGP++ Evaluator::gene_at for integer genome
        // -----------------------------------------------------------------------
        int gene_at(const Genotype& ind, int position) const {
            return ind.genome[position];
        }

        // -----------------------------------------------------------------------
        // position_from_node_number — mirrors CGP++ Species::position_from_node_number
        // -----------------------------------------------------------------------
        int position_from_node_number(int node_number) const {
            return (node_number - num_inputs) * (max_arity + 1);
        }

        // -----------------------------------------------------------------------
        // decode_genotype_at — mirrors CGP++ Species::decode_genotype_at
        // 0=CONNECTION_GENE, 1=FUNCTION_GENE, 2=OUTPUT_GENE
        // -----------------------------------------------------------------------
        int decode_genotype_at(int position) const {
            if (position >= num_nodes * (max_arity + 1)) {
                return 2; // OUTPUT_GENE
            } else if (position % (max_arity + 1) == 0) {
                return 1; // FUNCTION_GENE
            } else {
                return 0; // CONNECTION_GENE
            }
        }

        // -----------------------------------------------------------------------
        // node_number_from_position — mirrors CGP++ Species::node_number_from_position
        // -----------------------------------------------------------------------
        int node_number_from_position(int position) const {
            int phenotype = decode_genotype_at(position);
            if (phenotype == 2) { // OUTPUT_GENE
                return num_inputs + num_nodes + (position - (num_nodes * max_arity));
            } else {
                return num_inputs + (position / (max_arity + 1));
            }
        }

        // -----------------------------------------------------------------------
        // min_gene — mirrors CGP++ Species::min_gene exactly
        // -----------------------------------------------------------------------
        int min_gene(int position) const {
            int min_g;
            int phenotype = decode_genotype_at(position);
            if (phenotype == 2) { // OUTPUT_GENE
                min_g = num_inputs + num_nodes - levels_back_;
            } else if (phenotype == 1) { // FUNCTION_GENE
                min_g = 0;
            } else { // CONNECTION_GENE
                int node_number = node_number_from_position(position);
                min_g = node_number - levels_back_;
            }
            if (min_g < 0) min_g = 0;
            return min_g;
        }

        // -----------------------------------------------------------------------
        // max_gene — mirrors CGP++ Species::max_gene exactly
        // -----------------------------------------------------------------------
        int max_gene(int position) const {
            int max_g;
            int phenotype = decode_genotype_at(position);
            if (phenotype == 2) { // OUTPUT_GENE
                max_g = num_inputs + num_nodes - 1;
            } else if (phenotype == 1) { // FUNCTION_GENE
                max_g = num_functions_ - 1;
            } else { // CONNECTION_GENE
                max_g = node_number_from_position(position) - 1;
            }
            return max_g;
        }

        // -----------------------------------------------------------------------
        // visit_node — mirrors CGP++ Evaluator::visit_node exactly
        // -----------------------------------------------------------------------
        void visit_node(const Genotype& ind,
                        std::vector<int>& active_nodes,
                        int node_num) {
            if (node_number_map.count(node_num) == 1) return;
            if (node_num < num_inputs) return;

            active_nodes.push_back(node_num);
            node_number_map.insert({node_num, 1});

            int position = position_from_node_number(node_num);
            for (int i = 1; i <= max_arity; i++) {
                int connection_gene = gene_at(ind, position + i);
                visit_node(ind, active_nodes, connection_gene);
            }
        }

        // -----------------------------------------------------------------------
        // decode_path — mirrors CGP++ Evaluator::decode_path exactly
        // -----------------------------------------------------------------------
        std::vector<int> decode_path(const Genotype& ind) {
            std::vector<int> active_nodes;
            node_number_map.clear();

            for (int i = 0; i < num_outputs; i++) {
                int position = genome_size - i - 1;
                int output   = gene_at(ind, position);
                visit_node(ind, active_nodes, output);
            }

            std::sort(active_nodes.begin(), active_nodes.end());
            return active_nodes;
        }

        // -----------------------------------------------------------------------
        // get_bit — mirrors CGP++ LogicSynthesisProblem::get_bit exactly
        // -----------------------------------------------------------------------
        int get_bit(uint32_t n, uint32_t k) const {
            return (n >> k) & 1;
        }

        // -----------------------------------------------------------------------
        // evaluate_per_output — mirrors CGP++ LogicSynthesisProblem::evaluate(E,E)
        // -----------------------------------------------------------------------
        int evaluate_per_output(uint32_t output_real, uint32_t output_individual) {
            int diff = 0;
            uint32_t compare = output_individual ^ output_real;
            for (int j = 0; j < num_bits; j++) {
                uint32_t temp = compare; // mirrors CGP++: E temp = compare inside loop
                diff += get_bit(temp, j);
            }
            return diff;
        }

        // -----------------------------------------------------------------------
        // evaluate_iterative — mirrors CGP++ Evaluator::evaluate_iterative exactly
        // -----------------------------------------------------------------------
        void evaluate_iterative(const Genotype& ind,
                                 const std::vector<int>& active_nodes,
                                 const std::vector<uint32_t>& inputs_vec,
                                 std::vector<uint32_t>& outputs_vec) {
            node_value_map.clear();

            uint32_t arguments[2];

            for (int node_num : active_nodes) {
                int node_pos = position_from_node_number(node_num);
                int function = gene_at(ind, node_pos);

                for (int i = 0; i < max_arity; i++) {
                    int node_arg = gene_at(ind, node_pos + i + 1);
                    if (node_arg < num_inputs) {
                        arguments[i] = inputs_vec[node_arg];
                    } else {
                        arguments[i] = node_value_map.at(node_arg);
                    }
                }

                int gate_idx  = set_indices[function];
                uint32_t result = AllGates[gate_idx](arguments[0], arguments[1]);
                node_value_map.insert({node_num, result});
            }

            // Collect outputs
            for (int i = 0; i < num_outputs; i++) {
                int output_pos = genome_size - i - 1;
                int output_val = gene_at(ind, output_pos);

                uint32_t value;
                if (output_val < num_inputs) {
                    value = inputs_vec[output_val];
                } else {
                    value = node_value_map.at(output_val);
                }
                outputs_vec.push_back(value);
            }
        }

    public:
        Evaluator(int in, int nodes, int out, int l_back, FunctionSet fs = REDUCED)
            : num_inputs(in), num_nodes(nodes), num_outputs(out),
              levels_back_(l_back), active_set(fs), max_arity(2)
        {
            set_indices    = get_function_set(fs, set_size);
            num_functions_ = set_size;
            genome_size    = num_nodes * (max_arity + 1) + num_outputs;

            num_bits = (int)std::pow(2, num_inputs);
            if (num_bits > MAX_BITS) num_bits = MAX_BITS;
        }

        int num_functions()  const { return set_size; }
        int get_genome_size() const { return genome_size; }
        int get_min_gene(int pos) const { return min_gene(pos); }
        int get_max_gene(int pos) const { return max_gene(pos); }
        FunctionSet get_active_set() const { return active_set; }

        // -----------------------------------------------------------------------
        // calculate_fitness — mirrors CGP++ BlackBoxProblem::evaluate_individual
        // -----------------------------------------------------------------------
        int calculate_fitness(const Genotype& ind, const CgpPluReader& plu_data) {
            std::vector<int> active_nodes = decode_path(ind);
            int total_diff = 0;

            for (int p = 0; p < plu_data.num_patterns; p++) {
                // mirrors CGP++: inputs stored as inputs[instance][var]
                std::vector<uint32_t> inputs_vec(num_inputs);
                for (int i = 0; i < num_inputs; i++) {
                    inputs_vec[i] = plu_data.inputs[p][i];
                }

                std::vector<uint32_t> outputs_vec;
                evaluate_iterative(ind, active_nodes, inputs_vec, outputs_vec);

                for (int o = 0; o < num_outputs; o++) {
                    total_diff += evaluate_per_output(
                        plu_data.outputs[p][o],
                        outputs_vec[o]
                    );
                }
            }

            return total_diff;
        }
    };

} // namespace LogiSwarm
