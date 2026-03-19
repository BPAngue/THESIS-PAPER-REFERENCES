#pragma once
#include "logiswarm_types.hpp"
#include "evaluator.hpp"
#include <random>

// =============================================================================
// generator.hpp — mirrors CGP++ Individual::reset_genome() + Species exactly
//
// Key CGP++ behaviors replicated:
//   1. Genome is FLAT INT ARRAY of size genome_size = num_nodes*3 + num_outputs
//   2. Initialization iterates position by position using min_gene/max_gene
//      from Species — NOT node-by-node
//   3. Same min_gene/max_gene constraints as Species.h
// =============================================================================

namespace LogiSwarm {

    class Generator {
    private:
        int num_inputs;
        int num_outputs;
        int num_nodes;
        int levels_back;
        int num_functions;
        int genome_size;
        std::mt19937 rng;

    public:
        Generator(int in, int out, int nodes, int l_back, int funcs, int g_size)
            : num_inputs(in), num_outputs(out), num_nodes(nodes),
              levels_back(l_back), num_functions(funcs), genome_size(g_size)
        {
            std::random_device rd;
            rng.seed(rd());
        }

        int get_random_int(int min, int max) {
            if (min > max) max = min; // safety guard
            std::uniform_int_distribution<int> dist(min, max);
            return dist(rng);
        }

        // -----------------------------------------------------------------------
        // generate_random_individual — mirrors CGP++ Individual::reset_genome()
        //
        // CGP++ iterates position by position through the flat genome array
        // and uses Species::min_gene(pos) and Species::max_gene(pos) to
        // determine valid range at each position.
        //
        // This is fundamentally different from the original AI recreation which
        // iterated node by node using struct fields.
        // -----------------------------------------------------------------------
        Genotype generate_random_individual(const Evaluator& evaluator) {
            Genotype ind;
            ind.genome.resize(genome_size);
            ind.genome_size  = genome_size;
            ind.num_nodes    = num_nodes;
            ind.num_outputs  = num_outputs;

            // mirrors CGP++ Individual::reset_genome():
            // for (int i = 0; i < genome_size; i++)
            //     genome[i] = random_integer(min_gene(i), max_gene(i))
            for (int i = 0; i < genome_size; i++) {
                int min_g = evaluator.get_min_gene(i);
                int max_g = evaluator.get_max_gene(i);
                ind.genome[i] = get_random_int(min_g, max_g);
            }

            return ind;
        }
    };

} // namespace LogiSwarm
