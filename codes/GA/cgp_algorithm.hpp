#pragma once
#include "logiswarm_types.hpp"
#include "generator.hpp"
#include "evaluator.hpp"
#include "reader.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <map>

// =============================================================================
// cgp_algorithm.hpp — mirrors CGP++ OnePlusLambda.h + ProbabilisticPoint.h
//
// Key CGP++ behaviors replicated:
//   1. ProbabilisticPoint mutation:
//      - num_mutations = (int)(mutation_rate * genome_size)  [fixed count]
//      - picks random_pos from [0, genome_size-1] uniformly
//      - uses min_gene(pos)/max_gene(pos) from Species for valid range
//      - same position CAN be picked multiple times (no dedup)
//   2. OnePlusLambda (1+λ)-ES with neutral genetic drift:
//      - collect Q_better and Q_equal
//      - if Q_better non-empty: pick random from Q_better
//      - else if Q_equal non-empty: pick random from Q_equal (NGD)
//      - else: keep current parent
//   3. fitness_evaluations counts lambda per generation
//   4. Termination: max_generations OR ideal_fitness reached
// =============================================================================

namespace LogiSwarm {

    class CGPAlgorithm {
    private:
        int lambda;
        double mutation_rate;
        int max_generations;
        int num_inputs;
        int num_outputs;
        int num_nodes;
        int levels_back;
        int num_functions;
        int genome_size;
        bool neutral_genetic_drift;
        std::mt19937 rng;

        // -----------------------------------------------------------------------
        // mutate — mirrors CGP++ ProbabilisticPoint::variate exactly
        //
        // CGP++ ProbabilisticPoint:
        //   int num_mutations = mutation_rate * genome_size;
        //   for (int i = 0; i < num_mutations; i++) {
        //       random_pos = random_integer(0, genome_size - 1);
        //       genome[random_pos] = random_integer(min_gene(pos), max_gene(pos));
        //   }
        //
        // Key: random_pos is picked uniformly from entire genome — same position
        // can be picked multiple times (no dedup). This is different from
        // per-gene probabilistic mutation.
        // -----------------------------------------------------------------------
        Genotype mutate(const Genotype& parent, const Evaluator& evaluator) {
            Genotype offspring = parent;

            // mirrors CGP++: int num_mutations = mutation_rate * genome_size
            int num_mutations = (int)(mutation_rate * genome_size);

            for (int i = 0; i < num_mutations; i++) {
                // mirrors CGP++: random_pos = random_integer(0, genome_size - 1)
                int random_pos = get_random_int(0, genome_size - 1);

                // mirrors CGP++: min_gene = species->min_gene(random_pos)
                //                max_gene = species->max_gene(random_pos)
                int min_g = evaluator.get_min_gene(random_pos);
                int max_g = evaluator.get_max_gene(random_pos);

                // mirrors CGP++: genome[random_pos] = random_integer(min_gene, max_gene)
                offspring.genome[random_pos] = get_random_int(min_g, max_g);
            }

            return offspring;
        }

        int get_random_int(int min, int max) {
            if (min > max) max = min;
            std::uniform_int_distribution<int> dist(min, max);
            return dist(rng);
        }

        int pick_random(int size) {
            std::uniform_int_distribution<int> dist(0, size - 1);
            return dist(rng);
        }

    public:
        CGPAlgorithm(int lam, double mut_rate, int max_gens,
                     int in, int out, int nodes, int l_back,
                     int funcs, int g_size, bool ngd = true)
            : lambda(lam), mutation_rate(mut_rate), max_generations(max_gens),
              num_inputs(in), num_outputs(out), num_nodes(nodes),
              levels_back(l_back), num_functions(funcs),
              genome_size(g_size), neutral_genetic_drift(ngd)
        {
            std::random_device rd;
            rng.seed(rd());
        }

        // -----------------------------------------------------------------------
        // evolve — mirrors CGP++ OnePlusLambda::evolve() exactly
        //
        // Algorithm 1 from paper (1+λ)-ES with neutral genetic drift:
        //   1. initialize parent P
        //   2. repeat:
        //      3. Q = breed(P) — lambda offspring by mutation
        //      4. Evaluate(Q)
        //      5. Q+ = individuals with better fitness than P
        //      6. Q= = individuals with equal fitness to P
        //      8. if |Q+| > 0: P = random from Q+
        //     12. else if |Q=| > 0: P = random from Q= (neutral drift)
        //     16. until termination
        //   17. return P
        // -----------------------------------------------------------------------
        Genotype evolve(Generator& generator,
                        Evaluator& evaluator,
                        const CgpPluReader& plu_data)
        {
            // mirrors CGP++: best_fitness = fitness->worst_value() = INT_MAX
            int best_fitness = INT_MAX;
            bool is_ideal    = false;
            int fitness_evaluations = 0;
            int generation_number   = 1;

            // mirrors CGP++: parent_index = -1 for first selection
            // Here we just initialize parent directly
            Genotype parent = generator.generate_random_individual(evaluator);
            best_fitness    = evaluator.calculate_fitness(parent, plu_data);

            std::cout << "Initial fitness (Hamming distance): " << best_fitness << "\n";

            // mirrors CGP++: while (generation_number <= max_generations && !is_ideal)
            while (generation_number <= max_generations && !is_ideal) {

                // mirrors CGP++: breed lambda offspring
                std::vector<Genotype> offspring(lambda);
                std::vector<int>      offspring_fitness(lambda);

                for (int i = 0; i < lambda; i++) {
                    offspring[i]         = mutate(parent, evaluator);
                    offspring_fitness[i] = evaluator.calculate_fitness(offspring[i], plu_data);
                }

                // mirrors CGP++: fitness_evaluations += lambda
                fitness_evaluations += lambda;

                // mirrors CGP++ select_parent():
                // Q+ = better, Q= = equal
                std::vector<int> Q_better;
                std::vector<int> Q_equal;

                for (int i = 0; i < lambda; i++) {
                    // mirrors CGP++: fitness->is_better(offspring, parent)
                    // minimizing fitness: offspring < parent = better
                    if (offspring_fitness[i] < best_fitness) {
                        Q_better.push_back(i);
                    } else if (offspring_fitness[i] == best_fitness) {
                        Q_equal.push_back(i);
                    }
                }

                // mirrors CGP++ select_parent() selection logic
                if (!Q_better.empty()) {
                    // Step 9-10: choose random from Q+
                    int r      = pick_random((int)Q_better.size());
                    int chosen = Q_better[r];
                    parent       = offspring[chosen];
                    best_fitness = offspring_fitness[chosen];

                } else if (neutral_genetic_drift && !Q_equal.empty()) {
                    // Step 13-14: choose random from Q= (neutral genetic drift)
                    int r      = pick_random((int)Q_equal.size());
                    int chosen = Q_equal[r];
                    parent = offspring[chosen];
                    // best_fitness stays the same
                }
                // else: keep current parent (mirrors CGP++: return parent_index)

                // mirrors CGP++ EvolutionaryAlgorithm::report()
                std::cout << "Generation # " << generation_number
                          << " :: Best Fitness: " << best_fitness << "\n";

                // mirrors CGP++ check_ideal()
                if (best_fitness == 0) {
                    is_ideal = true;
                    std::cout << "Ideal fitness has been reached in generation # "
                              << generation_number << "\n";
                }

                generation_number++;
            }

            // mirrors CGP++ Evolver::execute_job() report
            std::cout << "\nJob # 1 :: Evaluations: " << fitness_evaluations
                      << " :: Best Fitness: " << best_fitness << "\n";

            return parent;
        }
    };

} // namespace LogiSwarm
