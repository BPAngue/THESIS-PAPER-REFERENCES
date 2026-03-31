#pragma once
// cgp_es_algorithm.hpp
//
// Parallel Streams CGP Evolutionary Strategy
// ===========================================
// This is the correct search algorithm for Cartesian Genetic Programming.
//
// Why (1+λ) ES fits CGP
// ----------------------
// CGP's fitness landscape is almost entirely neutral — thousands of mutations
// change the genotype without changing fitness (inactive nodes).  This is a
// feature, not a bug: neutral drift slowly builds up useful subgraphs in
// inactive nodes that can be activated in a single future mutation.  This only
// works if the search operator is pure mutation.  PSO's velocity/interpolation
// mechanic actively destroys neutral structure by pulling inactive genes toward
// another particle's inactive genes, which carries no fitness signal.
//
// The (1+λ) ES:
//   - One parent per stream
//   - λ offspring per generation, each a point-mutated copy of the parent
//   - Keep the best offspring; if none beats the parent, keep the parent
//     (neutral selection: equal fitness offspring are accepted — this is
//      critical for CGP, it's what drives neutral drift)
//
// Parallel Streams + Information Exchange (PSO-PSO structure)
// ------------------------------------------------------------
// Multi-evolutionary phase  (paper Steps 2-5):
//   K independent streams evolve for N generations on K threads.
//   Each tracks its own sbest (best genotype ever seen in that stream).
//
// Single-evolutionary phase (paper Steps 6-9):
//   gbest = min(sbest_1 .. sbest_K)
//   Each stream whose current parent is worse than gbest replaces its parent
//   with a mutated copy of gbest.  This is the information exchange — it's a
//   biologically meaningful injection rather than velocity interpolation.
//   Runs for M generations sequentially.
//
// Stagnation restart:
//   If gbest has not improved for `stagnation_threshold` consecutive cycles,
//   the worst-performing stream is re-initialised from scratch.  A fresh
//   random genotype explores a completely different region while the other
//   streams continue refining the current best.
//
// All existing headers (circuit_types, generator, evaluator, get_active_nodes,
// reader, printer) are used unchanged.

#include "../circuit_types.hpp"
#include "../generator.hpp"
#include "../evaluator.hpp"
#include "../reader.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>
#include <thread>
#include <atomic>
#include <stdexcept>

namespace LogicCircuitSynthesis {

// ============================================================================
//  ParallelStreamsCGP
// ============================================================================
class ParallelStreamsCGP {
public:
    // -----------------------------------------------------------------------
    //  Configuration
    // -----------------------------------------------------------------------
    struct Config {
        // Stream / population
        int num_streams           = 4;    // K: independent (1+λ) streams (one per thread)
        int lambda                = 8;    // λ: offspring per parent per generation
        
        // Phase lengths
        int multi_phase_gens      = 2000; // N: generations per stream in multi-phase
        int single_phase_gens     = 500;  // M: generations in single-phase (all streams)
        int max_outer_cycles      = 100000;

        // Mutation
        double mutation_rate      = 0.01; // per-gene probability
                                          // higher than PSO variant because every
                                          // mutation counts — no velocity dampening

        // Stagnation restart
        int stagnation_threshold  = 15;   // cycles without gbest improvement → restart
                                          // the worst stream
    };

private:
    Config cfg_;

    int num_inputs_, num_outputs_, num_nodes_, levels_back_, num_functions_;

    // -----------------------------------------------------------------------
    //  Per-stream state
    // -----------------------------------------------------------------------
    struct Stream {
        int      id         = 0;
        Genotype parent;               // current parent genotype
        int      parent_fit = std::numeric_limits<int>::max();
        Genotype sbest;                // best genotype this stream has ever seen
        int      sbest_fit  = std::numeric_limits<int>::max();
    };

    std::vector<Stream>  streams_;
    Genotype             gbest_genotype_;
    int                  gbest_fitness_ = std::numeric_limits<int>::max();

    std::atomic<bool>    solved_{ false };

    // -----------------------------------------------------------------------
    //  Mutation
    //  Point-mutation: each gene flipped independently at mutation_rate.
    //  Inactive genes mutate at the same rate — this drives neutral drift.
    //  CGP connectivity constraints are preserved exactly as in the original
    //  apply_discrete_mutation.
    // -----------------------------------------------------------------------
    Genotype mutate(const Genotype& parent, std::mt19937& rng) const {
        Genotype offspring = parent;   // copy

        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_int_distribution<int>     func_dist(0, num_functions_ - 1);
        std::uniform_int_distribution<int>     out_dist(0, num_inputs_ + num_nodes_ - 1);

        for (int i = 0; i < num_nodes_; i++) {
            int min_conn = (i >= levels_back_)
                           ? (num_inputs_ + i - levels_back_) : 0;
            int max_conn = num_inputs_ + i - 1;
            if (max_conn < min_conn) max_conn = min_conn;

            std::uniform_int_distribution<int> conn_dist(min_conn, max_conn);

            if (prob(rng) < cfg_.mutation_rate)
                offspring.logic_nodes[i].function_idx = func_dist(rng);
            if (prob(rng) < cfg_.mutation_rate)
                offspring.logic_nodes[i].input_1_idx  = conn_dist(rng);
            if (prob(rng) < cfg_.mutation_rate)
                offspring.logic_nodes[i].input_2_idx  = conn_dist(rng);
        }

        for (int o = 0; o < num_outputs_; o++)
            if (prob(rng) < cfg_.mutation_rate)
                offspring.output_genes[o] = out_dist(rng);

        return offspring;
    }

    // -----------------------------------------------------------------------
    //  One (1+λ) generation for a single stream.
    //  Returns the fitness of the new parent.
    //
    //  Neutral selection rule (critical for CGP):
    //    Accept an offspring if its fitness is <= parent fitness (not strictly
    //    better).  This allows neutral drift through equal-fitness mutations,
    //    which continuously rewires inactive nodes into new configurations.
    // -----------------------------------------------------------------------
    int run_one_generation(Stream& stream,
                           Evaluator& evaluator,
                           const PluReader& benchmark,
                           std::mt19937& rng)
    {
        int best_offspring_fit = std::numeric_limits<int>::max();
        Genotype best_offspring = stream.parent;

        for (int l = 0; l < cfg_.lambda; l++) {
            Genotype offspring = mutate(stream.parent, rng);
            int fit = evaluator.calculate_fitness(offspring, benchmark);

            // Neutral selection: accept if fit <= parent (not just <)
            if (fit <= best_offspring_fit) {
                best_offspring_fit = fit;
                best_offspring     = offspring;
            }
        }

        // Replace parent if any offspring is at least as good
        if (best_offspring_fit <= stream.parent_fit) {
            stream.parent     = best_offspring;
            stream.parent_fit = best_offspring_fit;
        }

        // Update stream's personal best
        if (stream.parent_fit < stream.sbest_fit) {
            stream.sbest_fit = stream.parent_fit;
            stream.sbest     = stream.parent;
        }

        return stream.parent_fit;
    }

    // -----------------------------------------------------------------------
    //  Multi-evolutionary phase worker (one thread per stream)
    // -----------------------------------------------------------------------
    void evolve_stream_independent(Stream& stream,
                                   Evaluator& evaluator,
                                   const PluReader& benchmark)
    {
        // Independent RNG per thread — critical for statistical independence
        std::mt19937 rng(std::random_device{}() ^
                         static_cast<std::mt19937::result_type>(stream.id));

        for (int g = 0; g < cfg_.multi_phase_gens && !solved_; ++g) {
            run_one_generation(stream, evaluator, benchmark, rng);

            if (stream.parent_fit == 0) {
                solved_.store(true);
                return;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Single-evolutionary phase (sequential, all streams share gbest)
    //
    //  Information exchange mechanism:
    //    Any stream whose parent is worse than gbest replaces its parent with
    //    a fresh mutant of gbest.  This injects the global best solution as a
    //    new starting point while immediately diversifying through mutation —
    //    not a direct copy, because that would collapse diversity.
    // -----------------------------------------------------------------------
    void evolve_single_phase(Evaluator& evaluator,
                             const PluReader& benchmark,
                             std::mt19937& rng)
    {
        for (int g = 0; g < cfg_.single_phase_gens && !solved_; ++g) {

            for (auto& stream : streams_) {

                // Inject gbest as parent if this stream is behind
                if (stream.parent_fit > gbest_fitness_) {
                    stream.parent     = mutate(gbest_genotype_, rng);
                    stream.parent_fit = evaluator.calculate_fitness(
                                            stream.parent, benchmark);
                }

                run_one_generation(stream, evaluator, benchmark, rng);

                // Update global best
                if (stream.parent_fit < gbest_fitness_) {
                    gbest_fitness_  = stream.parent_fit;
                    gbest_genotype_ = stream.parent;
                }

                if (stream.parent_fit == 0) {
                    solved_.store(true);
                    return;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Stagnation restart: reinitialise the worst stream from scratch
    // -----------------------------------------------------------------------
    void restart_worst_stream(Generator& generator) {
        int worst_idx = 0;
        int worst_fit = streams_[0].sbest_fit;
        for (int k = 1; k < (int)streams_.size(); k++) {
            if (streams_[k].sbest_fit > worst_fit) {
                worst_fit = streams_[k].sbest_fit;
                worst_idx = k;
            }
        }

        std::cout << "  [Restart] Re-initialising stream " << worst_idx + 1
                  << " (sbest=" << worst_fit << ")\n";

        Stream& s   = streams_[worst_idx];
        s.parent    = generator.generate_random_individual();
        s.parent_fit = std::numeric_limits<int>::max();
        s.sbest_fit  = std::numeric_limits<int>::max();
        s.sbest      = s.parent;
    }

public:
    // -----------------------------------------------------------------------
    //  Constructor
    // -----------------------------------------------------------------------
    ParallelStreamsCGP(const Config& cfg,
                       int num_inputs, int num_outputs,
                       int num_nodes, int levels_back, int num_functions)
        : cfg_(cfg),
          num_inputs_(num_inputs), num_outputs_(num_outputs),
          num_nodes_(num_nodes), levels_back_(levels_back),
          num_functions_(num_functions)
    {
        if (cfg_.num_streams < 1)
            throw std::invalid_argument("num_streams must be >= 1");
        if (cfg_.lambda < 1)
            throw std::invalid_argument("lambda must be >= 1");
    }

    // -----------------------------------------------------------------------
    //  Main optimisation loop
    // -----------------------------------------------------------------------
    Genotype optimize(Generator& generator,
                      Evaluator& evaluator,
                      const PluReader& benchmark)
    {
        const int K = cfg_.num_streams;
        std::mt19937 rng{ std::random_device{}() };

        // 1. Initialise streams
        std::cout << "Initialising " << K << " stream(s), lambda=" << cfg_.lambda
                  << " (total offspring/gen: " << K * cfg_.lambda << ")...\n";

        streams_.resize(K);
        for (int k = 0; k < K; k++) {
            streams_[k].id         = k;
            streams_[k].parent     = generator.generate_random_individual();
            streams_[k].parent_fit = std::numeric_limits<int>::max();
            streams_[k].sbest_fit  = std::numeric_limits<int>::max();
            streams_[k].sbest      = streams_[k].parent;
            std::cout << "  Stream " << k + 1 << " ready.\n";
        }

        gbest_fitness_ = std::numeric_limits<int>::max();
        gbest_genotype_ = streams_[0].parent;

        int  total_gens       = 0;
        int  stagnation_count = 0;
        int  prev_gbest       = gbest_fitness_;

        // 2. Outer loop
        for (int cycle = 1; cycle <= cfg_.max_outer_cycles && !solved_; ++cycle) {

            // ── Multi-evolutionary phase ──────────────────────────────
            std::vector<std::thread> threads;
            threads.reserve(K);
            for (int k = 0; k < K; k++) {
                threads.emplace_back([this, k, &evaluator, &benchmark]() {
                    evolve_stream_independent(streams_[k], evaluator, benchmark);
                });
            }
            for (auto& t : threads) t.join();
            total_gens += cfg_.multi_phase_gens;

            if (solved_) break;

            // Collect sbests → update gbest
            for (auto& s : streams_) {
                if (s.sbest_fit < gbest_fitness_) {
                    gbest_fitness_  = s.sbest_fit;
                    gbest_genotype_ = s.sbest;
                }
            }

            // ── Single-evolutionary phase ─────────────────────────────
            std::cout << "[Cycle " << cycle << "] Single phase | gbest: "
                      << gbest_fitness_ << "\n";

            evolve_single_phase(evaluator, benchmark, rng);
            total_gens += cfg_.single_phase_gens;

            // Final gbest update after single phase
            for (auto& s : streams_) {
                if (s.sbest_fit < gbest_fitness_) {
                    gbest_fitness_  = s.sbest_fit;
                    gbest_genotype_ = s.sbest;
                }
            }

            std::cout << "[Cycle " << cycle << "] Finished | Total gens: "
                      << total_gens << " | Global Best Error: "
                      << gbest_fitness_ << "\n";

            if (gbest_fitness_ == 0) break;

            // ── Stagnation detection ──────────────────────────────────
            if (gbest_fitness_ < prev_gbest) {
                stagnation_count = 0;
                prev_gbest       = gbest_fitness_;
            } else {
                ++stagnation_count;
                if (stagnation_count >= cfg_.stagnation_threshold) {
                    restart_worst_stream(generator);
                    stagnation_count = 0;
                }
            }
        }

        // 3. Return best found
        if (solved_) {
            for (auto& s : streams_)
                if (evaluator.calculate_fitness(s.parent, benchmark) == 0)
                    return s.parent;
        }

        return gbest_genotype_;
    }
};

} // namespace LogicCircuitSynthesis