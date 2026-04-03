#pragma once
#include "cgp_types.h"
#include "cgp_ops.h"
#include "cgp_eval.h"
#include "nsgaii.h"
#include "hes.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>

// ─────────────────────────────────────────────
//  Pretty print the current Pareto front
// ─────────────────────────────────────────────
inline std::vector<CGPChromosome>
extract_pareto_front(const std::vector<CGPChromosome>& pop) {
    std::vector<CGPChromosome> front;
    for (auto& c : pop) {
        bool dominated = false;
        for (auto& d : pop) {
            if (&c == &d) continue;
            if (::dominates(d, c)) { dominated = true; break; }
        }
        if (!dominated) front.push_back(c);
    }
    return front;
}

// ─────────────────────────────────────────────
//  Statistics for a Pareto front
// ─────────────────────────────────────────────
struct FrontStats {
    int    size        = 0;
    double best_error  = 1.0;
    double best_area   = 0.0;
    double best_delay  = 0.0;
    int    correct     = 0;    // # solutions with error == 0
};

inline FrontStats compute_stats(const std::vector<CGPChromosome>& front) {
    FrontStats s;
    s.size = (int)front.size();
    s.best_error = std::numeric_limits<double>::max();
    s.best_area  = std::numeric_limits<double>::max();
    s.best_delay = std::numeric_limits<double>::max();
    for (auto& c : front) {
        s.best_error = std::min(s.best_error, c.error);
        s.best_area  = std::min(s.best_area,  c.area);
        s.best_delay = std::min(s.best_delay, c.delay);
        if (c.error == 0.0) s.correct++;
    }
    return s;
}

// ─────────────────────────────────────────────
//  Algorithm name helper
// ─────────────────────────────────────────────
inline std::string alg_name(int id) {
    if (id == 0) return "NSGAII";
    if (id == 1) return "SPEA2";
    return "hES";
}

inline std::string scheme_name(const Config& cfg) {
    std::ostringstream ss;
    for (auto& [id, rep] : cfg.period) {
        ss << alg_name(id);
        if (rep > 1) ss << rep;
    }
    return ss.str();
}

// ─────────────────────────────────────────────
//  Run one generation of the specified algorithm
//  Returns updated population; evals counter updated
// ─────────────────────────────────────────────
inline std::vector<CGPChromosome>
run_one_step(int alg_id,
             std::vector<CGPChromosome>& pop,
             std::vector<CGPChromosome>& spea2_archive,
             const Config& cfg,
             long& evals) {
    if (alg_id == 0) {
        // NSGAII
        auto new_pop = nsgaii_step(pop, cfg);
        evals += (long)pop.size();
        return new_pop;
    } else if (alg_id == 1) {
        // SPEA2
        auto new_pop = spea2_step(pop, spea2_archive, cfg);
        evals += (long)pop.size();
        return new_pop;
    } else {
        // hES
        auto new_pop = hes_step(pop, cfg);
        evals += (long)pop.size() * cfg.lambda_hes;
        return new_pop;
    }
}

// ─────────────────────────────────────────────
//  Full periodized run
// ─────────────────────────────────────────────
struct RunResult {
    std::vector<CGPChromosome> final_pareto;
    FrontStats                  stats;
    long                        total_evals = 0;
    int                         generations = 0;
};

inline RunResult run_periodized(const Config& cfg) {
    seed_rng(cfg.seed);

    // ── Initialise population ──
    std::vector<CGPChromosome> pop;
    pop.reserve(cfg.pop_size);
    for (int i = 0; i < cfg.pop_size; i++) {
        auto ch = make_random(cfg);
        evaluate(ch, cfg.bench);
        pop.push_back(ch);
    }
    std::vector<CGPChromosome> spea2_archive;

    long evals = (long)cfg.pop_size;
    int  gen   = 0;
    long next_report = cfg.report_interval;

    if (cfg.verbose) {
        std::cout << "═══════════════════════════════════════════════════════\n";
        std::cout << "  CGP-hES  |  Scheme: " << scheme_name(cfg)
                  << "  |  Benchmark: "
                  << (cfg.bench == BenchmarkType::ADDER_2x2 ? "2x2 Adder" : "2x2 Multiplier")
                  << "\n";
        std::cout << "  Max evals: " << cfg.max_evals << "\n";
        std::cout << "═══════════════════════════════════════════════════════\n";
        std::cout << std::left
                  << std::setw(10) << "Evals"
                  << std::setw(8)  << "Gen"
                  << std::setw(12) << "FrontSize"
                  << std::setw(12) << "BestError"
                  << std::setw(10) << "BestArea"
                  << std::setw(10) << "BestDelay"
                  << std::setw(10) << "Correct\n";
        std::cout << std::string(72, '-') << "\n";
    }

    // ── Main loop ──
    while (evals < cfg.max_evals) {
        // Iterate over the period sequence
        for (auto& [alg_id, reps] : cfg.period) {
            if (evals >= cfg.max_evals) break;
            for (int r = 0; r < reps && evals < cfg.max_evals; r++) {
                pop = run_one_step(alg_id, pop, spea2_archive, cfg, evals);
                gen++;
            }
        }

        // Periodic reporting
        if (cfg.verbose && evals >= next_report) {
            auto front = extract_pareto_front(pop);
            auto st    = compute_stats(front);
            std::cout << std::left
                      << std::setw(10) << evals
                      << std::setw(8)  << gen
                      << std::setw(12) << st.size
                      << std::setw(12) << std::fixed << std::setprecision(4) << st.best_error
                      << std::setw(10) << (int)st.best_area
                      << std::setw(10) << (int)st.best_delay
                      << std::setw(10) << st.correct << "\n";
            next_report += cfg.report_interval;
        }
    }

    auto front = extract_pareto_front(pop);
    auto stats = compute_stats(front);

    if (cfg.verbose) {
        std::cout << std::string(72, '=') << "\n";
        std::cout << "  DONE  |  Evals: " << evals
                  << "  |  Pareto front size: " << stats.size
                  << "  |  Correct circuits: " << stats.correct << "\n\n";

        if (stats.correct > 0) {
            std::cout << "  ✓  Correct solutions on the Pareto front:\n";
            std::cout << "     " << std::setw(10) << "Area"
                      << std::setw(10) << "Delay" << "\n";
            for (auto& c : front) {
                if (c.error == 0.0)
                    std::cout << "     " << std::setw(10) << (int)c.area
                              << std::setw(10) << (int)c.delay << "\n";
            }
        }
        std::cout << "\n";
    }

    RunResult res;
    res.final_pareto = front;
    res.stats        = stats;
    res.total_evals  = evals;
    res.generations  = gen;
    return res;
}

// ─────────────────────────────────────────────
//  Verify a correct circuit by printing truth table
// ─────────────────────────────────────────────
inline void print_truth_table(const CGPChromosome& ch, BenchmarkType bench) {
    int cases = 1 << ch.ni;
    std::string sep(50, '-');
    std::cout << sep << "\n";
    std::cout << "Truth Table Verification\n" << sep << "\n";

    if (bench == BenchmarkType::ADDER_2x2) {
        std::cout << "  A(1:0)  B(1:0)  |  Got S(2:0)  |  Exp S(2:0)  |  OK?\n";
    } else {
        std::cout << "  A(1:0)  B(1:0)  |  Got P(3:0)  |  Exp P(3:0)  |  OK?\n";
    }
    std::cout << sep << "\n";

    int errors = 0;
    for (int in = 0; in < cases; in++) {
        auto out = simulate(ch, (uint32_t)in);
        uint32_t got = 0;
        for (int b = 0; b < ch.no; b++) got |= ((uint32_t)out[b]) << b;
        uint32_t exp = (bench == BenchmarkType::ADDER_2x2)
            ? adder_expected((uint32_t)in)
            : mult_expected ((uint32_t)in);

        int a = in & 3, b_ = (in >> 2) & 3;
        bool ok = (got == exp);
        if (!ok) errors++;
        std::cout << "    " << a << "        " << b_
                  << "    |     " << got
                  << "         |     " << exp
                  << "         |  " << (ok ? "✓" : "✗") << "\n";
    }
    std::cout << sep << "\n";
    std::cout << "  Errors: " << errors << " / " << (cases * ch.no) << " bits\n\n";
}

// ─────────────────────────────────────────────
//  Print CGP active graph summary
// ─────────────────────────────────────────────
inline void print_circuit_info(const CGPChromosome& ch) {
    std::cout << "  Circuit info:\n";
    std::cout << "    Active nodes : " << (int)ch.area   << "\n";
    std::cout << "    Critical path: " << (int)ch.delay  << " levels\n";
    std::cout << "    Error rate   : " << ch.error       << "\n\n";
}
