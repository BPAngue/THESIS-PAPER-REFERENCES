#pragma once
#include "cgp_types.h"
#include "cgp_eval.h"
#include <random>
#include <algorithm>

// ─────────────────────────────────────────────
//  Random number generator (seeded per run)
// ─────────────────────────────────────────────
inline std::mt19937& rng() {
    static std::mt19937 gen(42);
    return gen;
}
inline void seed_rng(int s) { rng().seed((uint32_t)s); }

inline int rand_int(int lo, int hi_inclusive) {
    std::uniform_int_distribution<int> d(lo, hi_inclusive);
    return d(rng());
}
inline double rand_double() {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng());
}

// ─────────────────────────────────────────────
//  Valid connection range for a node at column col
//  l-back: can connect to primary inputs OR nodes
//  in columns [max(0, col-l) .. col-1]
// ─────────────────────────────────────────────
inline int max_conn(const Config& cfg, int col) {
    // indices 0..ni-1 are primary inputs
    // nodes at column c → index ni + c*nr + row
    // we allow connections to any node in earlier columns (l-back)
    int earliest_col = std::max(0, col - cfg.l);
    // last valid index = ni + col*nr - 1  (one before this column)
    return cfg.ni + col * cfg.nr - 1;   // = ni + (col*nr) - 1
    (void)earliest_col;
}

// Random connection for a node in the given column
inline int random_conn(const Config& cfg, int col) {
    int hi = cfg.ni + col * cfg.nr - 1;
    if (hi < 0) hi = cfg.ni - 1;   // first column: only primary inputs
    int lo = 0;
    // Apply l-back restriction
    int l_start_col = std::max(0, col - cfg.l);
    if (l_start_col > 0)
        lo = cfg.ni + (l_start_col) * cfg.nr;
    if (lo > hi) lo = 0;
    return rand_int(lo, hi);
}

// ─────────────────────────────────────────────
//  Create a random chromosome
// ─────────────────────────────────────────────
inline CGPChromosome make_random(const Config& cfg) {
    CGPChromosome ch;
    ch.ni = cfg.ni; ch.no = cfg.no;
    ch.nr = cfg.nr; ch.nc = cfg.nc;
    ch.nn = cfg.nn; ch.l  = cfg.l;

    ch.nodes.resize(cfg.nr * cfg.nc);
    for (int col = 0; col < cfg.nc; col++) {
        for (int row = 0; row < cfg.nr; row++) {
            CGPNode& nd = ch.nodes[col * cfg.nr + row];
            nd.func = static_cast<Func>(rand_int(0, FUNC_COUNT - 1));
            for (int k = 0; k < cfg.nn; k++)
                nd.conn[k] = random_conn(cfg, col);
            nd.conn[2] = (cfg.nn >= 3) ? random_conn(cfg, col) : 0;
        }
    }

    int total = cfg.ni + cfg.nr * cfg.nc;
    ch.out_genes.resize(cfg.no);
    for (int i = 0; i < cfg.no; i++)
        ch.out_genes[i] = rand_int(0, total - 1);

    return ch;
}

// ─────────────────────────────────────────────
//  Point mutation  (paper: mutation_prob = 0.1)
// ─────────────────────────────────────────────
inline CGPChromosome mutate(const CGPChromosome& src, const Config& cfg,
                             double node_mut_prob = 0.1,
                             double out_mut_prob  = 0.1) {
    CGPChromosome ch = src;

    for (int col = 0; col < cfg.nc; col++) {
        for (int row = 0; row < cfg.nr; row++) {
            CGPNode& nd = ch.nodes[col * cfg.nr + row];
            if (rand_double() < node_mut_prob) {
                nd.func = static_cast<Func>(rand_int(0, FUNC_COUNT - 1));
            }
            for (int k = 0; k < cfg.nn; k++) {
                if (rand_double() < node_mut_prob)
                    nd.conn[k] = random_conn(cfg, col);
            }
        }
    }

    for (int i = 0; i < cfg.no; i++) {
        if (rand_double() < out_mut_prob) {
            int total = cfg.ni + cfg.nr * cfg.nc;
            ch.out_genes[i] = rand_int(0, total - 1);
        }
    }

    // Reset fitness
    ch.area = ch.delay = ch.error = std::numeric_limits<double>::max();
    ch.rank = 0; ch.crowding_dist = 0;
    return ch;
}

// ─────────────────────────────────────────────
//  SBX-like crossover adapted for integer genes
//  (used by NSGAII / SPEA2 on continuous benchmarks;
//   for CGP we use uniform crossover on node genes)
// ─────────────────────────────────────────────
inline CGPChromosome crossover(const CGPChromosome& p1,
                                const CGPChromosome& p2,
                                const Config& cfg,
                                double prob = 0.5) {
    CGPChromosome ch = p1;
    for (int i = 0; i < cfg.nr * cfg.nc; i++) {
        if (rand_double() < prob) {
            ch.nodes[i] = p2.nodes[i];
        }
    }
    for (int i = 0; i < cfg.no; i++) {
        if (rand_double() < prob)
            ch.out_genes[i] = p2.out_genes[i];
    }
    ch.area = ch.delay = ch.error = std::numeric_limits<double>::max();
    return ch;
}
