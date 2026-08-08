#pragma once
#include "cgp_types.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>

// ─────────────────────────────────────────────
//  Evaluate a single Boolean function node
// ─────────────────────────────────────────────
inline bool eval_func(Func f, bool a, bool b, bool c) {
    switch (f) {
        case Func::ZERO:     return false;
        case Func::ONE:      return true;
        case Func::A:        return a;
        case Func::B:        return b;
        case Func::NOT_A:    return !a;
        case Func::NOT_B:    return !b;
        case Func::AND:      return a & b;
        case Func::NAND:     return !(a & b);
        case Func::AND_NB:   return a & !b;
        case Func::AND_NA:   return !a & b;
        case Func::XOR:      return a ^ b;
        case Func::XNOR:     return !(a ^ b);
        case Func::OR:       return a | b;
        case Func::NOR:      return !(a | b);
        case Func::OR_NB:    return a | !b;
        case Func::OR_NA:    return !a | b;
        case Func::MUX_AC:   return (a & c)  | (b & !c);
        case Func::MUX_NAC:  return (!a & c) | (b & !c);
        case Func::MUX_ACN:  return (a & !c) | (b & c);
        case Func::MUX_NACN: return (!a & !c)| (b & c);
        default:             return false;
    }
}

// ─────────────────────────────────────────────
//  Topological signal propagation through CGP grid
//  Returns vector of output bits for given input word
// ─────────────────────────────────────────────
inline std::vector<bool> simulate(const CGPChromosome& ch, uint32_t input_word) {
    int total_nodes = ch.ni + ch.nr * ch.nc;
    std::vector<bool> sig(total_nodes, false);

    // Primary inputs
    for (int i = 0; i < ch.ni; i++)
        sig[i] = (input_word >> i) & 1u;

    // Propagate through columns left→right
    for (int col = 0; col < ch.nc; col++) {
        for (int row = 0; row < ch.nr; row++) {
            int idx = ch.ni + col * ch.nr + row;
            const CGPNode& nd = ch.nodes[col * ch.nr + row];
            bool a = sig[nd.conn[0]];
            bool b = sig[nd.conn[1]];
            bool c = (ch.nn >= 3) ? sig[nd.conn[2]] : false;
            sig[idx] = eval_func(nd.func, a, b, c);
        }
    }

    std::vector<bool> out(ch.no);
    for (int i = 0; i < ch.no; i++)
        out[i] = sig[ch.out_genes[i]];
    return out;
}

// ─────────────────────────────────────────────
//  Truth table helpers for 2x2 adder / multiplier
//  Adder:  inputs a1,a0,b1,b0  →  s2,s1,s0  (3-bit sum)
//  Mult:   inputs a1,a0,b1,b0  →  p3,p2,p1,p0 (4-bit product)
// ─────────────────────────────────────────────
inline uint32_t adder_expected(uint32_t in4) {
    uint32_t a = in4 & 3u;          // a1 a0
    uint32_t b = (in4 >> 2) & 3u;  // b1 b0
    return (a + b) & 7u;            // 3-bit sum
}

inline uint32_t mult_expected(uint32_t in4) {
    uint32_t a = in4 & 3u;
    uint32_t b = (in4 >> 2) & 3u;
    return (a * b) & 15u;           // 4-bit product
}

// ─────────────────────────────────────────────
//  Count active nodes (proxy for area)
// ─────────────────────────────────────────────
inline int count_active_nodes(const CGPChromosome& ch) {
    int total = ch.ni + ch.nr * ch.nc;
    std::vector<bool> active(total, false);

    // Mark outputs as active
    for (int g : ch.out_genes) active[g] = true;

    // Back-propagate from outputs (right to left)
    for (int col = ch.nc - 1; col >= 0; col--) {
        for (int row = 0; row < ch.nr; row++) {
            int idx = ch.ni + col * ch.nr + row;
            if (!active[idx]) continue;
            const CGPNode& nd = ch.nodes[col * ch.nr + row];
            for (int k = 0; k < ch.nn; k++)
                if (nd.conn[k] >= 0 && nd.conn[k] < total)
                    active[nd.conn[k]] = true;
        }
    }

    int cnt = 0;
    for (int i = ch.ni; i < total; i++) cnt += active[i];
    return cnt;
}

// ─────────────────────────────────────────────
//  Compute critical path delay (number of levels)
// ─────────────────────────────────────────────
inline int compute_delay(const CGPChromosome& ch) {
    int total = ch.ni + ch.nr * ch.nc;
    std::vector<int> depth(total, 0);

    for (int col = 0; col < ch.nc; col++) {
        for (int row = 0; row < ch.nr; row++) {
            int idx = ch.ni + col * ch.nr + row;
            const CGPNode& nd = ch.nodes[col * ch.nr + row];
            int max_d = 0;
            for (int k = 0; k < ch.nn; k++)
                if (nd.conn[k] >= 0 && nd.conn[k] < total)
                    max_d = std::max(max_d, depth[nd.conn[k]]);
            depth[idx] = max_d + 1;
        }
    }

    int max_delay = 0;
    for (int g : ch.out_genes)
        if (g < (int)depth.size())
            max_delay = std::max(max_delay, depth[g]);
    return max_delay;
}

// ─────────────────────────────────────────────
//  Full fitness evaluation — fills area/delay/error
// ─────────────────────────────────────────────
inline void evaluate(CGPChromosome& ch, BenchmarkType bench) {
    int num_inputs = ch.ni;
    int cases = 1 << num_inputs;
    int errors = 0;

    for (int in = 0; in < cases; in++) {
        auto out = simulate(ch, (uint32_t)in);

        uint32_t got = 0;
        for (int b = 0; b < ch.no; b++)
            got |= ((uint32_t)out[b]) << b;

        uint32_t expected = (bench == BenchmarkType::ADDER_2x2)
            ? adder_expected((uint32_t)in)
            : mult_expected((uint32_t)in);

        // Count wrong bits
        uint32_t diff = got ^ expected;
        errors += __builtin_popcount(diff);
    }

    ch.error = (double)errors / (cases * ch.no);   // normalised bit error rate
    ch.area  = (double)count_active_nodes(ch);
    ch.delay = (double)compute_delay(ch);
}
