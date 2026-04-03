#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <limits>

// ─────────────────────────────────────────────
//  CGP Boolean Function Set  (Table III in paper)
// ─────────────────────────────────────────────
enum class Func : uint8_t {
    ZERO     = 0,   // 0
    ONE      = 1,   // 1
    A        = 2,   // a
    B        = 3,   // b
    NOT_A    = 4,   // ~a
    NOT_B    = 5,   // ~b
    AND      = 6,   // a & b
    NAND     = 7,   // ~(a & b)
    AND_NB   = 8,   // a & ~b
    AND_NA   = 9,   // ~a & b
    XOR      = 10,  // a ^ b
    XNOR     = 11,  // ~(a ^ b)
    OR       = 12,  // a | b
    NOR      = 13,  // ~(a | b)
    OR_NB    = 14,  // a | ~b
    OR_NA    = 15,  // ~a | b
    MUX_AC   = 16,  // a*c + b*~c   (3-input)
    MUX_NAC  = 17,  // ~a*c + b*~c
    MUX_ACN  = 18,  // a*~c + b*c
    MUX_NACN = 19,  // ~a*~c + b*c
    NUM_FUNCS
};

static constexpr int FUNC_COUNT = static_cast<int>(Func::NUM_FUNCS);

// ─────────────────────────────────────────────
//  Node in the CGP grid
// ─────────────────────────────────────────────
struct CGPNode {
    Func     func   = Func::ZERO;
    int      conn[3] = {0, 0, 0};   // up to 3 inputs (nn=2 typical, 3 for mux)
};

// ─────────────────────────────────────────────
//  CGP Chromosome / Individual
// ─────────────────────────────────────────────
struct CGPChromosome {
    // Grid dimensions (set once, shared)
    int ni = 0;   // primary inputs
    int no = 0;   // primary outputs
    int nr = 0;   // rows
    int nc = 0;   // columns
    int nn = 2;   // inputs per node
    int l  = 0;   // max wire length (l-back)

    std::vector<CGPNode> nodes;      // nr * nc nodes
    std::vector<int>     out_genes;  // no output genes

    // ── Pareto objectives ──
    double area      = std::numeric_limits<double>::max();
    double delay     = std::numeric_limits<double>::max();
    double error     = std::numeric_limits<double>::max();

    // ── NSGA-II bookkeeping ──
    int    rank           = 0;
    double crowding_dist  = 0.0;
    int    dom_count      = 0;          // how many solutions dominate this
    std::vector<int> dominates;         // indices this solution dominates

    // ── hES bookkeeping ──
    int    parent_idx = -1;
};

// ─────────────────────────────────────────────
//  Objectives vector (for generic comparisons)
// ─────────────────────────────────────────────
struct ObjVec {
    double area, delay, error;
    bool operator==(const ObjVec& o) const {
        return area==o.area && delay==o.delay && error==o.error;
    }
    bool operator!=(const ObjVec& o) const { return !(*this==o); }
};

inline ObjVec get_obj(const CGPChromosome& c) {
    return {c.area, c.delay, c.error};
}

// ─────────────────────────────────────────────
//  Constrained Pareto dominance
//  Error is a soft constraint: solutions with lower error
//  always rank before those with higher error.
//  Among equal-error solutions, area and delay trade off.
//  This mirrors the paper's CGP treatment where functional
//  correctness is the primary concern.
// ─────────────────────────────────────────────
inline bool dominates(const ObjVec& a, const ObjVec& b) {
    // Both infeasible: lower error dominates
    if (a.error > 0.0 && b.error > 0.0) {
        if (a.error < b.error - 1e-9) return true;
        if (a.error > b.error + 1e-9) return false;
        // Same error: trade off area and delay
        return (a.area <= b.area && a.delay <= b.delay) &&
               (a.area <  b.area || a.delay <  b.delay);
    }
    // a feasible, b not
    if (a.error == 0.0 && b.error > 0.0) return true;
    // b feasible, a not
    if (a.error > 0.0 && b.error == 0.0) return false;
    // Both feasible: standard 2-obj Pareto on area + delay
    return (a.area <= b.area && a.delay <= b.delay) &&
           (a.area <  b.area || a.delay <  b.delay);
}
inline bool dominates(const CGPChromosome& a, const CGPChromosome& b) {
    return dominates(get_obj(a), get_obj(b));
}
inline bool weak_dominates(const ObjVec& a, const ObjVec& b) {
    return a.area <= b.area && a.delay <= b.delay && a.error <= b.error;
}

// ─────────────────────────────────────────────
//  Experiment / benchmark configuration
// ─────────────────────────────────────────────
enum class BenchmarkType { ADDER_2x2, MULTIPLIER_2x2 };

struct Config {
    BenchmarkType bench     = BenchmarkType::ADDER_2x2;
    int  ni                 = 4;    // primary inputs  (adder: 4, mult: 4)
    int  no                 = 3;    // primary outputs (adder: 3 sum bits, mult: 4)
    int  nr                 = 1;
    int  nc                 = 150;
    int  nn                 = 2;
    int  l                  = 150;  // l-back (full connectivity)
    int  pop_size           = 50;
    int  archive_size       = 100;
    int  lambda_hes         = 4;    // offspring per parent in hES
    long max_evals          = 400000;

    // Periodization scheme: list of (algorithm_id, repeat_count)
    // algorithm_id: 0=NSGAII, 1=SPEA2, 2=hES
    std::vector<std::pair<int,int>> period = {{0,1},{2,4}}; // nh4 default

    int  seed               = 42;
    bool verbose            = true;
    int  report_interval    = 5000;
};
