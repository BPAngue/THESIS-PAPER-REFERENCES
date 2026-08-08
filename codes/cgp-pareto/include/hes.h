#pragma once
#include "cgp_types.h"
#include "cgp_ops.h"
#include "cgp_eval.h"
#include "nsgaii.h"
#include <vector>
#include <algorithm>
#include <map>
#include <limits>

// ─────────────────────────────────────────────
//  Algorithm 2: ES-generate
//  Generate λ mutated offspring from parent p
// ─────────────────────────────────────────────
inline std::vector<CGPChromosome>
es_generate(const CGPChromosome& p, int lambda, const Config& cfg) {
    std::vector<CGPChromosome> Q;
    Q.reserve(lambda);
    for (int i = 0; i < lambda; i++) {
        CGPChromosome child = mutate(p, cfg, 0.1, 0.1);
        evaluate(child, cfg.bench);
        Q.push_back(child);
    }
    return Q;
}

// ─────────────────────────────────────────────
//  Algorithm 3: add-replace
//  Clone P, add offspring that have unique Pareto vectors;
//  if an offspring matches its parent's Pareto vector, replace parent
// ─────────────────────────────────────────────
inline std::vector<CGPChromosome>
add_replace(const std::vector<CGPChromosome>& P,
            const std::vector<CGPChromosome>& Q_all,
            // Q_all contains offspring with parent_idx set
            const std::vector<int>& parent_map) {
    // parent_map[offspring_index] = index in P
    std::vector<CGPChromosome> R = P;
    std::vector<bool> replaced(P.size(), false);

    int qi = 0;
    for (const auto& q : Q_all) {
        int pi = parent_map[qi++];
        ObjVec qv = get_obj(q);
        ObjVec pv = get_obj(P[pi]);

        bool q_equals_parent = (qv == pv);
        bool q_better_error  = (qv.error < pv.error - 1e-9);

        if ((q_equals_parent || q_better_error) && !replaced[pi]) {
            // Replace parent: genetic drift or genuine improvement
            R[pi] = q;
            replaced[pi] = true;
        } else {
            // Check if q is already represented in R
            bool dup = false;
            for (const auto& r : R) {
                ObjVec rv = get_obj(r);
                if (rv == qv) { dup = true; break; }
            }
            if (!dup) R.push_back(q);
        }
    }
    return R;
}

// ─────────────────────────────────────────────
//  Algorithm 1: hES-step
//  Perform one hES generation on archive Pt
// ─────────────────────────────────────────────
inline std::vector<CGPChromosome>
hes_step(std::vector<CGPChromosome>& Pt, const Config& cfg) {
    int n = (int)Pt.size();
    int lambda = cfg.lambda_hes;

    // ── Step 1-4: Generate offspring for every parent ──
    std::vector<CGPChromosome> Qt;
    std::vector<int>           parent_map;
    Qt.reserve(n * lambda);
    parent_map.reserve(n * lambda);

    for (int pi = 0; pi < n; pi++) {
        auto offspring = es_generate(Pt[pi], lambda, cfg);
        for (auto& o : offspring) {
            o.parent_idx = pi;
            Qt.push_back(o);
            parent_map.push_back(pi);
        }
    }

    // ── Step 5: add-replace ──
    std::vector<CGPChromosome> Rt = add_replace(Pt, Qt, parent_map);

    // ── Step 6: fast-non-dominated-sort(Rt) ──
    auto fronts = fast_non_dominated_sort(Rt);

    // ── Step 7-22: For each non-dominated front, pick one survivor per parent ──
    std::vector<CGPChromosome> Pt1;
    Pt1.reserve(n);

    std::vector<bool> parent_replaced(n, false);

    for (auto& Fi : fronts) {
        if ((int)Pt1.size() >= n) break;

        // Assign crowding distances within this front
        crowding_distance_assignment(Rt, Fi);

        // Group by parent
        // For each member of Fi, determine which parent it came from
        // Members of Pt (indices 0..n-1 in Rt) keep their parent_idx = their own index
        // Members of Qt offspring have parent_idx set
        std::map<int, std::vector<int>> groups; // parent_idx → indices in Fi

        for (int idx : Fi) {
            int par;
            if (idx < n) {
                // Original parent
                par = idx;
            } else {
                // Offspring
                par = Rt[idx].parent_idx;
            }
            if (par >= 0 && par < n)
                groups[par].push_back(idx);
        }

        for (auto& [par, members] : groups) {
            if (parent_replaced[par]) continue;
            if ((int)Pt1.size() >= n) break;

            // Check if the parent itself is in this group
            bool parent_present = false;
            for (int idx : members) {
                if (idx == par) { parent_present = true; break; }
            }

            if (parent_present) {
                Pt1.push_back(Rt[par]);
            } else {
                // Select least crowded member of this group
                int best = members[0];
                for (int idx : members)
                    if (Rt[idx].crowding_dist > Rt[best].crowding_dist)
                        best = idx;
                Pt1.push_back(Rt[best]);
            }
            parent_replaced[par] = true;
        }
    }

    // If not enough survivors (edge case), fill from Rt sorted by dominance
    if ((int)Pt1.size() < n) {
        // Sort Rt by rank then crowding
        std::vector<int> all_idx(Rt.size());
        std::iota(all_idx.begin(), all_idx.end(), 0);
        std::sort(all_idx.begin(), all_idx.end(), [&](int a, int b) {
            return crowded_less(Rt[a], Rt[b]);
        });
        for (int idx : all_idx) {
            if ((int)Pt1.size() >= n) break;
            // Avoid duplicates
            ObjVec ov = get_obj(Rt[idx]);
            bool dup = false;
            for (auto& c : Pt1)
                if (get_obj(c) == ov) { dup = true; break; }
            if (!dup) Pt1.push_back(Rt[idx]);
        }
    }

    // Trim to n if somehow larger
    if ((int)Pt1.size() > n) Pt1.resize(n);

    return Pt1;
}
