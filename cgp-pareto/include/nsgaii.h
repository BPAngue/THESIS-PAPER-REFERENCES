#pragma once
#include "cgp_types.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

// ─────────────────────────────────────────────
//  Fast Non-dominated Sort  (Deb et al. 2000)
//  Returns list of fronts; each front is a list
//  of indices into `pop`.
// ─────────────────────────────────────────────
inline std::vector<std::vector<int>>
fast_non_dominated_sort(std::vector<CGPChromosome>& pop) {
    int n = (int)pop.size();
    for (auto& c : pop) {
        c.dom_count = 0;
        c.dominates.clear();
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            if (::dominates(pop[i], pop[j]))
                pop[i].dominates.push_back(j);
            else if (::dominates(pop[j], pop[i]))
                pop[i].dom_count++;
        }
    }

    std::vector<std::vector<int>> fronts;
    std::vector<int> front0;
    for (int i = 0; i < n; i++) {
        pop[i].rank = 0;
        if (pop[i].dom_count == 0) front0.push_back(i);
    }
    fronts.push_back(front0);

    int fi = 0;
    while (!fronts[fi].empty()) {
        std::vector<int> next;
        for (int i : fronts[fi]) {
            for (int j : pop[i].dominates) {
                pop[j].dom_count--;
                if (pop[j].dom_count == 0) {
                    pop[j].rank = fi + 1;
                    next.push_back(j);
                }
            }
        }
        fronts.push_back(next);
        fi++;
    }
    if (fronts.back().empty()) fronts.pop_back();
    return fronts;
}

// ─────────────────────────────────────────────
//  Crowding Distance Assignment for one front
// ─────────────────────────────────────────────
inline void crowding_distance_assignment(std::vector<CGPChromosome>& pop,
                                          const std::vector<int>& front) {
    int l = (int)front.size();
    for (int i : front) pop[i].crowding_dist = 0.0;
    if (l <= 2) {
        for (int i : front) pop[i].crowding_dist = std::numeric_limits<double>::infinity();
        return;
    }

    // Three objectives: error (primary), area, delay
    auto get_obj_val = [&](int idx, int obj) -> double {
        if (obj == 0) return pop[idx].error;
        if (obj == 1) return pop[idx].area;
        return pop[idx].delay;
    };

    for (int obj = 0; obj < 3; obj++) {
        // Sort by this objective
        std::vector<int> sorted = front;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
            return get_obj_val(a, obj) < get_obj_val(b, obj);
        });

        double f_min = get_obj_val(sorted.front(), obj);
        double f_max = get_obj_val(sorted.back(),  obj);
        double range = f_max - f_min;

        // Boundary points get infinity
        pop[sorted.front()].crowding_dist = std::numeric_limits<double>::infinity();
        pop[sorted.back() ].crowding_dist = std::numeric_limits<double>::infinity();

        if (range < 1e-12) continue;   // all identical on this objective

        for (int i = 1; i < l - 1; i++) {
            pop[sorted[i]].crowding_dist +=
                (get_obj_val(sorted[i+1], obj) - get_obj_val(sorted[i-1], obj)) / range;
        }
    }
}

// ─────────────────────────────────────────────
//  Crowded comparison operator  ≺n
//  Prefer lower rank; break ties by higher crowding distance
// ─────────────────────────────────────────────
inline bool crowded_less(const CGPChromosome& a, const CGPChromosome& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowding_dist > b.crowding_dist;
}

// ─────────────────────────────────────────────
//  Binary tournament selection
// ─────────────────────────────────────────────
inline int tournament_select(const std::vector<CGPChromosome>& pop) {
    int a = rand_int(0, (int)pop.size()-1);
    int b = rand_int(0, (int)pop.size()-1);
    return crowded_less(pop[a], pop[b]) ? a : b;
}

// ─────────────────────────────────────────────
//  NSGA-II: one generation
//  Returns new population of size pop_size
// ─────────────────────────────────────────────
inline std::vector<CGPChromosome>
nsgaii_step(std::vector<CGPChromosome>& pop, const Config& cfg) {
    int n = (int)pop.size();

    // ── Sort current population ──
    auto fronts = fast_non_dominated_sort(pop);
    for (auto& f : fronts) crowding_distance_assignment(pop, f);

    // ── Generate offspring ──
    // For CGP: mostly mutation, crossover with lower probability
    std::vector<CGPChromosome> offspring;
    offspring.reserve(n);
    while ((int)offspring.size() < n) {
        int p1 = tournament_select(pop);
        CGPChromosome child;
        if (rand_double() < 0.5) {
            // crossover + mutation
            int p2 = tournament_select(pop);
            child = crossover(pop[p1], pop[p2], cfg, 0.5);
            child = mutate(child, cfg, 0.1, 0.1);
        } else {
            // mutation only (important for CGP)
            child = mutate(pop[p1], cfg, 0.05, 0.1);
        }
        evaluate(child, cfg.bench);
        offspring.push_back(child);
    }

    // ── Combine parent + offspring ──
    std::vector<CGPChromosome> combined = pop;
    combined.insert(combined.end(), offspring.begin(), offspring.end());

    // ── Re-sort combined pool ──
    auto all_fronts = fast_non_dominated_sort(combined);
    for (auto& f : all_fronts) crowding_distance_assignment(combined, f);

    // ── Select top n individuals ──
    std::vector<CGPChromosome> next;
    next.reserve(n);
    for (auto& f : all_fronts) {
        if ((int)(next.size() + f.size()) <= n) {
            for (int i : f) next.push_back(combined[i]);
        } else {
            // Sort this front by crowding and fill remaining slots
            std::sort(f.begin(), f.end(), [&](int a, int b) {
                return combined[a].crowding_dist > combined[b].crowding_dist;
            });
            int need = n - (int)next.size();
            for (int k = 0; k < need; k++) next.push_back(combined[f[k]]);
            break;
        }
    }
    return next;
}

// ─────────────────────────────────────────────
//  SPEA2: fitness assignment + archive update
//  Simplified: uses raw strength + density
// ─────────────────────────────────────────────
inline std::vector<double>
spea2_fitness(const std::vector<CGPChromosome>& pop,
              const std::vector<CGPChromosome>& archive) {
    std::vector<CGPChromosome*> all;
    all.reserve(pop.size() + archive.size());
    // combine
    std::vector<CGPChromosome> combined = pop;
    combined.insert(combined.end(), archive.begin(), archive.end());

    int n = (int)combined.size();
    std::vector<int> strength(n, 0);
    // Strength = number of solutions this dominates
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j && ::dominates(combined[i], combined[j]))
                strength[i]++;

    // Raw fitness = sum of strengths of dominators
    std::vector<double> raw(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j != i && ::dominates(combined[j], combined[i]))
                raw[i] += strength[j];

    // Density = 1 / (k-th nearest neighbour distance + 2)
    // k = sqrt(n)
    int k = (int)std::sqrt((double)n);
    for (int i = 0; i < n; i++) {
        std::vector<double> dists;
        dists.reserve(n);
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double da = combined[i].area  - combined[j].area;
            double dd = combined[i].delay - combined[j].delay;
            double de = combined[i].error - combined[j].error;
            dists.push_back(std::sqrt(da*da + dd*dd + de*de));
        }
        std::sort(dists.begin(), dists.end());
        double sigma_k = (k < (int)dists.size()) ? dists[k] : dists.back();
        raw[i] += 1.0 / (sigma_k + 2.0);
    }
    return raw; // lower is better
}

inline std::vector<CGPChromosome>
spea2_step(std::vector<CGPChromosome>& pop,
           std::vector<CGPChromosome>& archive,
           const Config& cfg) {
    int pop_n  = (int)pop.size();
    int arch_n = cfg.archive_size;

    // Combine pop + archive
    std::vector<CGPChromosome> combined = pop;
    combined.insert(combined.end(), archive.begin(), archive.end());
    int n = (int)combined.size();

    auto fit = spea2_fitness(pop, archive);

    // New archive: non-dominated individuals
    std::vector<int> nd_idx;
    for (int i = 0; i < n; i++)
        if (fit[i] < 1.0) nd_idx.push_back(i);

    // Truncate or fill archive
    while ((int)nd_idx.size() > arch_n) {
        // Remove individual with smallest distance to nearest neighbour
        double worst_dist = std::numeric_limits<double>::max();
        int    worst_i    = 0;
        for (int ii = 0; ii < (int)nd_idx.size(); ii++) {
            int i = nd_idx[ii];
            double min_d = std::numeric_limits<double>::max();
            for (int jj = 0; jj < (int)nd_idx.size(); jj++) {
                if (ii == jj) continue;
                int j = nd_idx[jj];
                double da = combined[i].area  - combined[j].area;
                double dd = combined[i].delay - combined[j].delay;
                double de = combined[i].error - combined[j].error;
                min_d = std::min(min_d, std::sqrt(da*da + dd*dd + de*de));
            }
            if (min_d < worst_dist) { worst_dist = min_d; worst_i = ii; }
        }
        nd_idx.erase(nd_idx.begin() + worst_i);
    }

    if ((int)nd_idx.size() < arch_n) {
        // Fill with best dominated individuals by fitness
        std::vector<std::pair<double,int>> dom_sorted;
        for (int i = 0; i < n; i++)
            if (fit[i] >= 1.0) dom_sorted.push_back({fit[i], i});
        std::sort(dom_sorted.begin(), dom_sorted.end());
        for (auto& [f, i] : dom_sorted) {
            if ((int)nd_idx.size() >= arch_n) break;
            nd_idx.push_back(i);
        }
    }

    std::vector<CGPChromosome> new_archive;
    new_archive.reserve(nd_idx.size());
    for (int i : nd_idx) new_archive.push_back(combined[i]);
    archive = new_archive;

    // Generate new population using binary tournament on fitness
    std::vector<CGPChromosome> new_pop;
    new_pop.reserve(pop_n);
    int na = (int)archive.size();
    while ((int)new_pop.size() < pop_n) {
        int p1 = nd_idx[rand_int(0, na-1)];
        int p2 = nd_idx[rand_int(0, na-1)];
        int chosen = (fit[p1] < fit[p2]) ? p1 : p2;
        CGPChromosome child = mutate(combined[chosen], cfg, 0.1, 0.1);
        evaluate(child, cfg.bench);
        new_pop.push_back(child);
    }
    return new_pop;
}
