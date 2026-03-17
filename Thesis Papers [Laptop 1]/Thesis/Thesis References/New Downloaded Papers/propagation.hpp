#pragma once
#include "pso_algorithm.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// =============================================================================
//  propagation.hpp  —  Inter-swarm migration for Parallel PSO
//
//  Four propagation schemes, all sharing the same helper interface:
//
//    get_best_particles(swarm, N_P)     → copies of the N_P fittest particles
//    inject_particles(target, migrants) → replaces the N_P worst slots
//    propagate(swarms, scheme, N_P, rng)→ dispatcher
//
//  NOTE: propagate() is called from inside std::barrier's completion function,
//  which is guaranteed to run on exactly one thread while all others are
//  suspended.  No additional locking is needed here.
// =============================================================================

namespace LogicCircuitSynthesis {

// ─── Propagation scheme selector ─────────────────────────────────────────────
enum class PropagationScheme {
    ONE_TO_ONE,   // 1 random unit  → 1 other random unit
    ONE_TO_N,     // 1 random unit  → all other units
    N_TO_ONE,     // all units      → 1 random unit
    N_TO_N        // all units      → all other units
};

// Returns a human-readable name for logging
inline const char* scheme_name(PropagationScheme s) {
    switch (s) {
        case PropagationScheme::ONE_TO_ONE: return "1-to-1";
        case PropagationScheme::ONE_TO_N:   return "1-to-N";
        case PropagationScheme::N_TO_ONE:   return "N-to-1";
        case PropagationScheme::N_TO_N:     return "N-to-N";
    }
    return "unknown";
}

// =============================================================================
//  Internal helpers
// =============================================================================

// Returns copies of the N_P particles with the lowest pbest_fitness.
// Uses partial_sort so it is O(S * log(N_P)) rather than O(S * log(S)).
inline std::vector<Particle> get_best_particles(const std::vector<Particle>& swarm, int N_P) {
    N_P = std::min(N_P, static_cast<int>(swarm.size()));

    // Build an index vector and partial-sort by pbest_fitness ascending
    std::vector<int> idx(swarm.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + N_P, idx.end(),
        [&swarm](int a, int b) {
            return swarm[a].pbest_fitness < swarm[b].pbest_fitness;
        });

    std::vector<Particle> best;
    best.reserve(N_P);
    for (int k = 0; k < N_P; k++)
        best.push_back(swarm[idx[k]]);

    return best;
}

// Replaces the N_P *worst* slots in 'target' with the migrants.
// If migrants.size() < N_P (shouldn't normally happen) we wrap around.
// The target's pbest and velocity are intentionally preserved from the
// migrant particle so the new occupant brings its search history with it.
inline void inject_particles(std::vector<Particle>& target,
                             const std::vector<Particle>& migrants) {
    if (migrants.empty()) return;

    int N_P = static_cast<int>(migrants.size());

    // Find indices of the N_P worst particles (highest pbest_fitness)
    std::vector<int> idx(target.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + N_P, idx.end(),
        [&target](int a, int b) {
            return target[a].pbest_fitness > target[b].pbest_fitness;
        });

    for (int k = 0; k < N_P; k++)
        target[idx[k]] = migrants[k % migrants.size()];
}

// =============================================================================
//  propagate() — main dispatcher (thread-safe when called from barrier
//  completion; no mutex required there)
// =============================================================================
inline void propagate(std::vector<std::vector<Particle>>& swarms,
                      PropagationScheme                   scheme,
                      int                                 N_P,
                      std::mt19937&                       rng) {
    const int N_I = static_cast<int>(swarms.size());
    if (N_I < 2) return;   // Nothing to migrate with only one unit

    // Helper: pick a random unit index, optionally excluding one value
    auto random_unit = [&](int exclude = -1) -> int {
        std::uniform_int_distribution<int> d(0, N_I - 1);
        int r;
        do { r = d(rng); } while (r == exclude);
        return r;
    };

    switch (scheme) {

        // ── 1 to 1 ─────────────────────────────────────────────────────────
        // A randomly selected source sends its N_P best particles to one
        // other randomly selected destination unit.
        case PropagationScheme::ONE_TO_ONE: {
            int src = random_unit();
            int dst = random_unit(src);

            auto migrants = get_best_particles(swarms[src], N_P);
            inject_particles(swarms[dst], migrants);
            break;
        }

        // ── 1 to N ─────────────────────────────────────────────────────────
        // A randomly selected source broadcasts its N_P best particles to
        // every other unit simultaneously.
        case PropagationScheme::ONE_TO_N: {
            int src = random_unit();
            auto migrants = get_best_particles(swarms[src], N_P);

            for (int j = 0; j < N_I; j++)
                if (j != src)
                    inject_particles(swarms[j], migrants);
            break;
        }

        // ── N to 1 ─────────────────────────────────────────────────────────
        // Every unit donates its N_P best particles to one randomly chosen
        // destination.  All donors are snapshotted before any injection so
        // the destination receives the original elites, not a mix.
        case PropagationScheme::N_TO_ONE: {
            int dst = random_unit();

            // Snapshot elites from every non-destination unit first
            std::vector<std::vector<Particle>> snapshots(N_I);
            for (int j = 0; j < N_I; j++)
                if (j != dst)
                    snapshots[j] = get_best_particles(swarms[j], N_P);

            for (int j = 0; j < N_I; j++)
                if (j != dst)
                    inject_particles(swarms[dst], snapshots[j]);
            break;
        }

        // ── N to N ─────────────────────────────────────────────────────────
        // Every unit sends its N_P best particles to every other unit.
        // All elites are snapshotted before any injection so each unit
        // receives the original elites from its peers, never a blend of
        // migrants that were themselves recently replaced.
        case PropagationScheme::N_TO_N: {
            // Phase 1: snapshot all elites
            std::vector<std::vector<Particle>> snapshots(N_I);
            for (int j = 0; j < N_I; j++)
                snapshots[j] = get_best_particles(swarms[j], N_P);

            // Phase 2: inject
            for (int dst = 0; dst < N_I; dst++)
                for (int src = 0; src < N_I; src++)
                    if (src != dst)
                        inject_particles(swarms[dst], snapshots[src]);
            break;
        }
    }
}

} // namespace LogicCircuitSynthesis
