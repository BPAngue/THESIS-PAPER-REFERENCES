// =============================================================================
//  main.cpp — Parallel Island-Model PSO for Logic Circuit Synthesis
//
//  Compile (C++20 required for std::barrier):
//    g++ -std=c++20 -O2 -pthread -o pso_parallel main.cpp
//
//  Algorithm overview
//  ──────────────────
//  N_I independent PSO "islands" run concurrently on separate threads.
//  Every N_R iterations all islands pause at a std::barrier.  The barrier's
//  completion function (which runs on exactly one thread while all others
//  are suspended) performs migration via the chosen PropagationScheme and
//  then updates the global-best tracker.  After the barrier every thread
//  resumes its next block of N_R iterations.  This repeats until either a
//  zero-error solution is found or MAX_ITERS is exhausted.
//
//  Deadlock guarantee
//  ──────────────────
//  std::barrier requires all N_I threads to arrive every epoch.  Even when a
//  solution is detected, threads only break *after* arriving at the barrier,
//  so the count is always satisfied.
// =============================================================================

#include "circuit_types.hpp"
#include "generator.hpp"
#include "evaluator.hpp"
#include "reader.hpp"
#include "algorithms/pso_algorithm.hpp"
#include "algorithms/propagation.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace LogicCircuitSynthesis;

// =============================================================================
//  Configuration — edit these constants to tune the run
// =============================================================================

// ── Parallel parameters ───────────────────────────────────────────────────────
static constexpr int N_I       = 4;        // Number of parallel processing units (islands)
static constexpr int N_R       = 200;      // Migration interval: migrate every N_R iterations
static constexpr int N_P       = 3;        // Particles migrated per event per source swarm
static constexpr int MAX_ITERS = 100'000;  // Hard stop if no solution is found

static constexpr PropagationScheme SCHEME = PropagationScheme::ONE_TO_N;

// ── PSO hyper-parameters (identical for every island) ────────────────────────
static constexpr int    SWARM_SIZE    = 20;
static constexpr double PHI1          = 1.5;   // Cognitive coefficient
static constexpr double PHI2          = 1.5;   // Social coefficient
static constexpr double V_MAX         = 6.0;   // Velocity clamp
static constexpr double MUTATION_RATE = 0.05;  // Per-gene mutation probability

// ── CGP topology ─────────────────────────────────────────────────────────────
static constexpr int NUM_INPUTS    = 4;
static constexpr int NUM_OUTPUTS   = 2;
static constexpr int NUM_NODES     = 20;
static constexpr int LEVELS_BACK   = 5;
static constexpr int NUM_FUNCTIONS = 8;

// Derived: 3 genes per node (func, in1, in2) + 1 gene per output
static constexpr int TOTAL_DIMS = (NUM_NODES * 3) + NUM_OUTPUTS;

// ── Benchmark file ────────────────────────────────────────────────────────────
static const std::string BENCHMARK_FILE = "benchmark.plu";

// =============================================================================
//  Shared runtime state (all accessed only inside barrier completion or
//  after all threads have joined — no races)
// =============================================================================

namespace {

struct SharedState {
    // Swarms: swarms[j] is owned exclusively by thread j between barriers.
    // During the barrier completion function all threads are suspended, so
    // migration can read/write all swarms without any additional locks.
    std::vector<std::vector<Particle>> swarms;

    std::atomic<bool> solution_found { false };
    std::atomic<int>  global_best_fitness { 999'999'999 };

    // Protected by best_mutex — only updated inside the barrier completion
    // (single-threaded context), so the mutex is technically redundant there,
    // but kept for clarity and in case the design is extended later.
    std::mutex  best_mutex;
    Genotype    global_best_genotype;

    // Iteration counter — written only inside barrier completion (single
    // thread), read by threads only after arriving at the barrier.
    int K { 0 };

    // RNG for migration (used only inside barrier completion → no races)
    std::mt19937 migration_rng { std::random_device{}() };

    // Mutex protecting stdout
    std::mutex print_mutex;

    explicit SharedState(int n_islands) : swarms(n_islands) {}
};

} // anonymous namespace

// =============================================================================
//  Per-island initialisation
// =============================================================================

static void init_swarm(int                     island_id,
                       std::vector<Particle>&  swarm,
                       DiscretePSO&            pso,
                       Generator&              gen) {
    swarm.resize(SWARM_SIZE);
    for (int i = 0; i < SWARM_SIZE; i++) {
        Particle& p     = swarm[i];
        p.genotype      = gen.generate_random_individual();
        p.current_pos   = pso.flatten_genotype(p.genotype);
        p.pbest_pos     = p.current_pos;
        p.velocity      = std::vector<double>(TOTAL_DIMS, 0.0);
        p.current_fitness = 999'999'999;
        p.pbest_fitness   = 999'999'999;
    }
}

// =============================================================================
//  Worker thread
//  Each island executes one PSO step per iteration, then arrives at the
//  barrier.  The barrier's completion function handles migration and global
//  tracking.  After the barrier, the thread checks solution_found and breaks
//  if the search is over.
// =============================================================================

static void island_worker(int          island_id,
                          SharedState& state,
                          DiscretePSO& pso,
                          Evaluator&   evaluator,
                          Generator& /*generator*/,
                          const PluReader& benchmark,
                          std::barrier<std::function<void()>>& sync_point) {
    for (int local_iter = 0; local_iter < MAX_ITERS; local_iter++) {

        // ── Execute one PSO step (reads/writes only swarms[island_id]) ──────
        if (!state.solution_found.load(std::memory_order_relaxed)) {
            pso.step(state.swarms[island_id], evaluator, benchmark);
        }

        // ── Arrive at the epoch barrier ──────────────────────────────────────
        //    The completion function runs here (on the last-arriving thread).
        //    All other threads block until the completion function returns.
        sync_point.arrive_and_wait();

        // ── Post-barrier: check termination ──────────────────────────────────
        if (state.solution_found.load(std::memory_order_acquire))
            break;
    }

    // Log exit
    {
        std::lock_guard<std::mutex> lk(state.print_mutex);
        std::cout << "[Island " << island_id << "] exiting after "
                  << state.K << " iterations.\n";
    }
}

// =============================================================================
//  main
// =============================================================================

int main() {

    // ── Load benchmark ────────────────────────────────────────────────────────
    PluReader benchmark;
    if (!benchmark.load(BENCHMARK_FILE)) {
        std::cerr << "ERROR: Could not load benchmark file: " << BENCHMARK_FILE << "\n";
        return 1;
    }

    // ── Construct per-island resources ────────────────────────────────────────
    //    Vectors are reserved first so that emplace_back never invalidates
    //    existing references (important: we capture references in threads).
    std::vector<DiscretePSO> pso_instances;
    std::vector<Generator>   generators;
    std::vector<Evaluator>   evaluators;

    pso_instances.reserve(N_I);
    generators.reserve(N_I);
    evaluators.reserve(N_I);

    for (int j = 0; j < N_I; j++) {
        // NOTE: adjust Generator / Evaluator constructors to match your project
        pso_instances.emplace_back(SWARM_SIZE, MAX_ITERS,
                                   PHI1, PHI2, V_MAX, MUTATION_RATE,
                                   NUM_INPUTS, NUM_OUTPUTS,
                                   NUM_NODES, LEVELS_BACK, NUM_FUNCTIONS);
        generators.emplace_back(NUM_INPUTS, NUM_OUTPUTS,
                                NUM_NODES, LEVELS_BACK, NUM_FUNCTIONS);
        evaluators.emplace_back();
    }

    // ── Initialise shared state & swarms ─────────────────────────────────────
    SharedState state(N_I);

    for (int j = 0; j < N_I; j++)
        init_swarm(j, state.swarms[j], pso_instances[j], generators[j]);

    // ── Build barrier with completion function ────────────────────────────────
    //    The completion lambda runs once per epoch on the last-arriving thread
    //    while all other threads are suspended — safe to touch all swarms.
    //
    //    Responsibilities:
    //      1. Increment global iteration counter K.
    //      2. Scan every swarm and update global best.
    //      3. If K is a multiple of N_R, trigger migration.
    //      4. Print a progress line.
    auto completion = [&state]() noexcept {
        // 1. Increment K
        state.K++;

        // 2. Update global best by scanning all swarms
        for (auto& swarm : state.swarms) {
            for (const Particle& p : swarm) {
                if (p.pbest_fitness < state.global_best_fitness.load(std::memory_order_relaxed)) {
                    state.global_best_fitness.store(p.pbest_fitness, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lk(state.best_mutex);
                    state.global_best_genotype = p.genotype;
                    if (p.pbest_fitness == 0)
                        state.solution_found.store(true, std::memory_order_release);
                }
            }
        }

        // 3. Migration — every N_R epochs
        if (state.K % N_R == 0 && !state.solution_found.load(std::memory_order_relaxed)) {
            propagate(state.swarms, SCHEME, N_P, state.migration_rng);
            std::lock_guard<std::mutex> lk(state.print_mutex);
            std::cout << "  [K=" << std::setw(6) << state.K
                      << "] Migration event ("
                      << scheme_name(SCHEME) << ", N_P=" << N_P << ")"
                      << "  Global best error: "
                      << state.global_best_fitness.load() << "\n";
        }

        // 4. Progress report every 1000 epochs
        if (state.K % 1000 == 0) {
            std::lock_guard<std::mutex> lk(state.print_mutex);
            std::cout << "  [K=" << std::setw(6) << state.K
                      << "] Global best error: "
                      << state.global_best_fitness.load() << "\n";
        }
    };

    std::barrier<std::function<void()>> sync_point(N_I, completion);

    // ── Launch island threads ─────────────────────────────────────────────────
    const auto t_start = std::chrono::steady_clock::now();

    std::cout << "=== Parallel Island PSO ===\n"
              << "  Islands       : " << N_I          << "\n"
              << "  Swarm size    : " << SWARM_SIZE   << "\n"
              << "  Migration every " << N_R << " iters, " << N_P
                                      << " particles, scheme: "
                                      << scheme_name(SCHEME)           << "\n"
              << "  Max iterations: " << MAX_ITERS    << "\n\n";

    std::vector<std::thread> threads;
    threads.reserve(N_I);

    for (int j = 0; j < N_I; j++) {
        threads.emplace_back(island_worker,
                             j,
                             std::ref(state),
                             std::ref(pso_instances[j]),
                             std::ref(evaluators[j]),
                             std::ref(generators[j]),
                             std::cref(benchmark),
                             std::ref(sync_point));
    }

    // ── Wait for all islands to finish ────────────────────────────────────────
    for (auto& t : threads)
        t.join();

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(t_end - t_start).count();

    // ── Final report ──────────────────────────────────────────────────────────
    std::cout << "\n=== RESULT ===\n";
    if (state.solution_found.load()) {
        std::cout << "SUCCESS: Circuit synthesised with 0 errors "
                  << "after " << state.K << " iterations "
                  << "(wall time: " << std::fixed << std::setprecision(2)
                  << elapsed_s << " s)\n";
    } else {
        std::cout << "TERMINATED: Max iterations (" << MAX_ITERS << ") reached.\n"
                  << "Best error  : " << state.global_best_fitness.load() << "\n"
                  << "Iterations  : " << state.K << "\n"
                  << "Wall time   : " << std::fixed << std::setprecision(2)
                  << elapsed_s << " s\n";
    }

    // global_best_genotype is ready to use here if needed
    // e.g. serialize / display it:
    // display_genotype(state.global_best_genotype);

    return state.solution_found.load() ? 0 : 1;
}
