#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <thread>

#include "header_files/circuit_types.hpp"
#include "header_files/reader.hpp"
#include "header_files/generator.hpp"
#include "header_files/evaluator.hpp"
#include "header_files/printer.hpp"
#include "header_files/algorithms/cgp_es_algorithm.hpp"

// Compile:
//   g++ -O2 -std=c++17 -pthread cgp_es_main.cpp -o build/cgp_es_main

using namespace LogicCircuitSynthesis;

// ── Tee logging (identical to pso_main.cpp) ────────────────────────────────
class TeeBuffer : public std::streambuf {
public:
    TeeBuffer(std::streambuf* c, std::streambuf* f) : con_(c), fil_(f) {}
protected:
    int overflow(int ch) override {
        if (ch == EOF) return EOF;
        if (con_->sputc(static_cast<char>(ch)) == EOF) return EOF;
        if (fil_->sputc(static_cast<char>(ch)) == EOF) return EOF;
        return ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        con_->sputn(s, n); fil_->sputn(s, n); return n;
    }
private:
    std::streambuf* con_; std::streambuf* fil_;
};

static std::string make_log_filename() {
    std::filesystem::create_directories("logs");
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << "logs/cgp_es_run_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".txt";
    return oss.str();
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <path_to_plu_file> <function_set>\n\n"
              << "  function_set:\n"
              << "    0 = REDUCED   {BUFa, NOTa, AND, OR}\n"
              << "    1 = EXTENDED  {BUFa, NOTa, AND, OR, XOR, NAND, NOR, XNOR}\n"
              << "    2 = CRYPTO    {AND, XOR, OR, XNOR, INHb}\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    // ── Log file ───────────────────────────────────────────────────────────
    std::string log_path = make_log_filename();
    std::ofstream log_file(log_path);
    std::streambuf* orig = std::cout.rdbuf();
    TeeBuffer tee(orig, log_file.rdbuf());
    if (log_file.is_open()) std::cout.rdbuf(&tee);
    struct Restore { std::streambuf* b; ~Restore(){ std::cout.rdbuf(b); } } _r{orig};
    std::cout << "Log file         : " << log_path << "\n";

    try {
        // ── Parse args ─────────────────────────────────────────────────────
        std::string filepath = argv[1];
        int fs_val = std::stoi(argv[2]);
        if (fs_val < 0 || fs_val > 2)
            throw std::invalid_argument("function_set must be 0, 1, or 2");
        FunctionSet function_set = static_cast<FunctionSet>(fs_val);

        // ── Load benchmark ─────────────────────────────────────────────────
        PluReader benchmark(filepath);
        std::cout << "Loaded Benchmark : " << filepath        << "\n"
                  << "Inputs           : " << benchmark.num_inputs
                  << " | Outputs: "        << benchmark.num_outputs << "\n";

        // ── CGP parameters ─────────────────────────────────────────────────
        //
        //  max_nodes / levels_back
        //  -----------------------
        //  With (1+λ) ES, large max_nodes is fine — neutral drift uses the
        //  extra nodes productively rather than drowning a velocity signal.
        //  5000 nodes with levels_back=5000 (fully connected) is a reasonable
        //  default for benchmarks up to ~10 inputs.
        //
        int max_nodes   = 5000;
        int levels_back = max_nodes;

        Evaluator evaluator(benchmark.num_inputs, max_nodes,
                            benchmark.num_outputs, function_set);
        int num_gate_types = evaluator.num_functions();

        Generator generator(benchmark.num_inputs, benchmark.num_outputs,
                            max_nodes, levels_back, num_gate_types);

        // ── ES parameters ──────────────────────────────────────────────────
        ParallelStreamsCGP::Config cfg;

        // num_streams: one thread per stream — cap at hardware concurrency.
        cfg.num_streams          = std::min(6,
                                   (int)std::thread::hardware_concurrency());

        // lambda: offspring per parent per generation.
        // Classic CGP literature uses (1+4) as a strong default.
        // More lambda = more exploration per gen but slower per-gen wall time.
        cfg.lambda               = 8;

        // Phase lengths.
        // multi_phase_gens: how long each stream evolves independently before
        //   sharing information.  Longer = more independent exploration.
        // single_phase_gens: how long streams run with gbest injection.
        //   Shorter is fine — it's mainly to propagate the best solution.
        cfg.multi_phase_gens     = 2000;
        cfg.single_phase_gens    = 500;
        cfg.max_outer_cycles     = 100000;

        // Mutation rate: per-gene probability.
        // CGP literature recommends ~1/n_active_genes, but since we don't
        // know active count upfront, 1% per gene is a good general default.
        // Higher than the PSO variant because every mutation is meaningful —
        // there is no velocity dampening to fight against.
        cfg.mutation_rate        = 0.01;

        // Stagnation restart threshold (cycles, not generations).
        cfg.stagnation_threshold = 15;

        int total_offspring_per_cycle =
            cfg.num_streams * cfg.lambda *
            (cfg.multi_phase_gens + cfg.single_phase_gens);

        // ── Print config ───────────────────────────────────────────────────
        std::cout << "\n--- Starting CGP Logic Synthesis via Parallel (1+λ) ES ---\n"
                  << "Algorithm        : Parallel Streams CGP-ES  "
                     "(PSO-PSO structure, ES engine)\n"
                  << "Function set     : " << function_set_name(function_set)
                  << " (" << num_gate_types << " gates)\n"
                  << "max_nodes        : " << max_nodes               << "\n"
                  << "levels_back      : " << levels_back             << "\n"
                  << "num_streams (K)  : " << cfg.num_streams         << "\n"
                  << "lambda (λ)       : " << cfg.lambda              << "\n"
                  << "multi_phase_gens : " << cfg.multi_phase_gens    << "\n"
                  << "single_phase_gens: " << cfg.single_phase_gens   << "\n"
                  << "mutation_rate    : " << cfg.mutation_rate       << "\n"
                  << "stagnation_thresh: " << cfg.stagnation_threshold
                  << " cycles\n"
                  << "offspring/cycle  : " << total_offspring_per_cycle << "\n"
                  << "hardware threads : "
                  << std::thread::hardware_concurrency()              << "\n\n";

        // ── Run ────────────────────────────────────────────────────────────
        ParallelStreamsCGP optimizer(cfg,
                                     benchmark.num_inputs,
                                     benchmark.num_outputs,
                                     max_nodes, levels_back,
                                     num_gate_types);

        Genotype best = optimizer.optimize(generator, evaluator, benchmark);

        // ── Report ─────────────────────────────────────────────────────────
        int final_error = evaluator.calculate_fitness(best, benchmark);
        std::cout << "\n=========================================\n"
                  << "Optimisation Complete!\n"
                  << "Best Circuit Hamming Distance: " << final_error << "\n";

        PrintCircuitFormula(best, benchmark.num_inputs, max_nodes,
                            benchmark.num_outputs, function_set);

    } catch (const std::invalid_argument& e) {
        std::cerr << "Argument Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Pipeline Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}