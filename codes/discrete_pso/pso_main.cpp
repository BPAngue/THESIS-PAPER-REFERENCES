#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include "header_files/circuit_types.hpp"
#include "header_files/reader.hpp"
#include "header_files/generator.hpp"
#include "header_files/evaluator.hpp"
#include "header_files/printer.hpp"
#include "header_files/algorithms/pso_algorithm.hpp"

// to run: g++ -O2 -std=c++17 pso_main.cpp -o build/pso_main

using namespace LogicCircuitSynthesis;

// --------------------------------------------------------------------------
// TeeBuffer: a custom streambuf that mirrors every write to both the original
// console buffer and an output file simultaneously.
// --------------------------------------------------------------------------
class TeeBuffer : public std::streambuf {
public:
    TeeBuffer(std::streambuf* console_buf, std::streambuf* file_buf)
        : console_(console_buf), file_(file_buf) {}

protected:
    int overflow(int c) override {
        if (c == EOF) return EOF;
        if (console_->sputc(static_cast<char>(c)) == EOF) return EOF;
        if (file_->sputc(static_cast<char>(c)) == EOF) return EOF;
        return c;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        console_->sputn(s, n);
        file_->sputn(s, n);
        return n;
    }

private:
    std::streambuf* console_;
    std::streambuf* file_;
};

// Builds a timestamped log filepath inside the "logs/" directory,
// e.g. "logs/pgphea_run_20250306_143022.txt".
// Creates the "logs/" directory if it does not already exist.
static std::string make_log_filename() {
    const std::filesystem::path log_dir = "logs";
    std::filesystem::create_directories(log_dir);   // no-op if already exists

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << "logs/pso_run_"
        << std::put_time(&tm, "%Y%m%d_%H%M%S")
        << ".txt";
    return oss.str();
}

static void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <path_to_plu_file> <function_set>\n\n"
              << "  function_set:\n"
              << "    0 = REDUCED   {BUFa, NOTa, AND, OR}\n"
              << "    1 = EXTENDED  {BUFa, NOTa, AND, OR, XOR, NAND, NOR, XNOR}\n"
              << "    2 = CRYPTO    {AND, XOR, OR, XNOR, INHb}\n";
}

static FunctionSet parse_function_set(const char* arg) {
    int val = std::stoi(arg);
    if (val < 0 || val > 2)
        throw std::invalid_argument(
            "function_set must be 0 (REDUCED), 1 (EXTENDED), or 2 (CRYPTO). Got: "
            + std::to_string(val));
    return static_cast<FunctionSet>(val);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Set up tee logging: every std::cout write also goes to a timestamped
    // .txt file inside the "logs/" directory.
    // -----------------------------------------------------------------------
    std::string log_filename = make_log_filename();
    std::ofstream log_file(log_filename);
    if (!log_file.is_open()) {
        std::cerr << "Warning: could not open log file '" << log_filename
                  << "'. Continuing without file logging.\n";
    }

    // Replace cout's buffer with our tee; restore on scope exit
    std::streambuf* original_cout = std::cout.rdbuf();
    TeeBuffer tee_buf(original_cout, log_file.rdbuf());
    if (log_file.is_open()) {
        std::cout.rdbuf(&tee_buf);
    }
    
    // Ensure cout is restored even if an exception escapes
    struct CoutRestorer {
        std::streambuf* original;
        ~CoutRestorer() { std::cout.rdbuf(original); }
    } restorer{original_cout};

    std::cout << "Log file         : " << log_filename << "\n";

    try {
         // 1. Parse arguments
        std::string filepath = argv[1];
        FunctionSet function_set = parse_function_set(argv[2]);

        // 2. Load benchmark
        PluReader benchmark(filepath);
        std::cout << "Loaded Benchmark : " << filepath << "\n";
        std::cout << "Inputs           : " << benchmark.num_inputs
                  << " | Outputs: " << benchmark.num_outputs << "\n";

        // 3. CGP graph parameters
        int max_nodes   = 5000;
        int levels_back = max_nodes;

        // 4. Evaluator and Generator
        Evaluator evaluator(benchmark.num_inputs, max_nodes,
                            benchmark.num_outputs, function_set);
        int num_gate_types = evaluator.num_functions();

        Generator generator(benchmark.num_inputs, benchmark.num_outputs,
                            max_nodes, levels_back, num_gate_types);

        // 5. PSO parameters
        int swarm_size = 15;
        int max_iterations = 100000000;
        int total_dimensions = (max_nodes * 3) + benchmark.num_outputs; 

        double phi1                = 2.0;      // c1 cognitive weight
        double phi2                = 2.0;      // c2 social weight
        double vMax                = 4.0;      // velocity max
        double mutation_rate   = 0.003;     // 0.3% per gene

        // 6. Print configuration
        std::cout << "\n--- Starting Logic Synthesis via Discrete PSO ---\n";
        std::cout << "Function set     : " << function_set_name(function_set)
                  << " (" << num_gate_types << " gates)\n"
                  << "max_nodes        : " << max_nodes        << "\n"
                  << "total_dimensions : " << total_dimensions << "\n"
                  << "swarm_size       : " << swarm_size       << "\n"
                  << "phi1 / phi2 / w  : " << phi1 << " / " << phi2 << "\n"
                  << "vMax             : " << vMax             << "\n"
                  << "mutation_rate    : " << mutation_rate    << "  (1/dims)\n"
                  << "max_iterations   : " << max_iterations   << "\n\n";
        
        // 7. Run the PSO optimization
        DiscretePSO pso(swarm_size, max_iterations, phi1, phi2, vMax, mutation_rate,
                        benchmark.num_inputs, benchmark.num_outputs, max_nodes,
                        levels_back, num_gate_types);

        Genotype best_circuit = pso.optimize(generator, evaluator, benchmark);

        // 8. Final evaluation
        int final_error = evaluator.calculate_fitness(best_circuit, benchmark);
        std::cout << "\n=========================================\n";
        std::cout << "Optimization Complete!\n";
        std::cout << "Best Circuit Hamming Distance: " << final_error << "\n";

        PrintCircuitFormula(best_circuit, benchmark.num_inputs, max_nodes,
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