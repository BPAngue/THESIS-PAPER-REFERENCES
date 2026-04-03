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
#include "header_files/algorithm/cgp_sa_algorithm.hpp"

// to run: g++ -O2 -std=c++17 cgp_sa_main.cpp -o build/cgp_sa

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
    oss << "logs/cgpsa_run_"
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

    // Main entry point of the algorithm
    try {
        // 1. Parse arguments
        std::string filepath = argv[1];
        FunctionSet function_set = parse_function_set(argv[2]);

        // 2. Load benchmark circuit
        PluReader benchmark(filepath);
        int num_inputs = benchmark.num_inputs;
        int num_outputs = benchmark.num_outputs;
        std::cout << "Loaded Benchmark : " << filepath << "\n";
        std::cout << "Inputs / Outputs : " << num_inputs  << " / " << num_outputs << "\n";

        // 3. CGP graph parameters
        int max_nodes = 5000;
        int levels_back = max_nodes;
        int total_dimensions = (max_nodes * 3) + num_outputs;

        // 4. Evaluator and Generator
        Evaluator evaluator(num_inputs, max_nodes, num_outputs, function_set);
        int num_gate_types = evaluator.num_functions();

        Generator generator(num_inputs, num_outputs, max_nodes, levels_back, num_gate_types);

        // 5. CGP-SA parameters 
        // CGP parameters
        int lambda = 5;
        double mutation_rate = 0.02;
        int max_generations = 100000;

        // SA parameters
        double T_max = 5;
        double T_min = 0.1;
        double C_cool = 0.95;

        // 6. Print algorithm configurations
        std::cout << "\n--- Starting 1+lambda Evolution via CGP-SA Hybrid ---\n";
        std::cout << "Function set     : " << function_set_name(function_set)
                  << " (" << num_gate_types << " gates)\n"
                  << "max_generations  : " << max_generations << "\n"
                  << "max_nodes (n_c)  : " << max_nodes << "\n"
                  << "levels_back      : " << levels_back << "\n"
                  << "total_dimensions : " << total_dimensions << "\n"
                  << "(1 + lambda)     : 1 + " << lambda << "\n"
                  << "mutation_rate    : " << mutation_rate << "\n"
                  << "Temp max         : " << T_max << "\n"
                  << "Temp min         : " << T_min << "\n"
                  << "Cooling rate     : " << C_cool << "\n\n";

        // 7. Run the CGP-SA evolution
        CGPSAAlgorithm cgp_sa(num_inputs, num_outputs, max_generations, max_nodes, num_gate_types, levels_back,
                              total_dimensions, lambda, mutation_rate, T_max, T_min, C_cool, function_set);

        Genotype best_circuit = cgp_sa.optimize(generator, evaluator, benchmark);

        // 8. Final Evaluation
        int final_error = evaluator.calculate_fitness(best_circuit, benchmark);
        int transistors = cgp_sa.count_active_transistors(best_circuit);
        int final_fitness = cgp_sa.calculate_paper_fitness(final_error, transistors);
        std::cout << "\n=========================================\n";
        std::cout << "Optimization Complete!\n";
        std::cout << "Best Hamming Distance: " << final_error << "\n";
        std::cout << "Best Circuit Fitness: " << final_fitness << "\n";
        std::cout << "Active Transistor Count: " << transistors << "\n";

        PrintCircuitFormula(best_circuit, num_inputs, max_nodes, num_outputs, function_set);

    } catch (const std::invalid_argument& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Pipeline error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}