#include <iostream>
#include <stdexcept>
#include "logiswarm_types.hpp"
#include "reader.hpp"
#include "generator.hpp"
#include "evaluator.hpp"
#include "cgp_algorithm.hpp"

// To compile:
// g++ -O2 -std=c++17 main.cpp -o logiswarm

// =============================================================================
// main.cpp — mirrors CGP++ cgp.cpp main() entry point
//
// Usage:
//   ./logiswarm <path_to_plu_file> <function_set> <num_nodes> <levels_back>
//
// Arguments:
//   <path_to_plu_file>  Path to the .plu benchmark file
//   <function_set>      0=REDUCED, 1=EXTENDED, 2=CRYPTO
//   <num_nodes>         Number of function nodes (e.g. 5000 for add4)
//   <levels_back>       Levels back (e.g. 5000 for full connectivity)
//
// Example:
//   ./logiswarm benchmarks/add4.plu 0 5000 5000
//   ./logiswarm benchmarks/epar8.plu 0 1000 1000
//   ./logiswarm benchmarks/mul3.plu 0 4000 4000
// =============================================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <plu_file> <function_set> <num_nodes> <levels_back>\n"
              << "  function_set: 0=REDUCED, 1=EXTENDED, 2=CRYPTO\n";
}

int main(int argc, char* argv[]) {

    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string filepath    = argv[1];
        int fs_int              = std::stoi(argv[2]);
        int max_nodes           = std::stoi(argv[3]);
        int levels_back         = std::stoi(argv[4]);

        if (fs_int < 0 || fs_int > 2)
            throw std::invalid_argument("function_set must be 0, 1, or 2");

        LogiSwarm::FunctionSet function_set =
            static_cast<LogiSwarm::FunctionSet>(fs_int);

        // Fixed algorithm parameters — mirrors CGP++ paper Section 5.3
        int    lambda             = 1;
        double mutation_rate      = 0.003;
        int    max_generations    = 100000000;
        bool   neutral_drift      = true;
        int    ideal_fitness      = 0;

        // Read benchmark file — mirrors CGP++ BenchmarkFileReader
        LogiSwarm::CgpPluReader benchmark(filepath);

        std::cout << "\t\tCGP++ MIRROR CONFIGURATION\n";
        std::cout << "-------------------------------------------------\n";
        std::cout << "Benchmark        : " << filepath         << "\n";
        std::cout << "Number of inputs : " << benchmark.num_inputs  << "\n";
        std::cout << "Number of outputs: " << benchmark.num_outputs << "\n";
        std::cout << "Num instances    : " << benchmark.num_patterns << "\n";

        // Instantiate Evaluator — owns function set, genome_size, min/max gene
        LogiSwarm::Evaluator evaluator(
            benchmark.num_inputs,
            max_nodes,
            benchmark.num_outputs,
            levels_back,
            function_set
        );

        int num_gate_types = evaluator.num_functions();
        int genome_size    = evaluator.get_genome_size();

        std::cout << "Number of function nodes: " << max_nodes        << "\n";
        std::cout << "Levels back      : " << levels_back             << "\n";
        std::cout << "Number of functions: " << num_gate_types         << "\n";
        std::cout << "Genome size      : " << genome_size             << "\n";
        std::cout << "Mutation rate    : " << mutation_rate           << "\n";
        std::cout << "Lambda           : " << lambda                  << "\n";
        std::cout << "Neutral drift    : " << neutral_drift           << "\n";
        std::cout << "Function set     : "
                  << LogiSwarm::function_set_name(function_set)       << "\n";
        std::cout << "Max generations  : " << max_generations         << "\n";
        std::cout << "-------------------------------------------------\n\n";

        // Instantiate Generator — mirrors CGP++ Individual initialization
        LogiSwarm::Generator generator(
            benchmark.num_inputs,
            benchmark.num_outputs,
            max_nodes,
            levels_back,
            num_gate_types,
            genome_size
        );

        // Instantiate Algorithm — mirrors CGP++ OnePlusLambda
        LogiSwarm::CGPAlgorithm cgp_runner(
            lambda,
            mutation_rate,
            max_generations,
            benchmark.num_inputs,
            benchmark.num_outputs,
            max_nodes,
            levels_back,
            num_gate_types,
            genome_size,
            neutral_drift
        );

        // Run evolution
        LogiSwarm::Genotype best = cgp_runner.evolve(generator, evaluator, benchmark);

        // Final verification
        int final_error = evaluator.calculate_fitness(best, benchmark);
        std::cout << "\nFinal Circuit Hamming Distance: " << final_error << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
