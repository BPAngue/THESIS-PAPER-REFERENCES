#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <filesystem>

// =============================================================================
// reader.hpp — mirrors CGP++ BenchmarkFileReader.h exactly
//
// Key CGP++ behaviors replicated:
//   1. Reads .i .o .p header then data rows
//   2. Stores inputs/outputs as vector<vector<E>> indexed [instance][var]
//      i.e. inputs[instance_idx][input_idx]
//   3. Uses unsigned int (uint32_t) to match CGP++ EVALUATION_TYPE
//   4. num_inputs, num_outputs, num_instances from file header
// =============================================================================

namespace LogiSwarm {

    class CgpPluReader {
    public:
        int num_inputs   = 0;
        int num_outputs  = 0;
        int num_patterns = 0;  // = num_instances in CGP++

        // Mirrors CGP++ BenchmarkFileReader storage:
        // inputs[instance_idx]  = vector of num_inputs values
        // outputs[instance_idx] = vector of num_outputs values
        std::vector<std::vector<uint32_t>> inputs;
        std::vector<std::vector<uint32_t>> outputs;

        CgpPluReader() {}

        CgpPluReader(const std::string& filepath) {
            read_benchmark_file(filepath);
        }

        // -----------------------------------------------------------------------
        // read_benchmark_file — mirrors CGP++ BenchmarkFileReader::read_benchmark_file
        //
        // CGP++ reads instances row by row:
        //   for i in 0..num_instances:
        //     read num_inputs values -> inputs[i]
        //     skip whitespace
        //     read num_outputs values -> outputs[i]
        // -----------------------------------------------------------------------
        void read_benchmark_file(const std::string& file_path) {

            if (file_path.empty()) {
                throw std::runtime_error("File path is an empty string!");
            }

            std::string extension = std::filesystem::path(file_path).extension().string();
            if (extension != ".plu" && extension != ".dat") {
                throw std::runtime_error("Method only accepts PLU or DAT files!");
            }

            std::ifstream ifs(file_path, std::ifstream::in);
            if (!ifs.is_open()) {
                throw std::runtime_error("Error opening benchmark file!");
            }

            std::string str;
            uint32_t input_val;
            uint32_t output_val;
            char c;

            // Read header — mirrors CGP++: ifs >> str >> num_inputs etc.
            ifs >> str >> num_inputs;
            ifs >> str >> num_outputs;
            ifs >> str >> num_patterns;

            // Read instances — mirrors CGP++ instance loop
            for (int i = 0; i < num_patterns; i++) {
                std::vector<uint32_t> input_chunk;
                std::vector<uint32_t> output_chunk;

                // Read num_inputs values
                for (int j = 0; j < num_inputs; j++) {
                    ifs >> input_val;
                    input_chunk.push_back(input_val);
                }

                // Skip whitespace between inputs and outputs
                // mirrors CGP++: do { ifs.get(c); } while (ifs.peek() == ' ');
                do {
                    ifs.get(c);
                } while (ifs.peek() == ' ');

                // Read num_outputs values
                for (int j = 0; j < num_outputs; j++) {
                    ifs >> output_val;
                    output_chunk.push_back(output_val);
                }

                inputs.push_back(input_chunk);
                outputs.push_back(output_chunk);
            }

            if (!ifs.good()) {
                throw std::runtime_error("Error while reading benchmark file!");
            }

            ifs.close();
        }
    };

} // namespace LogiSwarmcws