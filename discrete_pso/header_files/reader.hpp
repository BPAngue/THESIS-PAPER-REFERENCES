#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>

namespace LogicCircuitSynthesis {
    // Reads values inside a .plu file where the truth table is compressed into 32-bit integers
    class PluReader {
    public:
        int num_inputs = 0;
        int num_outputs = 0;
        int num_patterns = 0;
        std::vector<std::vector<uint32_t>> inputs;
        std::vector<std::vector<uint32_t>> outputs;

        PluReader(const std::string& filepath) {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open: " + filepath);
            }

            std::string token;
            bool reading_data = false;
            int current_pattern = 0;

            while (file >> token) {
                if (token == ".i") file >> num_inputs;
                else if (token == ".o") file >> num_outputs;
                else if (token == ".p") {
                    file >> num_patterns;
                    inputs.assign(num_inputs, std::vector<uint32_t>(num_patterns, 0));
                    outputs.assign(num_outputs, std::vector<uint32_t>(num_patterns, 0));
                    reading_data = true;
                } 
                else if (token == ".e") break;
                else if (reading_data) {
                    inputs[0][current_pattern] = std::stoul(token);
                    for (int i = 1; i < num_inputs; i++) file >> inputs[i][current_pattern];
                    for (int o = 0; o < num_outputs; o++) file >> outputs[o][current_pattern];
                    current_pattern++;
                }
            }
        }
    };
}