#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace LogicCircuitSynthesis {

    // Gate logic functions mapping
    inline uint32_t gate_NULL (uint32_t a, uint32_t b) { return 0; }
    inline uint32_t gate_NOR (uint32_t a, uint32_t b) { return ~(a | b); }
    inline uint32_t gate_INHb (uint32_t a, uint32_t b) { return b & ~a; }
    inline uint32_t gate_NOTa (uint32_t a, uint32_t b) { return ~a; }
    inline uint32_t gate_INHa (uint32_t a, uint32_t b) { return a & ~b; }
    inline uint32_t gate_NOTb (uint32_t a, uint32_t b) { return ~b; }
    inline uint32_t gate_XOR (uint32_t a, uint32_t b) { return a ^ b; }
    inline uint32_t gate_NAND (uint32_t a, uint32_t b) { return ~(a & b); }
    inline uint32_t gate_AND (uint32_t a, uint32_t b) { return a & b; }
    inline uint32_t gate_XNOR (uint32_t a, uint32_t b) { return ~(a ^ b); }
    inline uint32_t gate_BUFb (uint32_t a, uint32_t b) { return b; }
    inline uint32_t gate_IMPb (uint32_t a, uint32_t b) { return ~a | b; }
    inline uint32_t gate_BUFa (uint32_t a, uint32_t b) { return a; }
    inline uint32_t gate_IMPa (uint32_t a, uint32_t b) { return a | ~b; }
    inline uint32_t gate_OR (uint32_t a, uint32_t b) { return a | b; }
    inline uint32_t gate_ID (uint32_t a, uint32_t b) { return ~0u; }

    typedef uint32_t (*GateFunction)(uint32_t, uint32_t);

    inline const GateFunction AllGates[] = {
        gate_NULL,   // 0  NULL
        gate_NOR,    // 1  NOR
        gate_INHb,   // 2  INHb
        gate_NOTa,   // 3  NOTa
        gate_INHa,   // 4  INHa
        gate_NOTb,   // 5  NOTb
        gate_XOR,    // 6  XOR
        gate_NAND,   // 7  NAND
        gate_AND,    // 8  AND
        gate_XNOR,   // 9  XNOR
        gate_BUFb,   // 10 BUFb
        gate_IMPb,   // 11 IMPb
        gate_BUFa,   // 12 BUFa
        gate_IMPa,   // 13 IMPa
        gate_OR,     // 14 OR
        gate_ID      // 15 ID
    };
    
    // Gate type libraries used in General Boolean Function Benchmark Suite (GBFS) (Kalkreuth et. al, 2023)
    enum GateType {
        NULL_GATE = 0,
        NOR       = 1,
        INHb      = 2,
        NOTa      = 3,
        INHa      = 4,
        NOTb      = 5,
        XOR       = 6,
        NAND      = 7,
        AND       = 8,
        XNOR      = 9,
        BUFb      = 10,
        IMPb      = 11,
        BUFa      = 12,
        IMPa      = 13,
        OR        = 14,
        ID        = 15
    };

    inline const int FunctionSetReduced[]  = { GateType::BUFa, GateType::NOTa,
                                                GateType::AND,  GateType::OR };
    
    inline const int FunctionSetExtended[] = { GateType::BUFa, GateType::NOTa,
                                                GateType::AND,  GateType::OR,
                                                GateType::XOR,  GateType::NAND,
                                                GateType::NOR,  GateType::XNOR };

    inline const int FunctionSetCrypto[]   = { GateType::AND,  GateType::XOR,
                                                GateType::OR,   GateType::XNOR,
                                                GateType::INHb };

    // Function set identifiers used in GBFS (Kalkreuth et. al, 2023)
    enum FunctionSet {
        REDUCED  = 0, // {BUFa, NOTa, AND, OR}              
        EXTENDED = 1, // {BUFa, NOTa, AND, OR, XOR, NAND, NOR, XNOR}
        CRYPTO   = 2  // {AND, XOR, OR, XNOR, INHb}
    };

    // Returns a pointer to the chosen function set and writes its size
    inline const int* get_function_set(FunctionSet fs, int& size) {
        switch (fs) {
            case REDUCED:
                size = 4;
                return FunctionSetReduced;
            case EXTENDED:
                size = 8;
                return FunctionSetExtended;
            case CRYPTO:
                size = 5;
                return FunctionSetCrypto;
            default:
                throw std::invalid_argument("Unknown FunctionSet value");
        }
    }

    // For logging purposes
    inline std::string function_set_name(FunctionSet fs) {
        switch (fs) {
            case REDUCED:  return "Reduced  {BUFa, NOTa, AND, OR}";
            case EXTENDED: return "Extended {BUFa, NOTa, AND, OR, XOR, NAND, NOR, XNOR}";
            case CRYPTO:   return "Crypto   {AND, XOR, OR, XNOR, INHb}";
            default:       return "Unknown";
        }
    }


    // Represents a single logic gate in the circuit, based on CGP representation (Miller, )
    struct Node {
        int function_idx;
        int input_1_idx;
        int input_2_idx;
    };

    // Represents the entire circuit layout
    struct Genotype {
        std::vector<Node> logic_nodes;
        std::vector<int> output_genes;
    };
}