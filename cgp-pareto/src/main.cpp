#include "periodizer.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <map>

// ─────────────────────────────────────────────
//  Parse periodization scheme from string
//  e.g. "nh4" → [{NSGAII,1},{hES,4}}
//       "nh"  → [{NSGAII,1},{hES,1}]
//       "sh"  → [{SPEA2,1},{hES,1}]
//       "n"   → [{NSGAII,1}]
//       "nhs" → [{NSGAII,1},{hES,1},{SPEA2,1}]
// ─────────────────────────────────────────────
inline std::vector<std::pair<int,int>> parse_scheme(const std::string& s) {
    std::vector<std::pair<int,int>> period;
    int i = 0;
    while (i < (int)s.size()) {
        int alg_id = -1;
        if (s[i] == 'n') { alg_id = 0; i++; }
        else if (s[i] == 's') { alg_id = 1; i++; }
        else if (s[i] == 'h') { alg_id = 2; i++; }
        else { i++; continue; }

        int rep = 1;
        if (i < (int)s.size() && std::isdigit(s[i])) {
            rep = 0;
            while (i < (int)s.size() && std::isdigit(s[i]))
                rep = rep * 10 + (s[i++] - '0');
        }
        period.push_back({alg_id, rep});
    }
    if (period.empty()) period.push_back({0, 1}); // default: NSGAII
    return period;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --bench   adder|mult       Benchmark (default: adder)\n"
              << "  --scheme  SCHEME           Periodization scheme (default: nh4)\n"
              << "            n  = NSGAII\n"
              << "            s  = SPEA2\n"
              << "            h  = hES\n"
              << "            nh = NSGAII then hES (1 step each)\n"
              << "            nh4= NSGAII then 4 steps hES\n"
              << "            sh = SPEA2 then hES\n"
              << "            nhs= 3-tuple NSGAII→hES→SPEA2\n"
              << "  --evals   N                Max fitness evaluations (default: 400000)\n"
              << "  --pop     N                Population size (default: 50)\n"
              << "  --lambda  N                hES offspring per parent (default: 4)\n"
              << "  --seed    N                RNG seed (default: 42)\n"
              << "  --quiet                    Suppress progress output\n"
              << "  --verify                   Print truth table of best correct solution\n"
              << "  --compare                  Run all 6 schemes and compare\n\n"
              << "Examples:\n"
              << "  " << prog << " --bench adder --scheme nh4\n"
              << "  " << prog << " --bench mult  --scheme sh --evals 1600000\n"
              << "  " << prog << " --compare --bench adder --evals 50000\n";
}

// ─────────────────────────────────────────────
//  Run all schemes and print comparison table
// ─────────────────────────────────────────────
void run_comparison(const Config& base_cfg) {
    const std::vector<std::string> schemes = {"n","s","h","nh","nh4","nh10"};

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Periodization Comparison  (Table IX style)       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
    std::cout << std::left
              << std::setw(12) << "Scheme"
              << std::setw(14) << "BestError"
              << std::setw(12) << "BestArea"
              << std::setw(12) << "BestDelay"
              << std::setw(10) << "Correct"
              << std::setw(12) << "FrontSize\n";
    std::cout << std::string(72, '=') << "\n";

    for (auto& scheme : schemes) {
        Config cfg = base_cfg;
        cfg.period  = parse_scheme(scheme);
        cfg.verbose = false;

        auto result = run_periodized(cfg);
        auto& st    = result.stats;

        std::cout << std::left
                  << std::setw(12) << scheme
                  << std::setw(14) << std::fixed << std::setprecision(4) << st.best_error
                  << std::setw(12) << (int)st.best_area
                  << std::setw(12) << (int)st.best_delay
                  << std::setw(10) << st.correct
                  << std::setw(12) << st.size << "\n";
    }
    std::cout << std::string(72, '=') << "\n\n";
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg;
    std::string scheme_str = "nh4";
    bool do_verify  = false;
    bool do_compare = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        } else if (strcmp(argv[i], "--bench") == 0 && i+1 < argc) {
            std::string b = argv[++i];
            if (b == "mult" || b == "multiplier") {
                cfg.bench = BenchmarkType::MULTIPLIER_2x2;
                cfg.ni    = 4; cfg.no = 4;
                cfg.max_evals = 1600000;
            } else {
                cfg.bench = BenchmarkType::ADDER_2x2;
                cfg.ni    = 4; cfg.no = 3;
                cfg.max_evals = 400000;
            }
        } else if (strcmp(argv[i], "--scheme") == 0 && i+1 < argc) {
            scheme_str = argv[++i];
        } else if (strcmp(argv[i], "--evals") == 0 && i+1 < argc) {
            cfg.max_evals = std::stol(argv[++i]);
        } else if (strcmp(argv[i], "--pop") == 0 && i+1 < argc) {
            cfg.pop_size = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--lambda") == 0 && i+1 < argc) {
            cfg.lambda_hes = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) {
            cfg.seed = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--quiet") == 0) {
            cfg.verbose = false;
        } else if (strcmp(argv[i], "--verify") == 0) {
            do_verify = true;
        } else if (strcmp(argv[i], "--compare") == 0) {
            do_compare = true;
        } else if (strcmp(argv[i], "--nc") == 0 && i+1 < argc) {
            cfg.nc = std::stoi(argv[++i]);
            cfg.l  = cfg.nc;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]); return 1;
        }
    }

    if (do_compare) {
        cfg.report_interval = cfg.max_evals + 1; // suppress per-step output
        run_comparison(cfg);
        return 0;
    }

    cfg.period = parse_scheme(scheme_str);

    auto result = run_periodized(cfg);

    if (do_verify && !result.final_pareto.empty()) {
        // Find a correct circuit if any
        CGPChromosome* best = nullptr;
        for (auto& c : result.final_pareto) {
            if (c.error == 0.0) {
                if (!best || c.area < best->area) best = &c;
            }
        }
        if (best) {
            std::cout << "Best correct circuit found:\n";
            print_circuit_info(*best);
            print_truth_table(*best, cfg.bench);
        } else {
            // Show closest to correct
            auto it = std::min_element(result.final_pareto.begin(),
                                       result.final_pareto.end(),
                                       [](auto& a, auto& b){ return a.error < b.error; });
            std::cout << "No fully correct circuit found. Best approximation:\n";
            print_circuit_info(*it);
            print_truth_table(*it, cfg.bench);
        }
    }

    return 0;
}
