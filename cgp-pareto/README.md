# CGP-hES: Hybrid Evolutionary Strategy for Combinational Logic Circuit Evolution

A C++17 implementation of the paper:

> **"A Novel Hybrid Evolutionary Strategy and its Periodization with Multi-objective Genetic Optimizers"**
> Kaufmann, Knieper & Platzner — IEEE Congress on Evolutionary Computation

---

## Overview

This project evolves combinational logic circuits (2×2 adder and multiplier) using
**Cartesian Genetic Programming (CGP)** combined with a periodized execution of:

- **NSGA-II** — global multi-objective genetic algorithm (Pareto-based)
- **SPEA2**   — Strength Pareto Evolutionary Algorithm 2
- **hES**     — novel hybrid Evolutionary Strategy (local search, Pareto-aware)

The three objectives optimised simultaneously are:

| Objective | Description                                   |
|-----------|-----------------------------------------------|
| Error     | Normalised bit-error rate over all input cases |
| Area      | Number of active (reachable) nodes in the CGP graph |
| Delay     | Critical path length (number of logic levels) |

---

## Architecture

```
include/
  cgp_types.h     — Data structures: CGPChromosome, Config, ObjVec, dominance
  cgp_eval.h      — Circuit simulation, truth-table evaluation, area/delay metrics
  cgp_ops.h       — Random initialisation, point mutation, crossover, RNG
  nsgaii.h        — Fast non-dominated sort, crowding distance, NSGA-II step, SPEA2 step
  hes.h           — Algorithms 1,2,3 from paper: hES-step, ES-generate, add-replace
  periodizer.h    — Periodized execution engine, statistics, reporting

src/
  main.cpp        — CLI argument parsing, scheme selector, comparison runner
```

---

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Requirements:** C++17, CMake ≥ 3.14, GCC or Clang.

---

## Usage

```
./cgp_hes [options]

Options:
  --bench   adder|mult       Benchmark (default: adder)
  --scheme  SCHEME           Periodization scheme (default: nh4)
              n   = NSGA-II only
              s   = SPEA2 only
              h   = hES only
              nh  = NSGA-II then hES (1 step each)
              nh4 = NSGA-II then 4× hES steps  [best for ZDT6 in paper]
              sh  = SPEA2 then hES
              nhs = 3-tuple: NSGA-II → hES → SPEA2
  --evals   N                Max fitness evaluations (default: 400000)
  --pop     N                Population size (default: 50)
  --lambda  N                hES offspring per parent (default: 4)
  --seed    N                RNG seed (default: 42)
  --quiet                    Suppress progress output
  --verify                   Print truth table of best correct circuit
  --compare                  Run all 6 schemes and print comparison table
  --nc      N                CGP grid columns (default: 150)
```

---

## Periodization Schemes

The periodization model from the paper is:

```
P = A^F_I = (a^f1_i1, a^f2_i2, ..., a^fm_im)
```

Where **A** is the algorithm set, **I** is the execution index sequence and
**F** is the repetition count sequence.

| Scheme | Execution pattern per generation |
|--------|----------------------------------|
| `n`    | 1× NSGA-II                       |
| `s`    | 1× SPEA2                         |
| `h`    | 1× hES                           |
| `nh`   | 1× NSGA-II, 1× hES               |
| `nh4`  | 1× NSGA-II, 4× hES               |
| `nh10` | 1× NSGA-II, 10× hES              |
| `sh`   | 1× SPEA2, 1× hES                 |
| `nhs`  | 1× NSGA-II, 1× hES, 1× SPEA2    |

---

## hES Algorithm (Paper Algorithms 1–3)

The hybrid Evolutionary Strategy is a **1+λ ES tailored for Pareto-based search**:

```
hES-step(λ, Pt):
  1. For each parent p ∈ Pt:
       Generate λ mutated offspring via ES-generate(p, λ)
  2. Merge parents+offspring via add-replace (handles genetic drift)
  3. Fast non-dominated sort the merged set
  4. For each non-domination front:
       Group members by parent
       Select one survivor per parent group:
         - Keep the parent if it's in its own group
         - Otherwise, keep the least-crowded member (crowding distance)
  5. Discard all other individuals
```

Key innovations over standard ES:
- Uses **NSGA-II fast non-dominated sort** instead of scalar fitness
- Uses **crowding distance** to maintain Pareto front diversity
- **Genetic drift**: offspring with identical Pareto vectors replace parents, enabling neutral search

---

## CGP Representation

Each chromosome is a grid of `nr × nc` functional nodes plus `no` output genes:

```
Primary inputs (ni)
     │   │   │   │
     ▼   ▼   ▼   ▼
 ┌───────────────────── ...  ──────────────┐
 │ f0  f1  f2  f3              f(nr*nc-1)  │  ← nr rows × nc columns
 └───────────────────── ...  ──────────────┘
                                    │
                              Output genes (no)
```

Each node holds:
- **func**: one of 20 Boolean functions (AND, OR, XOR, MUX, etc.)
- **conn[0..nn-1]**: connections to earlier nodes (l-back connectivity)

Mutation randomly changes function or connection genes with probability 0.1 per gene.

---

## Boolean Function Set

| ID | Function     | ID | Function      |
|----|------------- |----|---------------|
| 0  | 0 (const)    | 10 | a ⊕ b (XOR)  |
| 1  | 1 (const)    | 11 | ¬(a⊕b) XNOR  |
| 2  | a            | 12 | a + b (OR)    |
| 3  | b            | 13 | ¬(a+b) NOR    |
| 4  | ¬a           | 14 | a + ¬b        |
| 5  | ¬b           | 15 | ¬a + b        |
| 6  | a·b (AND)    | 16 | MUX: a·c+b·¬c |
| 7  | ¬(a·b) NAND  | 17 | MUX variant   |
| 8  | a·¬b         | 18 | MUX variant   |
| 9  | ¬a·b         | 19 | MUX variant   |

---

## Dominance Strategy

This implementation uses **constrained Pareto dominance**:

- Among circuits with non-zero error: lower error always dominates
- Among circuits with identical error: standard 2-objective Pareto (area vs delay)  
- A correct circuit (error = 0) always dominates an incorrect one

This mirrors the paper's intent that correctness is the primary goal,
with area and delay as secondary trade-off objectives once correctness is achieved.

---

## Example Results

### 2×2 Adder (100k evaluations, seed=42, scheme=nh4)

```
Evals    FrontSize  BestError  BestArea  BestDelay  Correct
5150     1          0.1875     22        8          0
25550    2          0.0833     12        4          0
45100    1          0.0208     23        9          0
90150    1          0.0000     30        8          1      ← first correct
100150   2          0.0000     12        5          2
```

Truth table (all 16 cases correct):
```
A=0, B=0 → Sum=0  ✓    A=2, B=2 → Sum=4  ✓
A=1, B=0 → Sum=1  ✓    A=3, B=3 → Sum=6  ✓  (etc.)
```

### Scheme Comparison (adder, 100k evals)

| Scheme | Best Error | Correct | Front Size |
|--------|-----------|---------|-----------|
| n      | 0.0000    | 50      | 50        |
| s      | 0.0208    | 0       | 6         |
| h      | 0.0417    | 0       | 1         |
| nh     | 0.0000    | 50      | 50        |
| nh4    | 0.0000    | 2       | 2         |
| nh10   | 0.0208    | 0       | 36        |

---

## Paper Correspondence

| Paper element           | Implementation location          |
|-------------------------|----------------------------------|
| Algorithm 1 (hES-step)  | `hes.h::hes_step()`              |
| Algorithm 2 (ES-generate)| `hes.h::es_generate()`          |
| Algorithm 3 (add-replace)| `hes.h::add_replace()`          |
| Periodization model P   | `periodizer.h::run_periodized()` |
| Fast non-dom. sort      | `nsgaii.h::fast_non_dominated_sort()` |
| Crowding distance       | `nsgaii.h::crowding_distance_assignment()` |
| CGP evaluation          | `cgp_eval.h::evaluate()`         |
| CGP simulation          | `cgp_eval.h::simulate()`         |
| Table III function set  | `cgp_eval.h::eval_func()`        |
| Table II CGP config     | `cgp_types.h::Config`            |

---

## References

- Kaufmann, Knieper, Platzner (2009). "A Novel Hybrid Evolutionary Strategy..."
- Deb et al. (2000). "NSGA-II"
- Zitzler et al. (2001). "SPEA2"
- Miller & Thomson (2000). "Cartesian Genetic Programming"
