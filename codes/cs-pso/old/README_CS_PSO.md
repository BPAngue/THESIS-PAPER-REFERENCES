# CS-PSO Implementation for Combinatorial Circuit Optimization

## Overview

This implementation integrates the **Chaotic Particle Swarm Optimization (CS-PSO)** algorithm from the paper "CS-PSO: chaotic particle swarm optimization algorithm for solving combinatorial optimization problems" by Xu et al. (2018) into your existing multi-threaded PSO pipeline.

## Key Improvements Over Standard PSO

### 1. Chaos Initialization
- **Problem**: Standard PSO uses random initialization, which can lead to poor initial population quality
- **Solution**: CS-PSO uses the logistic map to generate initial particles
- **Benefits**: 
  - Better initial diversity
  - More thorough coverage of search space
  - Higher quality starting positions

### 2. Chaos Perturbation
- **Problem**: Standard PSO uses simple clamping/modulo when particles go out of bounds
- **Solution**: CS-PSO uses chaotic perturbation to explore the search space
- **Benefits**:
  - Avoids premature convergence
  - Better global search capability
  - Enhanced adaptability

### 3. Logistic Map
The logistic map equation used for chaos generation:
```
α_{n+1} = μ × α_n × (1 - α_n)
```
where μ = 4.0 for full chaotic dynamics.

## File Structure

```
.
├── problem4.py              # Original problem definition
├── pso_solver.py            # Original standard PSO implementation
├── cs_pso_solver.py         # NEW: CS-PSO implementation
├── multithreaded.py         # Original multi-threaded pipeline
├── multithreaded_cs_pso.py  # NEW: Multi-threaded pipeline with CS-PSO
├── comparison.py            # NEW: Side-by-side comparison tool
└── README.md                # This file
```

## Usage

### Running CS-PSO Multi-threaded Pipeline

```bash
python multithreaded_cs_pso.py
```

This will:
1. Initialize 5 parallel CS-PSO threads
2. Each thread uses chaotic initialization for better diversity
3. Particles are updated using chaotic perturbation
4. Ring topology communication between threads
5. Results saved to `results/cs_pso_log_[timestamp].txt`

### Running Comparison Test

```bash
python comparison.py
```

This will:
1. Run both Standard PSO and CS-PSO multiple times
2. Compare performance metrics:
   - Solution quality (fitness)
   - Correctness (matching truth table)
   - Simplicity (number of gates used)
   - Convergence speed
   - Computation time
3. Determine which algorithm performs better

## Implementation Details

### CS-PSO Solver (`cs_pso_solver.py`)

#### Key Components:

1. **ChaoticParticle Class**
   - Extends standard particle with chaotic variables
   - Each gene has its own chaotic variable for perturbation

2. **chaos_initialize_particle()**
   - Uses logistic map to generate initial positions
   - Maps chaotic values [0,1] to valid gene ranges
   - Ensures better initial diversity

3. **chaos_perturb_position()**
   - Replaces simple clamping with chaotic exploration
   - Updates chaotic variable using logistic map
   - Maps to valid gene value

4. **CSPSOSolver Class**
   - Implements full CS-PSO algorithm
   - Velocity update: standard PSO formula
   - Position update: uses chaotic perturbation instead of clamping

### Multi-threaded Pipeline (`multithreaded_cs_pso.py`)

The pipeline follows the same structure as your original but uses CS-PSO:

```
1. Initialization Phase
   └─> Create N CS-PSO solvers with chaotic initialization

2. Main Loop (for each round):
   ├─> Parallel Optimization Phase
   │   └─> Each thread runs CS-PSO with chaos perturbation
   │
   ├─> Voting Phase
   │   └─> Select best solution from all threads
   │
   ├─> Stopping Conditions
   │   ├─> Perfect solution found?
   │   ├─> Champion improved?
   │   └─> Stagnation limit reached?
   │
   └─> Feedback Phase
       └─> Ring topology communication

3. Output Phase
   └─> Report best solution and statistics
```

## Key Parameters

### Logistic Map
- **μ (mu)**: 4.0 (for full chaotic dynamics)
- **α₀ (alpha_0)**: Random value in (0.01, 0.99), avoiding {0, 0.25, 0.5, 0.75, 1.0}

### PSO Parameters (same for both algorithms)
- **Population**: 1000 particles per thread
- **Threads**: 5 parallel threads
- **Inertia (w)**: Constriction coefficient method
- **Cognitive (c1)**: chi × phi1
- **Social (c2)**: chi × phi2
- **Velocity limits**: ±0.2 × (varMax - varMin)

## Expected Performance Improvements

Based on the paper (Xu et al., 2018), CS-PSO should show:

1. **Better Ergodicity**
   - More thorough exploration of search space
   - Less clustering around local optima

2. **Faster Convergence**
   - Fewer iterations to find optimal solution
   - Better initial population quality

3. **Higher Quality Solutions**
   - Better fitness values
   - Simpler circuits (fewer active gates)

4. **Avoided Premature Convergence**
   - Chaotic perturbation prevents early stagnation
   - Maintains population diversity

## Comparison with Original PSO

### Standard PSO
✓ Simple and well-understood
✓ Fast updates (no chaos overhead)
✗ Can converge prematurely
✗ May miss global optimum
✗ Random initialization can be poor

### CS-PSO
✓ Better exploration (chaos ergodicity)
✓ Avoids local optima (chaos randomness)
✓ Higher quality initialization
✓ Better final solutions
✗ Slightly more computational overhead
✗ More complex to tune

## Verification

To verify the implementation is working correctly:

1. **Check Initialization**
   - Particles should have valid chaotic variables
   - Initial positions should be diverse
   - No two particles should be identical

2. **Monitor Chaos Perturbation**
   - Chaotic variables should update each iteration
   - Out-of-bounds genes should use chaos, not simple clamping

3. **Compare Results**
   - Run comparison.py to see performance differences
   - CS-PSO should find solutions faster or with better quality

## References

Xu, X., Rong, H., Trovati, M., Liptrott, M., & Bessis, N. (2018). 
CS-PSO: chaotic particle swarm optimization algorithm for solving 
combinatorial optimization problems. 
*Soft Computing*, 22, 783-795.

## Notes

- The chaos initialization and perturbation add minimal computational overhead
- Ring topology in multi-threaded version helps maintain diversity
- Logistic map parameters (μ=4.0) are set for maximum chaos
- Initial chaotic values are carefully selected to avoid periodic behavior

## Troubleshooting

**Issue**: No performance improvement over standard PSO
- Check that chaos functions are actually being called
- Verify chaotic variables are updating each iteration
- Ensure initial values are not problematic (0, 0.25, 0.5, 0.75, 1.0)

**Issue**: Solutions are identical to standard PSO
- Verify you're using `cs_pso_solver.CSPSOSolver` not `pso_solver.BasePSOSolver`
- Check that chaos_perturb_position is being called in clamping

**Issue**: Slower performance
- Small overhead from chaos is expected
- Should be compensated by faster convergence
- Check if MAX_ITERATIONS is sufficient
