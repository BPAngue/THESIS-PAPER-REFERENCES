"""
Robust clamping utilities for particle positions
Handles NaN, Inf, and overflow cases
"""
import numpy as np
import problem_def as problem_def

def clamp_particle_position(position_float):
    """
    Robustly clamps a particle's position vector to valid integer ranges.
    Handles NaN, Inf, and overflow cases gracefully.
    """
    # Handle NaN and Inf values first
    position_clean = np.copy(position_float)
    
    # Replace NaN with random valid values
    nan_mask = np.isnan(position_clean)
    if np.any(nan_mask):
        # Generate random values for NaN positions
        for i in np.where(nan_mask)[0]:
            position_clean[i] = np.random.uniform(-10, 10)
    
    # Replace Inf with large but manageable values
    inf_mask = np.isinf(position_clean)
    if np.any(inf_mask):
        position_clean[inf_mask] = np.sign(position_clean[inf_mask]) * 100
    
    # Now round to integers
    x = np.round(position_clean).astype(np.int64)  # Use int64 to avoid overflow
    
    idx = 0
    num_inputs = problem_def.truth_table.num_inputs
    num_rows = problem_def.num_rows
    num_outputs = problem_def.truth_table.num_outputs
    gate_type_high = len(problem_def.GateType)

    # 1. Clamp Circuit Matrix genes
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        
        # Clamp Input 1 (Col 0) - must be in range [1, input_high_inclusive]
        if x[idx] < 1:
            x[idx] = 1
        elif x[idx] > input_high_inclusive:
            x[idx] = input_high_inclusive
        idx += 1
        
        # Clamp Gate Type (Col 1) - must be in range [0, gate_type_high-1]
        if x[idx] < 0:
            x[idx] = 0
        elif x[idx] >= gate_type_high:
            x[idx] = gate_type_high - 1
        idx += 1
        
        # Clamp Input 2 (Col 2) - must be in range [1, input_high_inclusive]
        if x[idx] < 1:
            x[idx] = 1
        elif x[idx] > input_high_inclusive:
            x[idx] = input_high_inclusive
        idx += 1

    # 2. Clamp Output Array genes
    output_high_inclusive = num_inputs + num_rows
    for _ in range(num_outputs):
        # Must be in range [1, output_high_inclusive]
        if x[idx] < 1:
            x[idx] = 1
        elif x[idx] > output_high_inclusive:
            x[idx] = output_high_inclusive
        idx += 1
    
    # Convert back to standard int to avoid any issues
    return x.astype(int)

def create_random_particle(varsize):
    """Creates a 1D particle vector with valid, random integers for each gene."""
    particle = np.zeros(varsize, dtype=int)
    idx = 0
    num_inputs = problem_def.truth_table.num_inputs
    num_rows = problem_def.num_rows
    num_outputs = problem_def.truth_table.num_outputs
    gate_type_high = len(problem_def.GateType)

    # 1. Create Circuit Matrix (num_rows * 3)
    for i in range(num_rows):
        input_high_inclusive = num_inputs + i
        input_high_exclusive = input_high_inclusive + 1

        # Input 1 (Col 0)
        particle[idx] = np.random.randint(1, input_high_exclusive)
        idx += 1

        # Gate Type (Col 1)
        particle[idx] = np.random.randint(0, gate_type_high)
        idx += 1

        # Input 2 (Col 2)
        particle[idx] = np.random.randint(1, input_high_exclusive)
        idx += 1

    # 2. Create Output Array (num_outputs)
    output_high_inclusive = num_inputs + num_rows
    output_high_exclusive = output_high_inclusive + 1

    for _ in range(num_outputs):
        particle[idx] = np.random.randint(1, output_high_exclusive)
        idx += 1

    return particle