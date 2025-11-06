import numpy as np
from dataclasses import dataclass

# problem definition
num_rows = 5
num_cols = 5

@dataclass
class TruthTable:
    num_inputs: int
    num_outputs: int
    inputs: np.ndarray
    outputs: np.ndarray

# define truth table
num_inputs = 3
num_outputs = 1

inputs = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
])

outputs = np.array([
    [0], [0], [0], [1], [0], [1], [1], [0]
])

truth_table = TruthTable(num_inputs, num_outputs, inputs, outputs)

def dynamicFitness(x):
    # do dynamic fitness here
    input1, input2, gate_type = decode_matrix(x)

    print(x)  
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"Gate Type: {gate_type}")
    print()

    return 0

def problem(x):
    fitness = dynamicFitness(x)
    return fitness

def decode_matrix(x):
    n_cells = num_rows * num_cols
    input1 = np.zeros(n_cells, dtype=int)
    input2 = np.zeros(n_cells, dtype=int)
    gate_type = np.zeros(n_cells, dtype=int)

    idx = 0
    for i in range(n_cells):
        input1[i] = int(x[idx]) % num_rows
        input2[i] = int(x[idx + 1]) % num_rows
        gate_type[i] = int(x[idx + 2]) % 5
        idx += 3

    return input1, input2, gate_type