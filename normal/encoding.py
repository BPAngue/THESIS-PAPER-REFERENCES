import numpy as np
from dataclasses import dataclass
from enum import IntEnum

# --------------------------------------
# Problem Definition
# --------------------------------------
iterations = 5
num_cols = 3
num_rows = 4

class GateType(IntEnum):
    AND = 0
    OR = 1
    NOT = 2
    XOR = 3

@dataclass
class TruthTable:
    num_inputs: int
    num_outputs: int
    inputs: np.ndarray
    outputs: np.ndarray

    @property
    def num_rows_tt(self):
        return len(self.inputs)
    
    @property
    def total_outputs(self):
        return self.num_rows_tt * self.num_outputs

# --------------------------------------
# Define Truth Table
# --------------------------------------     
inputs = np.array([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
])

outputs = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
    [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
])

num_inputs = len(inputs[0])
num_outputs = len(outputs[0])
print(f"Num inputs: {num_inputs}")
print(f"Num outputs: {num_outputs}")

truth_table = TruthTable(num_inputs, num_outputs, inputs, outputs)

# --------------------------------------
# Encoding Variables
# -------------------------------------- 
def encoding_scheme():
    print("Initializing Circuit Matrix with Combinational-Only Restriction...")
    # Create an empty matrix to fill
    circuit_matrix = np.zeros((num_rows, num_cols), dtype=int)

    # Note: Using len(GateType) for the high end of the gate type
    # is a good practice, as it adapts if you add/remove gates.
    gate_type_high_exclusive = len(GateType)

    for i in range(num_rows):
        # This is the core of the restriction:
        # The allowed high-end for an input at row 'i' is (num_inputs + i)
        # This means it can only see primary inputs (1 to num_inputs)
        # and gate outputs *before* it (num_inputs + 1 to num_inputs + i)
        
        input_low = 1
        # We add +1 because np.random.randint's high parameter is exclusive
        input_high_exclusive = num_inputs + i + 1 
        
        # Manually set the values for each cell in the row
        
        # Col 0: input1
        circuit_matrix[i, 0] = np.random.randint(input_low, input_high_exclusive)
        
        # Col 1: gate type
        circuit_matrix[i, 1] = np.random.randint(0, gate_type_high_exclusive)
        
        # Col 2: input2
        circuit_matrix[i, 2] = np.random.randint(input_low, input_high_exclusive)

        print(f"  Row {i}: Inputs can be from range [1, {input_high_exclusive - 1}]")

    # --------------------------------------
    # --- NEW SECTION: OUTPUT ARRAY ---
    # --------------------------------------
    print("Initializing Output Array...")

    # 1. Define the allowed range for an output pointer.
    #    It can be any primary input (1 to num_inputs)
    #    or any gate output (num_inputs + 1 to num_inputs + num_rows).
    output_pointer_low = 1
    output_pointer_high_inclusive = num_inputs + num_rows

    # Add +1 because np.random.randint's high parameter is exclusive
    output_pointer_high_exclusive = output_pointer_high_inclusive + 1

    # 2. Get the required size of the array from the truth table.
    output_array_size = truth_table.num_outputs

    # 3. Create the randomly initialized output_array.
    output_array = np.random.randint(
        low=output_pointer_low, 
        high=output_pointer_high_exclusive, 
        size=(output_array_size)
    )

    print(f"Allowed output pointer range: [{output_pointer_low}, {output_pointer_high_inclusive}]")

    return circuit_matrix, output_array

def decode_formula(circuit_matrix, output_array):
    return 0