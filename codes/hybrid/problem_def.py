import numpy as np
from dataclasses import dataclass
from enum import IntEnum

# --------------------------------------
# Problem Definition
# --------------------------------------
num_rows = 16  # the number of *internal gates* we can use

class GateType(IntEnum):
    NOT1 = 0
    NOT2 = 1
    AND = 2
    OR = 3
    XOR = 4
    NAND = 5
    NOR = 6
    XNOR = 7

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

# example 1 in coello (3 inputs, 1 output) --> working
# num_inputs = 3
# num_outputs = 1
# inputs = np.array([
#     [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
#     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
# ])

# outputs = np.array([
#     [0], [0], [0], [1], 
#     [0], [1], [1], [0],
# ])

# # example 2 in coello (4 inputs, 1 output)
# num_inputs = 4
# num_outputs = 1
# inputs = np.array([
#     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
#     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
#     [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
#     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
# ])

# outputs = np.array([
#     [1], [1], [0], [1], 
#     [0], [0], [1], [1],
#     [1], [0], [1], [0], 
#     [0], [1], [0], [0],
# ])

# # example 3 in coello (4 inputs, 3 outputs)
# num_inputs = 4
# num_outputs = 3
# inputs = np.array([
#     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
#     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
#     [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
#     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
# ])

# outputs = np.array([
#     [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
#     [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
#     [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], 
#     [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
# ])

# # example 4.1 in reis (4-bit even parity checker) --> working
# num_inputs = 4
# num_outputs = 1
# inputs = np.array([
#     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
#     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
#     [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
#     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
# ])

# outputs = np.array([
#     [1], [0], [0], [1], 
#     [0], [1], [1], [0],
#     [0], [1], [1], [0], 
#     [1], [0], [0], [1],
# ])

# # example 4.2 in reis (4-bit even and odd parity checker) --> working
# num_inputs = 4
# num_outputs = 2
# inputs = np.array([
#     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
#     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
#     [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
#     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
# ])

# outputs = np.array([
#     [1, 0], [0, 1], [0, 1], [1, 0], 
#     [0, 1], [1, 0], [1, 0], [0, 1],
#     [0, 1], [1, 0], [1, 0], [0, 1], 
#     [1, 0], [0, 1], [0, 1], [1, 0],
# ])

# example 5 in reis (1-bit full adder with Cin) --> working
num_inputs = 3
num_outputs = 2
inputs = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
])

outputs = np.array([
    [0, 0], [1, 0], [1, 0], [0, 1], 
    [1, 0], [0, 1], [0, 1], [1, 1],
])

# # example 6 in reis (2-bit multiplier)
# num_inputs = 4
# num_outputs = 4
# inputs = np.array([
#     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
#     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
#     [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
#     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
# ])

# outputs = np.array([
#     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
#     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
#     [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], 
#     [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 0, 1]
# ])

# # chatgpt example
# num_inputs = 5
# num_outputs = 4
# inputs = np.array([
#     [0,0,0,0,0],
#     [0,0,0,0,1],
#     [0,0,0,1,0],
#     [0,0,0,1,1],
#     [0,0,1,0,0],
#     [0,0,1,0,1],
#     [0,0,1,1,0],
#     [0,0,1,1,1],
#     [0,1,0,0,0],
#     [0,1,0,0,1],
#     [0,1,0,1,0],
#     [0,1,0,1,1],
#     [0,1,1,0,0],
#     [0,1,1,0,1],
#     [0,1,1,1,0],
#     [0,1,1,1,1],
#     [1,0,0,0,0],
#     [1,0,0,0,1],
#     [1,0,0,1,0],
#     [1,0,0,1,1],
#     [1,0,1,0,0],
#     [1,0,1,0,1],
#     [1,0,1,1,0],
#     [1,0,1,1,1],
#     [1,1,0,0,0],
#     [1,1,0,0,1],
#     [1,1,0,1,0],
#     [1,1,0,1,1],
#     [1,1,1,0,0],
#     [1,1,1,0,1],
#     [1,1,1,1,0],
#     [1,1,1,1,1],
# ])

# outputs = np.array([
#     [0,0,0,0],
#     [0,0,1,1],
#     [0,0,1,1],
#     [0,1,0,0],
#     [0,0,1,1],
#     [0,1,0,0],
#     [0,1,0,0],
#     [0,1,1,1],
#     [0,0,1,1],
#     [0,1,0,0],
#     [0,1,0,0],
#     [0,1,1,1],
#     [0,1,0,0],
#     [0,1,1,1],
#     [0,1,1,1],
#     [1,0,0,0],
#     [0,0,1,1],
#     [0,1,0,0],
#     [0,1,0,0],
#     [0,1,1,1],
#     [0,1,0,0],
#     [0,1,1,1],
#     [0,1,1,1],
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,1,1,1],
#     [0,1,1,1],
#     [1,0,0,0],
#     [0,1,1,1],
#     [1,0,0,0],
#     [1,0,0,0],
#     [1,0,1,1],
# ])

truth_table = TruthTable(num_inputs, num_outputs, inputs, outputs)

# --------------------------------------
# Utility Functions
# --------------------------------------
def decode_particle(x):
    """Decode a 1D position array into a circuit_matrix and output_array."""
    circuit_matrix_flat = x[0 : num_rows * 3]
    output_array_flat = x[num_rows * 3 :]
    
    circuit_matrix = circuit_matrix_flat.reshape((num_rows, 3))
    output_array = output_array_flat
    
    return circuit_matrix, output_array

def evaluate_gate(gate_type, val1, val2):
    """Evaluate the output of a gate for given input values."""
    if gate_type == GateType.NOT1:
        return 1 if val1 == 0 else 0
    elif gate_type == GateType.NOT2:
        return 1 if val2 == 0 else 0
    if gate_type == GateType.AND:
        return val1 & val2
    elif gate_type == GateType.OR:
        return val1 | val2
    elif gate_type == GateType.XOR:
        return val1 ^ val2
    elif gate_type == GateType.NAND:
        return 1 - (val1 & val2)
    elif gate_type == GateType.NOR:
        return 1 - (val1 | val2)
    elif gate_type == GateType.XNOR:
        return 1 - (val1 ^ val2)
    return 0 

def count_active_gates(circuit_matrix, output_array):
    """
    Counts active gates by tracing back from outputs.
    This implements the "simplicity" measure.
    """
    pointers_to_check = set(output_array)
    used_pointers = set()

    while pointers_to_check:
        ptr = pointers_to_check.pop()

        if ptr in used_pointers:
            continue
        used_pointers.add(ptr)

        # If it's a primary input, stop tracing this branch
        if ptr <= num_inputs:
            continue

        # If it's a gate, find its inputs and add them to the check list
        gate_index = ptr - num_inputs - 1

        # Check if the pointer is a valid gate index
        if 0 <= gate_index < num_rows:
            gate_row = circuit_matrix[gate_index]
            ptr_in1 = gate_row[0]
            gate_type = gate_row[1]
            ptr_in2 = gate_row[2]
            
            pointers_to_check.add(ptr_in1)

            # NOT gate only uses one input
            if GateType(gate_type) not in (GateType.NOT1, GateType.NOT2):
                pointers_to_check.add(ptr_in2)

    # Count how many of the used pointers were gates
    active_gate_count = sum(1 for p in used_pointers if p > num_inputs)

    return active_gate_count

# --------------------------------------
# Fitness Evaluation
# --------------------------------------
def fitness_function(position):
    circuit_matrix, output_array = decode_particle(position)
    
    total_score = 0
    signal_values = {}

    for tt_row in range(truth_table.num_rows_tt):
        signal_values.clear()
        current_inputs = truth_table.inputs[tt_row]
        for i in range(num_inputs):
            signal_values[i + 1] = current_inputs[i] 

        for i in range(num_rows):
            gate_row = circuit_matrix[i]
            ptr_in1, gate_type, ptr_in2 = gate_row
            
            val1 = signal_values.get(ptr_in1, 0)
            val2 = signal_values.get(ptr_in2, 0)
            
            output_val = evaluate_gate(gate_type, val1, val2)
            signal_values[num_inputs + i + 1] = output_val

        for out_idx in range(truth_table.num_outputs):
            output_ptr = output_array[out_idx]
            evolved_out = signal_values.get(output_ptr, 0)
            target_out = truth_table.outputs[tt_row, out_idx]
            
            if evolved_out == target_out:
                total_score += 1
            
    num_gates = count_active_gates(circuit_matrix, output_array)
    num_no_gates = num_rows - num_gates
    
    # CRITICAL FIX: Conditional Simplicity Bonus
    if total_score == truth_table.total_outputs:
        fitness = total_score + (0.1 * num_no_gates)
    else:
        fitness = total_score # Ignore gates, focus on correctness

    return fitness, total_score, num_gates, num_no_gates

def problem(position):
    return fitness_function(position)

# --------------------------------------
# Circuit Formula Derivation
# --------------------------------------
def get_circuit_formula(position):
    """Converts the best particle into a human-readable formula."""
    circuit_matrix, output_array = decode_particle(position)

    # Initialize formulas list with primary inputs
    # Note: We add a "DUMMY" at index 0 so that pointer '1' maps to index 1
    formulas = ["DUMMY"] + [f"I{i+1}" for i in range(truth_table.num_inputs)]

    print("\n--- CIRCUIT FORMULA DERIVATION ---")
    print("Internal Gates:")

    # Evaluate gates sequentially, adding them to the formulas list
    for i in range(num_rows):
        ptr_in1, gate_type_val, ptr_in2 = circuit_matrix[i]
        g = GateType(gate_type_val)
        
        # Get input formulas from the list
        f1 = formulas[ptr_in1]
        f2 = formulas[ptr_in2]

        # construct symbolic logic expression
        if g == GateType.NOT1:
            expr = f"(NOT {f1})"
        elif g == GateType.NOT2:
            expr = f"(NOT {f2})"
        elif g == GateType.AND:
            expr = f"({f1} AND {f2})"
        elif g == GateType.OR:
            expr = f"({f1} OR {f2})"
        elif g == GateType.XOR:
            expr = f"({f1} XOR {f2})"
        elif g == GateType.NAND:
            expr = f"({f1} NAND {f2})"
        elif g == GateType.NOR:
            expr = f"({f1} NOR {f2})"
        elif g == GateType.XNOR:
            expr = f"({f1} XNOR {f2})"
        else:
            expr = "UNKNOWN"
            
        print(f"  Gate {i+num_inputs+1}: {expr}")
        formulas.append(expr)

    # Get final output formulas using the output_array
    final_outputs = []
    print("\n--- FINAL OUTPUT FORMULA ---")
    for i, output_ptr in enumerate(output_array):
        f_out = formulas[output_ptr]
        print(f"  Output {i+1}: {f_out}")
        final_outputs.append(f_out)
        
    return final_outputs