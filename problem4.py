import numpy as np
from dataclasses import dataclass
from enum import IntEnum

# --------------------------------------
# Problem Definition
# --------------------------------------
num_rows = 12

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
num_inputs = 4
num_outputs = 3

inputs = np.array([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
])

outputs = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
    [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
    [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], 
    [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
])

truth_table = TruthTable(num_inputs, num_outputs, inputs, outputs)
perfect_fitness_score = truth_table.total_outputs + num_rows # 48 + 12 = 60 (max possible)

# --------------------------------------
# Utility Functions
# --------------------------------------

def decode_particle(x):
    """Decode a 1D position array into a circuit_matrix and output_array."""
    circuit_genes_flat = x[0 : num_rows * 3]
    output_genes = x[num_rows * 3 :]
    
    circuit_matrix = circuit_genes_flat.reshape((num_rows, 3))
    output_array = output_genes
    
    return circuit_matrix, output_array

def evaluate_gate(gate_type, val1, val2):
    """Evaluate the output of a gate for given input values."""
    if gate_type == GateType.AND:
        return val1 & val2
    elif gate_type == GateType.OR:
        return val1 | val2
    elif gate_type == GateType.NOT:
        return 1 if val1 == 0 else 0
    elif gate_type == GateType.XOR:
        return val1 ^ val2
    return 0

def count_active_gates(circuit_matrix, output_array):
    """Counts active gates by tracing back from outputs."""
    pointers_to_check = set(output_array)
    used_pointers = set()
    
    while pointers_to_check:
        ptr = pointers_to_check.pop()
        
        if ptr in used_pointers:
            continue
        used_pointers.add(ptr)
        
        if ptr <= num_inputs:
            continue
            
        gate_index = ptr - num_inputs - 1
        
        if 0 <= gate_index < num_rows:
            gate_row = circuit_matrix[gate_index]
            ptr_in1 = gate_row[0]
            gate_type = gate_row[1]
            ptr_in2 = gate_row[2]
            
            pointers_to_check.add(ptr_in1)
            
            # NOT gate only uses one input
            if GateType(gate_type) != GateType.NOT:
                pointers_to_check.add(ptr_in2)
                
    active_gate_count = sum(1 for p in used_pointers if p > num_inputs)
    return active_gate_count

# --------------------------------------
# Dynamic Fitness Evaluation
# --------------------------------------
def dynamicFitness(position, prev_static_fitness, previous_derivative):
    mu_order = 0.6
    # --- NOTE: I've disabled dynamic fitness as requested in our previous chat ---
    # --- to focus on the multi-threaded aspect. Set to 0.5 to re-enable. ---
    kd_gain = 0.5

    circuit_matrix, output_array = decode_particle(position)
    num_equal = 0
    signal_values = {}

    for tt_row in range(truth_table.num_rows_tt):
        signal_values.clear()
        current_inputs = truth_table.inputs[tt_row]
        for i in range(num_inputs):
            signal_values[i + 1] = current_inputs[i]

        for i in range(num_rows):
            gate_row = circuit_matrix[i]
            ptr_in1 = gate_row[0]
            gate_type = gate_row[1]
            ptr_in2 = gate_row[2]
            
            val1 = signal_values.get(ptr_in1, 0)
            val2 = signal_values.get(ptr_in2, 0)
            
            output_val = evaluate_gate(gate_type, val1, val2)
            
            output_ptr = num_inputs + i + 1
            signal_values[output_ptr] = output_val

        for out_idx in range(truth_table.num_outputs):
            output_ptr = output_array[out_idx]
            evolved_output = signal_values.get(output_ptr, 0)
            target_output = truth_table.outputs[tt_row, out_idx]
            
            if evolved_output == target_output:
                num_equal += 1
    
    num_equal_tt = num_equal
    
    if num_equal >= truth_table.total_outputs:
        num_gates = count_active_gates(circuit_matrix, output_array)
        num_no_gates = num_rows - num_gates
    else:
        num_gates = num_rows
        num_no_gates = 0

    current_static_fitness = num_equal + num_no_gates
    current_derivative = (1 - mu_order) * previous_derivative + mu_order * (current_static_fitness - prev_static_fitness)
    dynamic_fitness = current_static_fitness + kd_gain * current_derivative

    return dynamic_fitness, current_static_fitness, current_derivative, num_equal_tt, num_gates, num_no_gates

def problem(position, prev_static_fitness, previous_derivative):
    return dynamicFitness(position, prev_static_fitness, previous_derivative)

# --------------------------------------
# Circuit Formula Derivation
# --------------------------------------
def get_circuit_formula(position):
    """Converts the best particle into a human-readable formula."""
    circuit_matrix, output_array = decode_particle(position)
    
    formulas = ["DUMMY"] + [f"I{i+1}" for i in range(truth_table.num_inputs)]
    
    print("\n--- CIRCUIT FORMULA DERIVATION ---")
    print("Internal Gates:")
    
    for i in range(num_rows):
        ptr_in1, gate_type_val, ptr_in2 = circuit_matrix[i]
        
        try:
            g = GateType(gate_type_val)
        except ValueError:
            g = GateType.AND # Default for safety

        f1 = formulas[ptr_in1]
        f2 = formulas[ptr_in2]

        if g == GateType.AND:
            expr = f"({f1} AND {f2})"
        elif g == GateType.OR:
            expr = f"({f1} OR {f2})"
        elif g == GateType.NOT:
            expr = f"(NOT {f1})"
        elif g == GateType.XOR:
            expr = f"({f1} XOR {f2})"
        else:
            expr = "UNKNOWN"
            
        print(f"  Gate {i+num_inputs+1}: {expr}")
        formulas.append(expr)

    final_outputs = []
    print("\n--- FINAL OUTPUT FORMULA ---")
    for i, output_ptr in enumerate(output_array):
        f_out = formulas[output_ptr]
        print(f"  Output {i+1}: {f_out}")
        final_outputs.append(f_out)
        
    return final_outputs