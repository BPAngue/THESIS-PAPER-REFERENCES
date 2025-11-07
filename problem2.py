import numpy as np
from dataclasses import dataclass
from enum import IntEnum

# problem definition
# matrix size
num_rows = 5
num_cols = 5

class GateType(IntEnum):
    AND = 0
    OR = 1
    NOT = 2
    WIRE = 3
    XOR = 4

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

def dynamicFitness(position, num_equal_tt, num_gates, num_no_gates, prev_cost, d_cost):
    mu_order = 0.5
    kd_gain = 0.5

    # do dynamic fitness here
    input1, input2, gate_type = decode_matrix(position)

    num_equal = 0

    # for each row in truth table
    for tt_row in range(truth_table.num_rows_tt):
        # initialize internal truth table
        in_tt = np.zeros(num_rows, dtype=int)
        in_tt[:truth_table.num_inputs] = truth_table.inputs[tt_row]

        output = np.zeros(num_rows, dtype=int)

        # evaluate for each column
        for col in range(num_cols):
            for row in range(num_rows):
                cell_idx = col * num_rows + row
                val1 = in_tt[input1[cell_idx]]
                val2 = in_tt[input2[cell_idx]]
                output[row] = evaluate_gate(gate_type[cell_idx], val1, val2)

            # update in internal truth table
            in_tt = output.copy()

        # compare outputs
        for out_idx in range(truth_table.num_outputs):
            if output[out_idx] == truth_table.outputs[tt_row, out_idx]:
                num_equal += 1
    
    num_equal_tt = num_equal

    # count gates is solution is valid
    if num_equal >= truth_table.total_outputs:
        num_gates = count_gates(input1, input2, gate_type)
        num_no_gates = (num_rows * num_cols) - num_gates
    else:
        num_gates = 0
        num_no_gates = 0

    # Dynamic fitness function
    Fs = num_equal + num_no_gates
    d_curr = (1 - mu_order) * d_cost + mu_order * (Fs - prev_cost)

    Fd = Fs + kd_gain * d_curr

    # store state
    return Fd, Fs, d_curr, num_equal_tt, num_gates, num_no_gates

def problem(position, num_equal_tt, num_gates, num_no_gates, prev_cost, d_cost):
    cost, new_prev_cost, new_d_cost, new_num_equal_tt, new_num_gates, new_num_no_gates = dynamicFitness(position, num_equal_tt, num_gates, num_no_gates, prev_cost, d_cost) # x (position), y (num_equal_tt), z (num_gates), w (num_no_gates), u (previous cost), v (derivative cost)
    return cost, new_prev_cost, new_d_cost, new_num_equal_tt, new_num_gates, new_num_no_gates

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

def evaluate_gate(gate_type, val1, val2):
    if gate_type in [GateType.AND]:
        return val1 & val2
    elif gate_type in [GateType.OR]:
        return val1 | val2
    elif gate_type in [GateType.NOT]:
        return 1 if val1 == 0 else 0
    elif gate_type in [GateType.WIRE]:
        return val1
    elif gate_type in [GateType.XOR]:
        return val1 ^ val2
    return 0

def count_gates(input1, input2, gate_type):
    gate_used = np.zeros(num_rows * num_cols, dtype=bool)

    # start from output columns
    start_output = num_rows * (num_cols - 1)
    for out_idx in range(truth_table.num_outputs):
        count_gate_recursive(start_output + out_idx, num_cols - 1, input1, input2, gate_type, gate_used)

    return np.sum(gate_used)

def count_gate_recursive(cell, col, input1, input2, gate_type, gate_used):
    # recursively count gates
    if col < 0 or cell < 0:
        return
    
    gt = gate_type[cell]
    if gt not in [GateType.WIRE]:
        if gt not in [GateType.AND, GateType.OR] or input1[cell] != input2[cell]:
            gate_used[cell] = True

    if col > 0:
        new1 = num_rows * (col - 1) + input1[cell]
        new2 = num_rows * (col - 1) + input2[cell]
        count_gate_recursive(new1, col - 1, input1, input2, gate_type, gate_used)
        count_gate_recursive(new2, col - 1, input1, input2, gate_type, gate_used)

def get_circuit_formula(position):
    input1, input2, gate_type = decode_matrix(position)
    input_names = [f"I{i}" for i in range(truth_table.num_inputs)]
    formulas = input_names + [f"N{i}" for i in range(num_rows - truth_table.num_inputs)]

    print("\n--- CIRCUIT FORMULA DERIVATION ---")
    print(f"Inputs: {', '.join(input_names)}")
    print(f"Intermediate Nodes: {', '.join(formulas[truth_table.num_inputs:])}")
    print("")

    all_nodes = []

    # for each column in the matrix
    for col in range(num_cols):
        new_formulas = []
        print(f"Column {col + 1}:")
        for row in range(num_rows):
            cell_idx = col * num_rows + row
            g = GateType(gate_type[cell_idx])
            f1 = formulas[input1[cell_idx]]
            f2 = formulas[input2[cell_idx]]

            # construct symbolic logic expression
            if g == GateType.AND:
                expr = f"({f1} AND {f2})"
            elif g == GateType.OR:
                expr = f"({f1} OR {f2})"
            elif g == GateType.NOT:
                expr = f"(NOT {f1})"
            elif g == GateType.XOR:
                expr = f"({f1} XOR {f2})"
            elif g == GateType.WIRE:
                expr = f"{f1}"
            else:
                expr = "0"

            new_formulas.append(expr)
            print(f"  N{row}: {expr}")

        # update formulas for next column
        all_nodes.append(new_formulas)
        formulas = new_formulas
        print("")

    # final output(s)
    outputs = formulas[:truth_table.num_outputs]
    print("--- FINAL OUTPUT FORMULA ---")
    for i, f in enumerate(outputs):
        print(f"Output {i}: {f}")

    return all_nodes, outputs