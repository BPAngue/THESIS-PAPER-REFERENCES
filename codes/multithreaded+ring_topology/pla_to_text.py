from itertools import product

def expand_pattern(pattern):
    """
    Expand a PLA pattern with dashes (-) into all fully specified binary strings.
    """
    dash_indices = [i for i, c in enumerate(pattern) if c == '-']
    combos = product("01", repeat=len(dash_indices))
    
    expanded = []
    for combo in combos:
        chars = list(pattern)
        for idx, bit in zip(dash_indices, combo):
            chars[idx] = bit
        expanded.append("".join(chars))
    return expanded

def pla_to_txt(pla_file, txt_file):
    truth_table = []

    with open(pla_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip headers and comments
            if not line or line.startswith('.') or line.startswith('#'):
                continue

            # Split input and output
            parts = line.split()
            if len(parts) != 2:
                continue
            input_pattern, output_pattern = parts

            # Expand dashes in input
            expanded_inputs = expand_pattern(input_pattern)

            # Append each expanded input with its output
            for inp in expanded_inputs:
                truth_table.append((inp, output_pattern))

    # Sort by input value for consistency
    truth_table.sort(key=lambda x: int(x[0], 2))

    # Write to txt in requested schema
    with open(txt_file, 'w') as f_out:
        f_out.write("inputs ; outputs\n")
        for inp, out in truth_table:
            f_out.write(f"{inp} ; {out}\n")

    print(f"PLA converted to {txt_file} successfully!")

# ---------------------------
# Example usage
# ---------------------------
pla_file = "truth.pla"     # replace with your PLA file path
txt_file = "truth_table.txt" # desired output file path
pla_to_txt(pla_file, txt_file)
