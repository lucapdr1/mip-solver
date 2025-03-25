import gurobipy as gp
import numpy as np

def parse_dec_file(filepath):
    """
    Parses a .dec file and returns a dictionary with keys:
      - "blocks": a dict mapping block id (as string) to a list of constraint names
      - "master": a list of constraint names
    """
    order = {"blocks": {}, "master": []}
    current_section = None
    current_block = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # skip empty lines and lines starting with a comment marker (e.g. "\")
            if not line or line.startswith('\\'):
                continue
            # recognize section keywords (case insensitive)
            upper = line.upper()
            if upper == "NBLOCKS":
                current_section = "nblocks"
                continue
            elif upper.startswith("BLOCK"):
                # e.g. "BLOCK 1"
                parts = line.split()
                if len(parts) >= 2:
                    current_block = parts[1]
                    order["blocks"][current_block] = []
                    current_section = "block"
                continue
            elif upper == "MASTERCONSS":
                current_section = "master"
                continue
            # Skip other key words that are not used here.
            elif upper in {"PRESOLVED", "CONSDEFAULTMASTER", "BLOCKVARS", "MASTERVARS", "LINKINGVARS"}:
                current_section = None
                current_block = None
                continue
            # According to current section, add the line value
            if current_section == "block" and current_block:
                order["blocks"][current_block].append(line)
            elif current_section == "master":
                order["master"].append(line)
    return order

def constraint_permutation_from_dec(model, dec_file):
    """
    Given a GCG .dec file and a Gurobi model,
    returns a permutation array for constraints such that
    reordering the model's constraints by this permutation yields
    the order specified in the .dec file.
    """
    # Parse the .dec file to get ordering info.
    order = parse_dec_file(dec_file)
    
    # Build the desired ordering:
    # First, list constraints in each block (ordered by block number)
    ordered_constr_names = []
    for block_id in sorted(order["blocks"].keys(), key=lambda x: int(x)):
        ordered_constr_names.extend(order["blocks"][block_id])
    # Then, add master constraints.
    ordered_constr_names.extend(order["master"])
    
    # Append any constraints not mentioned in the dec file (in original order)
    dec_names = set(ordered_constr_names)
    remaining = [c.ConstrName for c in model.getConstrs() if c.ConstrName not in dec_names]
    complete_order = ordered_constr_names + remaining

    # Build a mapping from constraint name to its original index.
    orig_constrs = model.getConstrs()
    name_to_index = {constr.ConstrName: idx for idx, constr in enumerate(orig_constrs)}

    # Create the permutation array: new_order[i] = original index of the constraint that should appear at position i.
    perm = []
    for name in complete_order:
        if name in name_to_index:
            perm.append(name_to_index[name])
        else:
            print(f"Warning: Constraint '{name}' not found in the model.")
    return np.array(perm)

