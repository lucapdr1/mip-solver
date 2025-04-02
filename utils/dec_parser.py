import os
from collections import defaultdict

class DecFileParser:
    def __init__(self, mps_file_path):
        dec_file_path = self._get_dec_file_path(mps_file_path)
        self.blocks = []  # List of lists of constraint names for each block
        self.master = []  # List of constraint names in MASTERCONSS
        self.nblocks = 0
        self._parse_dec_file(dec_file_path)

    def _get_dec_file_path(self, mps_file_path):
        """Derives the .dec file path from the .mps file path."""
        return os.path.splitext(mps_file_path)[0] + ".dec"
    
    def _parse_dec_file(self, path):
        current_block = None
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                
                # Skip empty lines and comments
                if not stripped or stripped.startswith('\\'):
                    i += 1
                    continue
                    
                # Handle PRESOLVED (with value on same line or next line)
                if stripped.upper().startswith('PRESOLVED'):
                    # Check if value is on same line
                    parts = stripped.split()
                    if len(parts) > 1:
                        # Has value on same line (e.g., "PRESOLVED 0")
                        try:
                            presolved_val = int(parts[1])
                            i += 1
                            continue
                        except ValueError:
                            raise ValueError(f"Invalid PRESOLVED value: {parts[1]}")
                    else:
                        # Value is on next line
                        if i + 1 >= len(lines):
                            raise ValueError("PRESOLVED missing value")
                        next_line = lines[i+1].strip()
                        try:
                            presolved_val = int(next_line)
                            i += 2
                            continue
                        except ValueError:
                            raise ValueError(f"Invalid PRESOLVED value: {next_line}")
                    
                # Handle NBLOCKS (with value on same line or next line)
                if stripped.upper() == 'NBLOCKS':
                    # Check if value is on same line
                    parts = stripped.split()
                    if len(parts) > 1:
                        # Has value on same line (e.g., "NBLOCKS 24")
                        try:
                            self.nblocks = int(parts[1])
                            i += 1
                            continue
                        except ValueError:
                            raise ValueError(f"Invalid NBLOCKS value: {parts[1]}")
                    else:
                        # Value is on next line
                        if i + 1 >= len(lines):
                            raise ValueError("NBLOCKS missing value")
                        next_line = lines[i+1].strip()
                        try:
                            self.nblocks = int(next_line)
                            i += 2
                            continue
                        except ValueError:
                            raise ValueError(f"Invalid NBLOCKS value: {next_line}")
                    
                # Handle BLOCK declaration
                if stripped.upper().startswith('BLOCK'):
                    parts = stripped.split()
                    if len(parts) < 2:
                        raise ValueError("BLOCK line is malformed - missing block number")
                    try:
                        block_num = int(parts[1])
                    except ValueError:
                        raise ValueError(f"Invalid block number: {parts[1]}")
                        
                    if block_num != len(self.blocks) + 1:  # Blocks are 1-indexed
                        raise ValueError(f"Block number {block_num} out of order")
                        
                    current_block = []
                    self.blocks.append(current_block)
                    i += 1
                    continue
                    
                # Handle MASTERCONSS
                if stripped.upper().startswith('MASTERCONSS'):
                    current_block = self.master
                    i += 1
                    continue
                    
                # Handle constraint names
                if current_block is not None:
                    current_block.append(stripped)
                    i += 1
                else:
                    # Skip unexpected lines outside block/master sections
                    i += 1
        
        # Validate we got the expected number of blocks
        if len(self.blocks) != self.nblocks:
            raise ValueError(f"Expected {self.nblocks} blocks, found {len(self.blocks)}")

    def get_permutations(self, model):
        # Build constraint name to index mapping
        constr_name_to_idx = {}
        for idx, constr in enumerate(model.getConstrs()):
            name = constr.ConstrName
            if not name:
                raise ValueError(f"Constraint at index {idx} has no name")
            if name in constr_name_to_idx:
                raise ValueError(f"Duplicate constraint name: {name}")
            constr_name_to_idx[name] = idx

        # Generate constraint permutation - MASTER FIRST, then blocks
        constr_perm = []
        
        # Add master constraints first
        for name in self.master:
            if name not in constr_name_to_idx:
                raise ValueError(f"Constraint {name} not found in model")
            constr_perm.append(constr_name_to_idx[name])
            
        # Then add all block constraints
        for block in self.blocks:
            for name in block:
                if name not in constr_name_to_idx:
                    raise ValueError(f"Constraint {name} not found in model")
                constr_perm.append(constr_name_to_idx[name])

        # Build variable to constraints mapping
        var_to_constrs = defaultdict(list)
        for constr in model.getConstrs():
            row = model.getRow(constr)
            for i in range(row.size()):
                var = row.getVar(i)
                var_to_constrs[var].append(constr)

        # Determine block for each constraint
        constr_block = {}
        # Master constraints are identified as 'master'
        for name in self.master:
            constr_block[name] = 'master'
        # Regular block constraints
        for block_idx, block in enumerate(self.blocks):
            for name in block:
                constr_block[name] = block_idx

        # Determine block for each variable
        var_block = {}
        for var in model.getVars():
            blocks_in = set()
            for constr in var_to_constrs[var]:
                name = constr.ConstrName
                if name not in constr_block:
                    raise ValueError(f"Constraint {name} not in .dec file")
                block = constr_block[name]
                if block != 'master':
                    blocks_in.add(block)
            if len(blocks_in) == 1:
                var_block[var] = blocks_in.pop()
            else:
                var_block[var] = 'linking'

        # Collect master variables
        master_vars = set()
        for constr_name in self.master:
            constr = model.getConstrByName(constr_name)
            if not constr:
                raise ValueError(f"Constraint {constr_name} not found")
            row = model.getRow(constr)
            for i in range(row.size()):
                var = row.getVar(i)
                master_vars.add(var)
                
        # Collect variables for each block in order of their first occurrence
        block_vars = [[] for _ in range(self.nblocks)]
        for block_idx, block_constr_names in enumerate(self.blocks):
            seen_vars = set()
            for constr_name in block_constr_names:
                constr = model.getConstrByName(constr_name)
                if not constr:
                    raise ValueError(f"Constraint {constr_name} not found")
                row = model.getRow(constr)
                for i in range(row.size()):
                    var = row.getVar(i)
                    if var_block.get(var) == block_idx and var not in seen_vars:
                        block_vars[block_idx].append(var)
                        seen_vars.add(var)

        # Collect linking variables
        linking_vars = [var for var in model.getVars() if var_block.get(var) == 'linking']

        # Generate variable permutation - master variables first
        var_perm = []
        var_to_idx = {var: idx for idx, var in enumerate(model.getVars())}
        
        # Get variables that only appear in master constraints
        master_only_vars = [var for var in master_vars if var_block.get(var) not in range(self.nblocks)]
        
        # Add master-only variables first
        for var in master_only_vars:
            if var not in linking_vars:  # Avoid duplication with linking vars
                var_perm.append(var_to_idx[var])
        
        # Add block variables
        for block_idx in range(self.nblocks):
            for var in block_vars[block_idx]:
                var_perm.append(var_to_idx[var])
                
        # Add linking variables last
        for var in linking_vars:
            var_perm.append(var_to_idx[var])

        return constr_perm, var_perm