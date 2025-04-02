from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class DecompositionRule(OrderingRule):
    def __init__(self, dec_parser):
        """
        Initialize with a DecFileParser instance
        """
        self.parser = dec_parser
        self.var_block_assignment = None
        self.constr_block_assignment = None
        self.initialized = False

    def _initialize_assignments(self, constraints, vars, A_csc):
        """
        Initialize block assignments for variables and constraints
        """
        # Build constraint name to index mapping
        constr_name_to_idx = {constr.ConstrName: idx for idx, constr in enumerate(constraints)}
        
        # Assign constraints to blocks
        self.constr_block_assignment = np.zeros(len(constraints), dtype=int)
        current_block = 0
        for block in self.parser.blocks:
            for name in block:
                if name in constr_name_to_idx:
                    self.constr_block_assignment[constr_name_to_idx[name]] = current_block
            current_block += 1
        
        # Master constraints get highest block number
        for name in self.parser.master:
            if name in constr_name_to_idx:
                self.constr_block_assignment[constr_name_to_idx[name]] = current_block

        # Classify variables using the CSC matrix
        self.var_block_assignment = self._classify_variables(vars, constraints, A_csc)
        self.initialized = True

    def _classify_variables(self, vars, constraints, A_csc):
        """
        Classify variables into blocks based on which constraints they appear in
        """
        # Build constraint index to block mapping
        constr_block = {}
        for block_idx, block in enumerate(self.parser.blocks):
            for name in block:
                for constr_idx, constr in enumerate(constraints):
                    if constr.ConstrName == name:
                        constr_block[constr_idx] = block_idx
        for name in self.parser.master:
            for constr_idx, constr in enumerate(constraints):
                if constr.ConstrName == name:
                    constr_block[constr_idx] = self.parser.nblocks  # Master block

        # Classify variables using the CSC matrix
        var_block = np.zeros(len(vars), dtype=int)
        
        # Get CSC matrix components
        indptr = A_csc.indptr
        indices = A_csc.indices
        
        for var_idx in range(len(vars)):
            blocks_in = set()
            # Iterate through all constraints this variable appears in
            for i in range(indptr[var_idx], indptr[var_idx+1]):
                constr_idx = indices[i]
                if constr_idx in constr_block:
                    block = constr_block[constr_idx]
                    if block != self.parser.nblocks:  # Not master
                        blocks_in.add(block)
            
            if len(blocks_in) == 1:
                var_block[var_idx] = blocks_in.pop()
            else:
                var_block[var_idx] = self.parser.nblocks  # Linking variables
        
        return var_block

    def _reset(self):
        """
        Reset the initialization state
        """
        self.var_block_assignment = None
        self.constr_block_assignment = None
        self.initialized = False

    # The following methods maintain compatibility with the interface
    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        if not self.initialized:
            self._initialize_assignments(constraints, vars, A_csc)
        return self.var_block_assignment

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        if not self.initialized:
            self._initialize_assignments(constraints, vars, A_csc)
        return self.constr_block_assignment
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        if not self.initialized:
            self._initialize_assignments(constraints, vars, A_csc)
        # Check if index is within bounds
        if idx >= len(self.var_block_assignment):
            raise IndexError(f"Variable index {idx} is out of bounds for array of size {len(self.var_block_assignment)}")
        return (self.var_block_assignment[idx],)
    
    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        if not self.initialized:
            self._initialize_assignments(constraints, vars, A_csc)
        # Check if index is within bounds
        if idx >= len(self.constr_block_assignment):
            raise IndexError(f"Constraint index {idx} is out of bounds for array of size {len(self.constr_block_assignment)}")
        return (self.constr_block_assignment[idx],)
    
    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block based on the decomposition structure from the .dec file.
        Always resets and reinitializes before processing.
        """
        # Reset to ensure fresh initialization
        self._reset()
        
        # Initialize with fresh data
        self._initialize_assignments(constraints, vars, A_csc)
        
        # Validate indices to prevent out of bounds error
        valid_var_indices = var_indices[var_indices < len(self.var_block_assignment)]
        valid_constr_indices = constr_indices[constr_indices < len(self.constr_block_assignment)]
        
        # Check if we lost any indices
        if len(valid_var_indices) < len(var_indices):
            print(f"Warning: {len(var_indices) - len(valid_var_indices)} variable indices were out of bounds")
        if len(valid_constr_indices) < len(constr_indices):
            print(f"Warning: {len(constr_indices) - len(valid_constr_indices)} constraint indices were out of bounds")
        
        # Get the relevant subset of constraint assignments
        sub_constr_scores = self.constr_block_assignment[valid_constr_indices]
        
        # Group constraints by their block assignments
        unique_scores = np.unique(sub_constr_scores)
        constr_groups = {}
        for score in unique_scores:
            mask = (sub_constr_scores == score)
            constr_groups[score] = valid_constr_indices[mask]
        
        # Group variables by their block assignments (within the current var_indices)
        sub_var_scores = self.var_block_assignment[valid_var_indices]
        var_unique_scores = np.unique(sub_var_scores)
        var_groups = {}
        for score in var_unique_scores:
            mask = (sub_var_scores == score)
            var_groups[score] = valid_var_indices[mask]
        
        # Create partition map by combining variable and constraint groups
        partition_map = {}
        label = 0
        
        # First handle regular blocks (0 to nblocks-1)
        for block in range(self.parser.nblocks):
            if block in constr_groups or block in var_groups:
                # Get variables and constraints for this block (may be empty)
                block_vars = var_groups.get(block, np.array([], dtype=int))
                block_constrs = constr_groups.get(block, np.array([], dtype=int))
                
                # Only add if at least one variable or constraint exists
                if len(block_vars) > 0 or len(block_constrs) > 0:
                    partition_map[label] = (block_vars, block_constrs)
                    label += 1
        
        # Then handle master/linking components
        master_vars = []
        master_constrs = []
        
        # Collect all master/linking variables
        for score in var_unique_scores:
            if score >= self.parser.nblocks:
                master_vars.append(var_groups.get(score, np.array([], dtype=int)))
        
        # Collect all master constraints
        for score in unique_scores:
            if score >= self.parser.nblocks:
                master_constrs.append(constr_groups.get(score, np.array([], dtype=int)))
        
        # Combine if we have any master components
        if len(master_vars) > 0 or len(master_constrs) > 0:
            # Handle empty arrays safely
            final_master_vars = np.concatenate([v for v in master_vars if v.size > 0]) if any(v.size > 0 for v in master_vars) else np.array([], dtype=int)
            final_master_constrs = np.concatenate([c for c in master_constrs if c.size > 0]) if any(c.size > 0 for c in master_constrs) else np.array([], dtype=int)
            
            if final_master_vars.size > 0 or final_master_constrs.size > 0:
                partition_map[label] = (final_master_vars, final_master_constrs)
        
        return partition_map