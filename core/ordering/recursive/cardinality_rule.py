from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class CardinalityRule(OrderingRule):
    """
    Assigns scores based on the number of nonzero coefficients (cardinality of
    each column/row). The rule is scale-invariant because it considers only
    the count of nonzero entries rather than their magnitudes.

    - score_variables(...) returns the number of nonzeros in the column of each variable.
    - score_constraints(...) returns the number of nonzeros in the row of each constraint.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        """
        :param scaling: A scaling factor applied to the count.
        :param tol: A tolerance below which a value is considered zero.
        """
        self.scaling = scaling  
        self.tol = tol

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Calculate the number of nonzero coefficients in each variable's column.
        Handles both dense and sparse matrices.
        """
        num_constraints, num_vars = A.shape  # Get dimensions
        scores = np.zeros(num_vars, dtype=int)
        
        # Use CSC for fast column access if available
        if hasattr(A, "tocsc"):
            A = A.tocsc()
        
        for j in range(num_vars):
            column = A[:, j]
            if hasattr(column, "toarray"):
                column = column.toarray().flatten()
            # Use tolerance to determine nonzero entries
            nonzero_count = np.count_nonzero(np.abs(column) > self.tol)
            scores[j] = nonzero_count * self.scaling
        
        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Calculate the number of nonzero coefficients in each constraint's row.
        Handles both dense and sparse matrices.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_constraints, dtype=int)
        
        # If A is sparse and supports getnnz, use it:
        if hasattr(A, "getnnz"):
            # getnnz(axis=1) returns the nonzero count for each row.
            scores = A.getnnz(axis=1) * self.scaling
            return scores.tolist()
        else:
            # Otherwise, convert to CSR for fast row access if possible.
            if hasattr(A, "tocsr"):
                A = A.tocsr()
            for i in range(num_constraints):
                row = A[i, :]
                if hasattr(row, "toarray"):
                    row = row.toarray().flatten()
                nonzero_count = np.count_nonzero(np.abs(row) > self.tol)
                scores[i] = nonzero_count * self.scaling
            return scores.tolist()

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Wraps the variable scoring method to return a single score as a tuple,
        so that it can be used in lexicographic ordering.
        """
        return self.score_variables([vars[idx]],
                                    obj_coeffs[idx:idx+1],
                                    [bounds[idx]],
                                    A, constraints, rhs)[0]

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Wraps the constraint scoring method to return a single score as a tuple,
        so that it can be used in lexicographic ordering.
        """
        rhs_single = np.array([rhs[idx]]) if rhs is not None else None
        return self.score_constraints(vars, obj_coeffs, bounds,
                                      A, [constraints[idx]], rhs_single)[0]

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block defined by the indices (var_indices, constr_indices) using
        the cardinality scores computed on the submatrix.
        
        This method reuses the existing score_variables and score_constraints methods by:
          1. Constructing sub-lists for variables, bounds, constraints, and rhs corresponding to
             the given indices.
          2. Extracting the submatrix from A.
          3. Calling score_variables and score_constraints on the submatrix and sub-lists.
          4. Grouping the original indices by the computed scores.
          
        Returns a dictionary mapping labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Construct sub-lists corresponding to the current block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constraints_sub = [constraints[i] for i in constr_indices]
        if rhs is not None:
            rhs_sub = [rhs[i] for i in constr_indices]
        else:
            rhs_sub = None

        # Extract the submatrix corresponding to the block.
        # First slice rows, then columns.
        submatrix = A[constr_indices, :][:, var_indices]
        
        # Compute variable scores on the submatrix.
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constraints_sub, rhs_sub)
        # Compute constraint scores on the submatrix.
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constraints_sub, rhs_sub)
        
        # Group original indices by their scores.
        var_partitions = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_partitions[score].append(idx)
            
        constr_partitions = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_partitions[score].append(idx)
            
        # Form subblocks via Cartesian product.
        partition_map = {}
        label = 0
        for score_v, var_group in var_partitions.items():
            for score_c, constr_group in constr_partitions.items():
                partition_map[label] = (var_group, constr_group)
                label += 1
                
        return partition_map
