from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class NonZeroCountRule(OrderingRule):
    """
    Assigns scores based on the number of nonzero coefficients (cardinality of
    each column/row). The rule is scale-invariant because it considers only
    the count of nonzero entries rather than their magnitudes.

    - score_variables(...) returns the number of nonzeros in the column of each variable.
    - score_constraints(...) returns the number of nonzeros in the row of each constraint.
    
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling  
        self.tol = tol

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Calculate the number of nonzero coefficients in each variable's column,
        considering entries with absolute value > self.tol as nonzero.
        The score for each variable is: nonzero_count * self.scaling.
        
        This function handles both dense and sparse matrices.
        """
        num_constraints, num_vars = A.shape
        if hasattr(A, "tocsc"):
            # Convert A to CSC format for fast column slicing.
            A_csc = A.tocsc()
            # For each column j, np.diff(A_csc.indptr)[j] gives the number of stored (nonzero) elements.
            # To get the "nonzero" count based on tol, we first create a column index for each stored element.
            col_indices = np.repeat(np.arange(num_vars), np.diff(A_csc.indptr))
            # Create a boolean mask for entries that are "nonzero" (exceed tol).
            mask = np.abs(A_csc.data) > self.tol
            # Count the number of True values for each column using np.bincount.
            counts = np.bincount(col_indices[mask], minlength=num_vars)
            scores = counts * self.scaling
        else:
            # Assume A is a dense NumPy array.
            mask = np.abs(A) > self.tol  # shape (num_constraints, num_vars)
            counts = np.sum(mask, axis=0)  # count nonzeros per column
            scores = counts * self.scaling

        return scores  # Returns a NumPy array

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Calculate the number of nonzero coefficients in each constraint's row,
        considering entries with absolute value > self.tol as nonzero.
        The score for each constraint is: nonzero_count * self.scaling.
        
        This function handles both dense and sparse matrices.
        """
        num_constraints, num_vars = A.shape
        if hasattr(A, "tocsr"):
            # Convert A to CSR for fast row slicing.
            A_csr = A.tocsr()
            # Create row indices for each nonzero entry.
            row_indices = np.repeat(np.arange(num_constraints), np.diff(A_csr.indptr))
            mask = np.abs(A_csr.data) > self.tol
            counts = np.bincount(row_indices[mask], minlength=num_constraints)
            scores = counts * self.scaling
        else:
            # Assume A is a dense NumPy array.
            mask = np.abs(A) > self.tol  # shape (num_constraints, num_vars)
            counts = np.sum(mask, axis=1)  # count nonzeros per row
            scores = counts * self.scaling

        return scores  # Returns a NumPy array

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Wraps the variable scoring method to return a single score as a tuple,
        so that it can be used in lexicographic ordering.
        """
        return self.score_variables([vars[idx]],
                                    obj_coeffs[idx:idx+1],
                                    [bounds[idx]],
                                    A, A_csc, A_csr, constraints, rhs)[0]

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Wraps the constraint scoring method to return a single score as a tuple,
        so that it can be used in lexicographic ordering.
        """
        rhs_single = np.array([rhs[idx]]) if rhs is not None else None
        return self.score_constraints(vars, obj_coeffs, bounds,
                                      A, A_csc, A_csr, [constraints[idx]], rhs_single)[0]

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
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
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the block.
        # First slice rows, then columns.
        # Ensure A is in CSR for efficient row slicing:
        A_csr = A.tocsr()
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()
        
        # Compute scores on the sub-block.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}
        
        # Group the original constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form the partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_partitions[score_v], constr_partitions[score_c])
                label += 1
                    
        return partition_map
