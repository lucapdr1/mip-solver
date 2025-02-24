from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class SignPatternRule(OrderingRule):
    """
    Modified rule that extracts sign information only from constraints.
    
    For **variables (columns)**:
      - Since the sign pattern is not invariant (due to possible row scaling
        and multiplication by -1), we do not attempt to extract a sign pattern.
        Instead, we return a constant score (0) for each variable.
    
    For **constraints (rows)**:
      - We count the number of positive and negative entries (using a tolerance).
      - Then, we adjust these counts by adding one to the count corresponding
        to the slack variable’s sign (based on the inequality operator in the constraint).
        For example, for a "≤" constraint we add one to the positive count (since
        the slack is nonnegative), and for a "≥" constraint we add one to the negative count.
      - The final score is the maximum of these two adjusted counts.
    """

    def __init__(self):
        self.tol = 1e-15  # tolerance for considering a value nonzero

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        """
        Since sign information for columns is not invariant under row scaling,
        we do not try to compute it. Instead, we simply return a constant score,
        for example 0 for each variable.
        """
        return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        For each constraint (row), count the positive and negative entries
        (using a tolerance) and adjust the count based on the slack variable's sign.
        The final score is the maximum of the two (adjusted) counts.
        """
        num_constraints, num_vars = A.shape

        # Convert A to a dense array if possible.
        if hasattr(A, "toarray"):
            A_dense = A.toarray()
        else:
            A_dense = np.array(A)

        # Compute counts of positive and negative entries for all rows at once.
        pos_counts = np.sum(A_dense > self.tol, axis=1)
        neg_counts = np.sum(A_dense < -self.tol, axis=1)

        # Extract slack signs from each constraint string.
        # A '+' indicates that the slack is expected to be positive (i.e., add one to pos_counts),
        # and a '-' indicates negative slack (i.e., add one to neg_counts).
        slack_signs = np.array([
            '+' if ("<=" in str(c) or "<" in str(c)) else 
            '-' if (">=" in str(c) or ">" in str(c)) else ''
            for c in constraints
        ])

        # Create boolean arrays for the adjustments.
        add_to_pos = (slack_signs == '+').astype(int)
        add_to_neg = (slack_signs == '-').astype(int)

        # Adjust counts accordingly.
        pos_counts += add_to_pos
        neg_counts += add_to_neg

        # Final score is the maximum of the two counts for each row.
        scores = np.maximum(pos_counts, neg_counts)

        return scores  # Returning a NumPy array (avoid .tolist() for performance)

    # --- Methods for Rectangular Block Reordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the (constant) score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the sign-based score for a single constraint (row) as a one-element tuple.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) using sign pattern-based grouping.
        
        The process is:
          1. Construct sub-lists for variables, bounds, and constraints corresponding to the block.
          2. Extract the submatrix corresponding to these indices.
          3. Compute variable scores (which will all be 0) and constraint scores using the above methods.
          4. Group the original indices by these computed scores.
          5. Form the partition as the Cartesian product of the variable groups and constraint groups.
        
        Returns a dictionary mapping block labels to tuples:
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

        # Extract the submatrix corresponding to the current block.
        # Ensure A is in CSR for efficient row slicing:
        A_csr = A.tocsr()
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()

        # Compute scores on the submatrix.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score.
        unique_var_scores = np.unique(sub_var_scores)
        var_groups = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}
        
        # Group the original constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_groups = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form the partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_groups[score_v], constr_groups[score_c])
                label += 1
                    
        return partition_map
