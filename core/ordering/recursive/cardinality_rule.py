from core.ordering.ordering_rule_interface import OrderingRule
from collections import defaultdict
import numpy as np

class CardinalityRule(OrderingRule):
    """
    Assigns scores based on the number of nonzero coefficients (cardinality of
    each column/row). Scale-invariant because it does not consider the magnitudes
    of coefficients.

    - score_variables(...) returns #nonzero in the column of each variable.
    - score_constraints(...) returns #nonzero in the row of each constraint.
    """
    
    def __init__(self, scaling=1):
        self.scaling = scaling  # optional; if you want to uniformly rescale

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Calculate #nonzero coefficients in each variable's column.
        Handles both dense and sparse matrices.
        """
        num_constraints, num_vars = A.shape  # Get dimensions
        scores = np.zeros(num_vars, dtype=int)  # Preallocate array

        # Convert sparse matrix to CSC format for fast column-wise operations
        if hasattr(A, "tocsc"):
            A = A.tocsc()

        for j in range(num_vars):
            column = A[:, j]  # Extract column
            if hasattr(column, "toarray"):
                column = column.toarray().flatten()  # Convert sparse to dense
            nonzero_count = np.count_nonzero(column)  # Count nonzeros
            scores[j] = nonzero_count * self.scaling

        return scores.tolist()  # Return as a list

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Calculate #nonzero coefficients in each constraint's row.
        Handles both dense and sparse matrices.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_constraints, dtype=int)

        # Convert sparse matrix to CSR format for fast row-wise operations
        if hasattr(A, "tocsr"):
            A = A.tocsr()

        for i in range(num_constraints):
            row = A[i, :]  # Extract row
            if hasattr(row, "toarray"):
                row = row.toarray().flatten()  # Convert sparse to dense
            nonzero_count = np.count_nonzero(row)  # Count nonzeros
            scores[i] = nonzero_count * self.scaling

        return scores.tolist()

    # === New methods for rectangular block ordering ===

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Provides a score for a single variable (column) that is compatible with
        a rectangular block ordering scheme.
        
        This method wraps the existing score_variables method by passing a list
        containing the single variable.
        """
        return self.score_variables([vars[idx]],
                                    obj_coeffs[idx:idx+1],
                                    [bounds[idx]],
                                    A, constraints, rhs)[0]

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Provides a score for a single constraint (row) that is compatible with
        a rectangular block ordering scheme.
        
        This method wraps the existing score_constraints method by passing a list
        containing the single constraint.
        """
        # Prepare rhs as a single-element array if rhs is provided.
        rhs_single = np.array([rhs[idx]]) if rhs is not None else None
        return self.score_constraints(vars, obj_coeffs, bounds,
                                      A, [constraints[idx]], rhs_single)[0]

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block using natural groupings based on cardinality scores.

        - Variables with the same nonzero count are grouped together.
        - Constraints with the same nonzero count are grouped together.
        - Forms rectangular sub-blocks by intersecting these groups.

        Returns a dictionary:
            {label: (list_of_var_indices, list_of_constr_indices)}
        """

        # Compute scores for variables (columns) and constraints (rows)
        var_scores = {var_idx: self.score_matrix_for_variable(var_idx, vars, obj_coeffs, bounds, A, constraints, rhs)
                    for var_idx in var_indices}
        constr_scores = {constr_idx: self.score_matrix_for_constraint(constr_idx, vars, obj_coeffs, bounds, A, constraints, rhs)
                        for constr_idx in constr_indices}

        # Group variables by their cardinality scores
        var_partitions = defaultdict(list)
        for var_idx, score in var_scores.items():
            var_partitions[score].append(var_idx)

        # Group constraints by their cardinality scores
        constr_partitions = defaultdict(list)
        for constr_idx, score in constr_scores.items():
            constr_partitions[score].append(constr_idx)

        # Generate sub-blocks based on the intersection of variable and constraint groups
        partition_map = {}
        label = 0
        for var_score, var_group in var_partitions.items():
            for constr_score, constr_group in constr_partitions.items():
                partition_map[label] = (var_group, constr_group)
                label += 1

        return partition_map
