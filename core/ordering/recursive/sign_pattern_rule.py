from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class SignPatternRule(OrderingRule):
    """
    Classifies the sign pattern of the nonzero coefficients:
      - For columns (variables): are coefficients all >= 0, all <= 0, or mixed?
      - For rows (constraints): same check across that row.

    Returns integer 'pattern codes':
      - 2 => all nonnegative
      - 1 => all nonpositive
      - 0 => mixed
    """

    def __init__(self):
        pass  # No need for a scaling parameter here

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Determine sign pattern for each variable's column in A.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_vars, dtype=int)

        for j in range(num_vars):
            column = A[:, j]  

            if hasattr(column, "toarray"):
                column = column.toarray().flatten()  # Convert sparse to dense

            has_pos = np.any(column > 1e-15)
            has_neg = np.any(column < -1e-15)

            if has_pos and not has_neg:
                scores[j] = 2  # All positive
            elif has_neg and not has_pos:
                scores[j] = 1  # All negative
            else:
                scores[j] = 0  # Mixed or all zero

        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Determine sign pattern for each constraint's row in A.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_constraints, dtype=int)

        for i in range(num_constraints):
            row = A[i, :]

            if hasattr(row, "toarray"):
                row = row.toarray().flatten()

            has_pos = np.any(row > 1e-15)
            has_neg = np.any(row < -1e-15)

            if has_pos and not has_neg:
                scores[i] = 2  # All positive
            elif has_neg and not has_pos:
                scores[i] = 1  # All negative
            else:
                scores[i] = 0  # Mixed or all zero

        return scores.tolist()

    # === New Methods for Rectangular Block Reordering ===

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes the sign pattern for a **single variable column**.
        """
        return self.score_variables([vars[idx]], obj_coeffs[idx:idx+1], [bounds[idx]], A, constraints, rhs)[0]

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes the sign pattern for a **single constraint row**.
        """
        rhs_single = np.array([rhs[idx]]) if rhs is not None else None
        return self.score_constraints(vars, obj_coeffs, bounds, A, [constraints[idx]], rhs_single)[0]

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block using **sign pattern-based grouping**.

        - Variables with the same sign pattern are grouped together.
        - Constraints with the same sign pattern are grouped together.
        - Forms rectangular sub-blocks by intersecting these groups.

        Returns a dictionary:
            {label: (list_of_var_indices, list_of_constr_indices)}
        """

        # Compute sign pattern scores for variables (columns) and constraints (rows)
        var_scores = {var_idx: self.score_matrix_for_variable(var_idx, vars, obj_coeffs, bounds, A, constraints, rhs)
                      for var_idx in var_indices}
        constr_scores = {constr_idx: self.score_matrix_for_constraint(constr_idx, vars, obj_coeffs, bounds, A, constraints, rhs)
                         for constr_idx in constr_indices}

        # Group variables based on their sign pattern scores
        var_partitions = defaultdict(list)
        for var_idx, score in var_scores.items():
            var_partitions[score].append(var_idx)

        # Group constraints based on their sign pattern scores
        constr_partitions = defaultdict(list)
        for constr_idx, score in constr_scores.items():
            constr_partitions[score].append(constr_idx)

        # Generate sub-blocks by intersecting the variable and constraint partitions
        partition_map = {}
        label = 0
        for var_score, var_group in var_partitions.items():
            for constr_score, constr_group in constr_partitions.items():
                partition_map[label] = (var_group, constr_group)
                label += 1

        return partition_map
