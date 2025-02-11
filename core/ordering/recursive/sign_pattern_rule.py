from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

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

    def __init__(self, scaling=1):
        self.scaling = scaling  # optional scaling parameter

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Determine sign pattern for each variable's column in A.
        """
        num_constraints, num_vars = A.shape  # Get actual matrix dimensions
        scores = np.zeros(num_vars, dtype=int)  # Preallocate as NumPy array

        for j in range(num_vars):
            column = A[:, j]  # Get entire column as a NumPy array or sparse matrix

            # Ensure `column` is converted to a dense NumPy array if it's sparse
            if hasattr(column, "toarray"):
                column = column.toarray().flatten()  # Convert sparse to dense and flatten

            # Fix: Convert to boolean scalars
            has_pos = np.any(column > 1e-15)
            has_neg = np.any(column < -1e-15)

            # Assign sign pattern codes
            if has_pos and not has_neg:
                scores[j] = 2  # All positive
            elif has_neg and not has_pos:
                scores[j] = 1  # All negative
            else:
                scores[j] = 0  # Mixed or all zero

        return scores.tolist()  # Ensure output is a list

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Determine sign pattern for each constraint's row in A.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_constraints, dtype=int)

        for i in range(num_constraints):
            row = A[i, :]  # Get entire row as a NumPy array or sparse matrix

            # Ensure `row` is converted to a dense NumPy array if it's sparse
            if hasattr(row, "toarray"):
                row = row.toarray().flatten()

            # Fix: Convert to boolean scalars
            has_pos = np.any(row > 1e-15)
            has_neg = np.any(row < -1e-15)

            # Assign sign pattern codes
            if has_pos and not has_neg:
                scores[i] = 2  # All positive
            elif has_neg and not has_pos:
                scores[i] = 1  # All negative
            else:
                scores[i] = 0  # Mixed or all zero

        return scores.tolist()
