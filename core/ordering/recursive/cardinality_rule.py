from core.ordering.ordering_rule_interface import OrderingRule
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
