import numpy as np
from core.ordering.ordering_rule_interface import OrderingRule

class MaxMinRatioRule(OrderingRule):
    """
    Computes the max-to-min coefficient ratio for each column (variable) or row (constraint).
    If all coefficients in a column/row are of similar scale, the ratio is close to 1.
    If there is a wide variation, the ratio is large.
    """

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_vars = A.shape[1]
        scores = np.zeros(num_vars)

        A = A.tocsc() if hasattr(A, "tocsc") else A  # Convert sparse matrix to CSC for fast column access

        for j in range(num_vars):
            column = A[:, j].toarray().flatten() if hasattr(A[:, j], "toarray") else A[:, j]
            nonzero = column[column != 0]

            if len(nonzero) == 0:
                scores[j] = 0  # If all zero, no meaningful ratio
            else:
                scores[j] = np.max(np.abs(nonzero)) / np.min(np.abs(nonzero))

        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = np.zeros(num_constraints)

        A = A.tocsr() if hasattr(A, "tocsr") else A  # Convert sparse matrix to CSR for fast row access

        for i in range(num_constraints):
            row = A[i, :].toarray().flatten() if hasattr(A[i, :], "toarray") else A[i, :]
            nonzero = row[row != 0]

            if len(nonzero) == 0:
                scores[i] = 0  # If all zero, no meaningful ratio
            else:
                scores[i] = np.max(np.abs(nonzero)) / np.min(np.abs(nonzero))

        return scores.tolist()
    
#TODO: modify it's not truly scale invariant
class MaxSumRatioRule(OrderingRule):
    """
    Computes max-to-sum coefficient ratio for each column (variable) or row (constraint).
    Captures whether one coefficient dominates a column/row.
    """

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_vars = A.shape[1]
        scores = np.zeros(num_vars)

        A = A.tocsc() if hasattr(A, "tocsc") else A

        for j in range(num_vars):
            column = A[:, j].toarray().flatten() if hasattr(A[:, j], "toarray") else A[:, j]
            abs_values = np.abs(column)
            max_coeff = np.max(abs_values)
            sum_coeff = np.sum(abs_values) + 1e-12  # Small epsilon to avoid division by zero

            scores[j] = max_coeff / sum_coeff

        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = np.zeros(num_constraints)

        A = A.tocsr() if hasattr(A, "tocsr") else A

        for i in range(num_constraints):
            row = A[i, :].toarray().flatten() if hasattr(A[i, :], "toarray") else A[i, :]
            abs_values = np.abs(row)
            max_coeff = np.max(abs_values)
            sum_coeff = np.sum(abs_values) + 1e-12  # Small epsilon to avoid division by zero

            scores[i] = max_coeff / sum_coeff

        return scores.tolist()
    
class ConstraintIntegerCountRule(OrderingRule):
    """
    Counts the number of integer variables appearing in each constraint.
    """
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # No effect on variables
        return [0] * len(vars)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = []

        A = A.tocsr() if hasattr(A, "tocsr") else A

        for i in range(num_constraints):
            row = A[i, :].toarray().ravel() if hasattr(A[i, :], "toarray") else np.ravel(A[i, :])

            integer_count = sum(vars[j].VType in {2, 3} for j in range(len(vars)) if float(row[j]) != 0)

            scores.append(integer_count)  # Return a single scalar instead of tuple

        return scores
    
class ConstraintContinuousCountRule(OrderingRule):
    """
    Counts the number of continuous variables appearing in each constraint.
    """
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        return [0] * len(vars)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = []

        A = A.tocsr() if hasattr(A, "tocsr") else A

        for i in range(num_constraints):
            row = A[i, :].toarray().ravel() if hasattr(A[i, :], "toarray") else np.ravel(A[i, :])

            continuous_count = sum(vars[j].VType == 1 for j in range(len(vars)) if float(row[j]) != 0)

            scores.append(continuous_count)  # Return a single scalar instead of tuple

        return scores

