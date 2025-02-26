from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class ConstraintRangeRule(OrderingRule):
    def __init__(self, scaling=1.0):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not reorder variables
       return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Compute the range of coefficients for each row, ignoring infinities
        ranges = []
        for row in A:
            ranges.append(np.max(row) - np.min(row))
        return (self.scaling * np.log1p(np.abs(ranges))).tolist()
    
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the score for a single variable (column) as a one-element tuple.
        Since this rule does not assign meaningful scores to variables, it always returns (0,).
        """
        return (0,)
    
    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the sign-based score for a single constraint (row) as a one-element tuple.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       A_csc,
                                       A_csr,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)