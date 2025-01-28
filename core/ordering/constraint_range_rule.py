from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class ConstraintRangeRule(OrderingRule):
    def __init__(self, scaling=1.0):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, A, bounds):
        # This rule does not reorder variables
        return [0] * len(vars)

    def score_constraints(self, constraints, A, rhs):
        # Compute the range of coefficients for each row, ignoring infinities
        ranges = []
        for row in A:
            ranges.append(np.max(row) - np.min(row))
        return (self.scaling * np.log1p(np.abs(ranges))).tolist()