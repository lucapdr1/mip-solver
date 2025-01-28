from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class RHSValueRule(OrderingRule):
    def score_variables(self, vars, obj_coeffs, bounds):
        # This rule does not reorder variables
        return [0] * len(vars)

    def score_constraints(self, constraints, A, rhs):
       # Higher absolute values of rhs get higher scores
        return np.abs(rhs).tolist()
