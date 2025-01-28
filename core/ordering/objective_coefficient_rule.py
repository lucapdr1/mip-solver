from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class ObjectiveCoefficientRule(OrderingRule):
    def score_variables(self, vars, obj_coeffs, bounds):
        # Higher absolute values of objective coefficients get higher scores
        return np.abs(obj_coeffs).tolist()

    def score_constraints(self, constraints, A, rhs):
        # This rule does not reorder constraints
        return [0] * len(constraints)
