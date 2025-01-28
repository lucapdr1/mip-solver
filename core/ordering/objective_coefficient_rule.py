from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class ObjectiveCoefficientRule(OrderingRule):
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, A, bounds):
        # Higher absolute values of objective coefficients get higher scores
        
        #Simple absolute value
        #return np.abs(obj_coeffs).tolist()

        #Log(1 + abs_value)
        return (self.scaling * np.log1p(np.abs(obj_coeffs))).tolist()

    def score_constraints(self, constraints, A, rhs):
        # This rule does not reorder constraints
        return [0] * len(constraints)
    
