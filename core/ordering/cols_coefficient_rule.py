from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class ColumnsCoefficientRule(OrderingRule):
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, A, bounds):
        # This rule does not reorder variables
        return (self.scaling * np.sum(np.log1p(np.abs(A)), axis=0)).tolist()[0]
        
    def score_constraints(self, constraints, A, rhs):
        #Simple absolute value
        #return np.sum(np.abs(A), axis=1).flatten().tolist()[0]
        return [0] * len(constraints)