from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class RowCoefficientRule(OrderingRule):
    def score_variables(self, vars, obj_coeffs, bounds):
        # This rule does not reorder variables
        return [0] * len(vars)

    def score_constraints(self, constraints, A, rhs):
        #Simple absolute value
        #return np.sum(np.abs(A), axis=1).flatten().tolist()[0]

        return np.sum(np.log1p(np.abs(A)), axis=1).flatten().tolist()[0]