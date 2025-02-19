from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class ColumnsCoefficientRule(OrderingRule):
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule does not reorder variables
        return (self.scaling * np.sum(np.log1p(np.abs(A)), axis=0)).tolist()[0]
        
    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        #Simple absolute value
        #return np.sum(np.abs(A), axis=1).flatten().tolist()[0]
        return [0] * len(constraints)
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the computed score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)
    
    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the score for a single constraint (row) as a one-element tuple.
        Since this rule does not assign meaningful scores to constraints, it always returns (0,).
        """
        return (0,)