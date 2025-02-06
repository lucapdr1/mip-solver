from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from scipy import sparse

class VariableOccurrenceRule(OrderingRule):
    def __init__(self, scaling=1):
        self.scaling = scaling
        
    def count_occurrences(self, A):
        if sparse.issparse(A):
            # For sparse matrix, count non-zeros in each column
            return np.array(A.astype(bool).sum(axis=0)).flatten()
        else:
            # For dense matrix, count non-zeros in each column
            return np.count_nonzero(A, axis=0)
    
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):    
        # Count occurrences for each variable
        occurrences = self.count_occurrences(A)
        
        # Weight the occurrences by the specified factor
        return (occurrences * self.scaling).tolist()
    
    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        This rule does not affect constraint ordering.
        
        Returns:
            List of zeros for each constraint
        """
        return [0] * len(constraints)