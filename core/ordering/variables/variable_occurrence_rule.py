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
    
    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):    
        # Count occurrences for each variable
        occurrences = self.count_occurrences(A)
        
        # Weight the occurrences by the specified factor
        return (occurrences * self.scaling).tolist()
    
    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        This rule does not affect constraint ordering.
        
        Returns:
            List of zeros for each constraint
        """
        return np.zeros(len(constraints), dtype=int)
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the computed score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)
    
    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the score for a single constraint (row) as a one-element tuple.
        Since this rule does not assign meaningful scores to constraints, it always returns (0,).
        """
        return (0,)