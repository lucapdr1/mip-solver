import numpy as np
from core.ordering.ordering_rule_interface import OrderingRule

class LadderIntraRule(OrderingRule):
    """
    A ladder-style intra rule that orders variables based on the lexicographic order of their
    column patterns in the constraint matrix (A_csc). If the size of the current block 
    (number of variables × number of constraints) is less than a given fraction of the total 
    matrix size (original_var_count * original_constr_count), then identity ordering is used.
    
    Otherwise, for each variable the key is computed as:
         key[i] = tuple( A_csc.indices[A_csc.indptr[i]:A_csc.indptr[i+1] ] )
    Variables are then sorted lexicographically by these keys. This means that variables
    whose columns have nonzeros in earlier rows appear first (i.e. are shifted left).
    
    Constraints always receive identity ordering.
    
    When using this rule, the caller should pass the full numbers of variables and constraints
    as keyword arguments “original_var_count” and “original_constr_count” to the per-item methods.
    """
    
    def __init__(self, threshold=0.3):
        # threshold is a fraction (e.g. 0.3 means 30% of the full matrix size)
        self.threshold = threshold

    # ----------------------------------------------------------------
    # Full-block scoring methods
    # ----------------------------------------------------------------

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs,
                        original_var_count=None, original_constr_count=None):
        """
        Returns a list of scores (floats) for all variables.
        If the current block size is less than threshold*(total matrix size),
        identity ordering is returned. Otherwise, each variable’s key is computed from
        its nonzero row indices in A_csc and the variables are sorted lexicographically.
        The rank in the sorted order is then used as the score.
        """
        n = len(vars)
        m = len(constraints)
        if original_var_count is None:
            original_var_count = n
        if original_constr_count is None:
            original_constr_count = m

        total_matrix_size = original_var_count * original_constr_count
        block_size = n * m
        if block_size < self.threshold * total_matrix_size:
            return [float(i) for i in range(n)]

        # Build a lexicographic key for each variable's column.
        keys = []
        for i in range(n):
            start = A_csc.indptr[i]
            end = A_csc.indptr[i+1]
            # If the column has no nonzeros, use a key that is "large" (pushes it right).
            if end - start == 0:
                key = (m,)
            else:
                key = tuple(A_csc.indices[start:end])
            keys.append(key)
        # Sort variables by their keys.
        sorted_keys = sorted(enumerate(keys), key=lambda x: x[1])
        scores = [0.0] * n
        for rank, (i, key) in enumerate(sorted_keys):
            scores[i] = float(rank)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs, **kwargs):
        """
        Returns a list of scores (floats) for all constraints.
        Identity ordering is used (score[i] = i).
        """
        m = len(constraints)
        return [float(i) for i in range(m)]

    # ----------------------------------------------------------------
    # Per-item scoring methods
    # ----------------------------------------------------------------

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr,
                                  constraints, rhs, **kwargs):
        """
        Returns a one-element tuple with the score for a single variable.
        Expects keyword arguments 'original_var_count' and 'original_constr_count'.
        """
        original_var_count = kwargs.get("original_var_count", len(vars))
        original_constr_count = kwargs.get("original_constr_count", len(constraints))
        all_scores = self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr,
                                          constraints, rhs, original_var_count, original_constr_count)
        return (all_scores[idx],)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr,
                                    constraints, rhs, **kwargs):
        """
        Returns a one-element tuple with the score for a single constraint.
        """
        all_scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr,
                                            constraints, rhs)
        return (all_scores[idx],)
