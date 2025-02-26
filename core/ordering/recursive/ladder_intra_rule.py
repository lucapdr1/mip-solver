import numpy as np
from core.ordering.ordering_rule_interface import OrderingRule

class LadderIntraRule(OrderingRule):
    """
    A ladder-style intra rule that orders variables based on the earliest constraint
    referencing them (via A_csc). If the size of the current block (number of variables
    in the block × number of constraints in the block) is less than a given fraction of
    the total matrix size (original_var_count * original_constr_count), then identity ordering
    is used (i.e. each variable's score is its index). Otherwise, ladder logic is applied:
    
         score[i] = earliest_constraint_index_referencing_var_i + (i / (n+1))
    
    Constraints always receive identity ordering.
    
    When using this rule, the calling code should pass the total numbers of variables and
    constraints as keyword arguments “original_var_count” and “original_constr_count” to the
    per-item methods.
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
        If the current block size is less than threshold * (total matrix size), identity ordering is returned.
        Otherwise, for each variable the score is:
              score[i] = earliest_row_that_references_var_i + (i / (n+1))
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

        scores = [0.0] * n
        for i in range(n):
            start = A_csc.indptr[i]
            end = A_csc.indptr[i + 1]
            nonzero_rows = A_csc.indices[start:end]
            if len(nonzero_rows) == 0:
                row_pos = m
            else:
                row_pos = min(nonzero_rows)
            scores[i] = float(row_pos) + (i / (n + 1.0))
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
        all_scores = self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs,
                                          original_var_count, original_constr_count)
        return (all_scores[idx],)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr,
                                    constraints, rhs, **kwargs):
        """
        Returns a one-element tuple with the score for a single constraint.
        """
        all_scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        return (all_scores[idx],)
