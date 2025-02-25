import numpy as np
from core.ordering.ordering_rule_interface import OrderingRule

class LadderIntraRule(OrderingRule):
    """
    A ladder-style intra rule, structured like your simple working example rules.
    - If #vars * #constraints < threshold, it returns identity ordering for variables.
    - Otherwise, it finds for each variable the earliest constraint index (row) referencing it.
    - Constraints are always given identity ordering (score = their global index).
    
    The per-item methods (score_matrix_for_variable / constraint) simply call the full-block
    methods and return a single-element tuple, matching the same style as your example rules.
    """

    def __init__(self, threshold=28):
        self.threshold = threshold

    # ----------------------------------------------------------------
    # Full-block scoring methods
    # ----------------------------------------------------------------

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a list of scores (floats) for ALL variables in the problem.
        If block_size < threshold, each variable's score = variable index (identity).
        Otherwise, we compute a 'ladder' score:
            score[i] = earliest_row_that_references_var_i + (i / (n+1))
        """
        n = len(vars)
        m = len(constraints)
        block_size = n * m

        # Small block => identity ordering
        if block_size < self.threshold:
            return [float(i) for i in range(n)]

        # Large block => "ladder" logic.
        # For each variable i, find the earliest constraint referencing it in the full matrix A_csc.
        scores = [0.0] * n
        for i in range(n):
            start = A_csc.indptr[i]
            end = A_csc.indptr[i + 1]
            # All row indices referencing column i
            nonzero_rows = A_csc.indices[start:end]
            if len(nonzero_rows) == 0:
                # if no row references this var, push it to the bottom
                row_pos = m
            else:
                row_pos = min(nonzero_rows)
            # offset by i/(n+1) to break ties
            scores[i] = float(row_pos) + (i / (n + 1.0))

        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a list of scores (floats) for ALL constraints in the problem.
        We simply do identity: score[i] = i.
        """
        m = len(constraints)
        return [float(i) for i in range(m)]

    # ----------------------------------------------------------------
    # Per-item scoring methods
    # ----------------------------------------------------------------

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a one-element tuple with the score for a single variable.
        We call score_variables(...) and take the element at [idx].
        """
        all_scores = self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        return (all_scores[idx],)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a one-element tuple with the score for a single constraint.
        We call score_constraints(...) and take the element at [idx].
        """
        all_scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        return (all_scores[idx],)
