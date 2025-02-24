import numpy as np
from core.ordering.ordering_rule_interface import OrderingRule
from collections import defaultdict

class NormalizedOccurrenceCountRule(OrderingRule):
    """
    For each variable (column), computes:
      ratio(v) = (# of nonzero entries in v in the current block) / (number of rows in the block)
    For each constraint (row), computes:
      ratio(c) = (# of nonzero entries in c in the current block) / (number of columns in the block)
    
    Optionally, the ratio is multiplied by a scale factor and discretized.
    This makes the rule scale invariant: the score depends on relative density.
    """
    
    def __init__(self, scale_factor=10, discretize=True, tol=1e-12):
        """
        :param scale_factor: Factor to multiply the ratio before discretizing.
        :param discretize: If True, use floor() to discretize the score.
        :param tol: Tolerance for determining nonzero.
        """
        self.scale_factor = scale_factor
        self.discretize = discretize
        self.tol = tol

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # A is assumed to be the matrix corresponding to the block.
        num_rows = A.shape[0]
        scores = []
        # Use CSC if available for fast column access.
        if hasattr(A, "tocsc"):
            A = A.tocsc()
        for j in range(A.shape[1]):
            column = A[:, j]
            if hasattr(column, "toarray"):
                column = column.toarray().flatten()
            nonzero_count = np.count_nonzero(np.abs(column) > self.tol)
            ratio = nonzero_count / num_rows if num_rows > 0 else 0
            if self.discretize:
                score = int(np.floor(self.scale_factor * ratio))
            else:
                score = ratio
            scores.append(score)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        num_cols = A.shape[1]
        scores = []
        # Use CSR if available for fast row access.
        if hasattr(A, "tocsr"):
            A = A.tocsr()
        for i in range(A.shape[0]):
            row = A[i, :]
            if hasattr(row, "toarray"):
                row = row.toarray().flatten()
            nonzero_count = np.count_nonzero(np.abs(row) > self.tol)
            ratio = nonzero_count / num_cols if num_cols > 0 else 0
            if self.discretize:
                score = int(np.floor(self.scale_factor * ratio))
            else:
                score = ratio
            scores.append(score)
        return scores

    # --- Methods for Rectangular Block Ordering ---
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        score = self.score_variables([vars[idx]],
                                      obj_coeffs[idx:idx+1],
                                      [bounds[idx]],
                                      A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        rhs_single = np.array([rhs[idx]]) if rhs is not None else None
        score = self.score_constraints(vars, obj_coeffs, bounds,
                                       A, A_csc, A_csr, [constraints[idx]], rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block using normalized occurrence count scores computed on the A.
        The rule reuses its score_variables and score_constraints methods.
        """
        sub_var_scores = self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        sub_constr_scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        
        # Group original indices by their scores.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)
            
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_groups[score].append(idx)
            
        partition_map = {}
        label = 0
        for score_v, vgroup in var_groups.items():
            for score_c, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
                
        return partition_map
