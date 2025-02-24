from core.ordering.ordering_rule_interface import OrderingRule
from collections import defaultdict
import numpy as np
import math

class BoundCategoryRule(OrderingRule):
    """
    Scores variables based on the "category" of their bounds:
      4 -> Both bounds finite and nonnegative or nonpositive
      3 -> Both bounds finite and straddle zero (l < 0 < u)
      2 -> Exactly one bound is infinite
      1 -> Both bounds are infinite
      0 -> fallback, if needed

    You can adjust or invert the hierarchy as needed.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        """
        Computes a score for each variable based solely on its bounds.
        
        The logic is:
        - If both bounds are finite:
                - If lower bound >= 0 or upper bound <= 0: score = 4
                - Else if the bounds straddle zero (lb < 0 < ub): score = 3
                - Otherwise (fallback): score = 2
        - If exactly one bound is infinite: score = 2
        - If both bounds are infinite: score = 1
        - Otherwise: score = 0
        
        The final score is multiplied by self.scaling.
        
        Returns a NumPy array of scores.
        """
        # Convert bounds to a NumPy array of shape (n, 2)
        bounds_arr = np.array(bounds, dtype=float)
        lb = bounds_arr[:, 0]
        ub = bounds_arr[:, 1]
        n = len(lb)
        
        # Determine where the lower or upper bounds are infinite.
        is_lb_inf = np.isinf(lb)
        is_ub_inf = np.isinf(ub)
        
        # Initialize scores array.
        scores = np.zeros(n, dtype=float)
        
        # Case 1: Both bounds are finite.
        finite_mask = (~is_lb_inf) & (~is_ub_inf)
        # For finite bounds, assign 4 if the entire range is nonnegative or nonpositive.
        mask_nonnegative = lb >= 0
        mask_nonpositive = ub <= 0
        mask_condition1 = finite_mask & (mask_nonnegative | mask_nonpositive)
        scores[mask_condition1] = 4
        
        # For finite bounds, assign 3 if they straddle zero.
        mask_straddle = finite_mask & ((lb < 0) & (ub > 0))
        scores[mask_straddle] = 3
        
        # Fallback for finite bounds (if any remain) to 2.
        mask_fallback = finite_mask & ~(mask_condition1 | mask_straddle)
        scores[mask_fallback] = 2
        
        # Case 2: Exactly one bound is infinite.
        one_inf = (is_lb_inf ^ is_ub_inf)
        scores[one_inf] = 2
        
        # Case 3: Both bounds are infinite.
        both_inf = is_lb_inf & is_ub_inf
        scores[both_inf] = 1
        
        # Multiply by the scaling factor.
        scores *= self.scaling
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule is only concerned with variable bound properties.
        return np.zeros(len(constraints), dtype=int)
    
    # --- Methods for Rectangular Block Partitioning ---
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the bound-category score for a single variable as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Since bound category does not affect constraints, we return a fixed tuple.
        """
        return (0,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) based on bound-category scores.
        
        The method proceeds as follows:
          1. Constructs sub-lists for the current block for variables, bounds, and constraints.
          2. Calls score_variables on these sub-lists (ignoring A, since this rule depends solely on bounds).
          3. Groups the original variable indices by the computed score.
          4. Since this rule does not affect constraints (all score 0), all constraints are grouped together.
          5. Forms the partition as the Cartesian product of the variable groups with the constraint group.
        
        Returns:
            A dictionary mapping block labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None

        # Compute variable scores for this block.
        sub_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score using vectorized masking.
        unique_scores = np.unique(sub_scores)
        var_groups = {}
        for score in unique_scores:
            mask = (sub_scores == score)
            var_groups[score] = var_indices[mask]
        
        # All constraints are scored 0 by this rule.
        constr_groups = {0: constr_indices}
        
        # Form the partition map as the Cartesian product of the variable groups and constraint groups.
        partition_map = {}
        label = 0
        for score in unique_scores:
            vgroup = var_groups[score]
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
                    
        return partition_map
