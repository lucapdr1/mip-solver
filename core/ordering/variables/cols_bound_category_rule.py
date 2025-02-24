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

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes a score for each variable based solely on its bounds.
        The logic is:
          - If var.VType is SEMICONT or SEMIINT: score = 4
          - Else if var.VType is BINARY: score = 3
          - Else if var.VType is INTEGER:
                if (upper_bound - lower_bound) is approximately 1, score = 3 (treated as binary);
                otherwise, score = 2.
          - Else if var.VType is CONTINUOUS: score = 1
          - Otherwise: score = 0
        """
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]  # (lower_bound, upper_bound)
            # Check infinite bounds
            is_lb_inf = math.isinf(lb)
            is_ub_inf = math.isinf(ub)

            if (not is_lb_inf) and (not is_ub_inf):
                # Both bounds finite
                if lb >= 0 or ub <= 0:
                    # Entire range is nonnegative or nonpositive
                    cat = 4
                elif lb < 0 < ub:
                    # Straddles zero
                    cat = 3
                else:
                    # Fallback if needed
                    cat = 2
            elif (is_lb_inf and not is_ub_inf) or (not is_lb_inf and is_ub_inf):
                # Exactly one bound is infinite
                cat = 2
            elif is_lb_inf and is_ub_inf:
                # Both bounds infinite
                cat = 1
            else:
                cat = 0

            scores.append(cat * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule is only concerned with variable bound properties.
        return np.zeros(len(constraints), dtype=int)
    
    # --- Methods for Rectangular Block Partitioning ---
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the bound-category score for a single variable as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Since bound category does not affect constraints, we return a fixed tuple.
        """
        return (0,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
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

        # Compute scores on the sub-block.
        sub_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, A, constr_sub, rhs_sub)
        
        # Group the original variable indices by their computed score.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_scores):
            var_groups[score].append(idx)
            
        # All constraints receive a score of 0.
        constr_groups = {0: list(constr_indices)}
        
        # Form the partition map as the Cartesian product of variable groups and constraint groups.
        partition_map = {}
        label = 0
        for score_v, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map
