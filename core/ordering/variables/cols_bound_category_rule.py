import math
from collections import defaultdict
from core.ordering.ordering_rule_interface import OrderingRule

class BoundCategoryRule(OrderingRule):
    """
    Scores variables based on the "category" of their bounds:
      4 -> Both bounds finite and nonnegative or nonpositive
      3 -> Both bounds finite and straddle zero (l<0<u)
      2 -> Exactly one bound is infinite
      1 -> Both bounds are infinite
      0 -> fallback, if needed

    You can invert or tweak the hierarchy if you prefer a different order.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
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
                    # straddles zero
                    cat = 3
                else:
                    # fallback if needed (or combine with above logic)
                    cat = 2
            elif (is_lb_inf and not is_ub_inf) or (not is_lb_inf and is_ub_inf):
                # exactly one infinite bound
                cat = 2
            elif (is_lb_inf and is_ub_inf):
                # both infinite
                cat = 1
            else:
                cat = 0

            scores.append(cat * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        return [0] * len(constraints)
    
    # --- Methods for Rectangular Block Partitioning ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the bound-category score for a single variable as a one-element tuple.
        This is used for lexicographic ordering within a block.
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
        Partitions the block based on bound categories.
        
        - Computes a score for each variable in var_indices using this rule.
        - Groups variables into partitions according to their unique bound-category scores.
        - Since all constraints score 0, they are grouped into a single block.
        
        Returns a dictionary mapping block labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Compute the score (as an integer) for each variable in var_indices.
        var_scores = {
            i: self.score_matrix_for_variable(i, vars, obj_coeffs, bounds, A, constraints, rhs)[0]
            for i in var_indices
        }
        
        # Group variables by their bound-category score.
        var_groups = defaultdict(list)
        for i, score in var_scores.items():
            var_groups[score].append(i)
        
        # For constraints, since all score 0, group them into a single group.
        constr_groups = {0: list(constr_indices)}
        
        # Create a partition map as the Cartesian product of variable groups and constraint groups.
        partition_map = {}
        label = 0
        for vscore, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map