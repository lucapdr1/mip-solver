from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB
from collections import defaultdict
import math
import numpy as np

class VariableTypeRule(OrderingRule):
    """
    Extended variable-type rule that:
      1) Distinguishes semi-continuous or semi-integer variables.
      2) Treats integer variables with [a, a+1] as effectively binary.
      3) Otherwise uses standard binary=3, integer=2, continuous=1.

    (You can adjust the numeric scores as needed.)
    """

    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a list of scores for the variables reflecting their "type priority."
        Higher score implies higher priority.
        - SEMICONT or SEMIINT => 4
        - BINARY => 3
        - INTEGER: if (ub - lb) is approximately 1 => 3 (treated as binary); else 2
        - CONTINUOUS => 1
        - Fallback => 0
        """
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]  # [lower_bound, upper_bound]
            
            if var.VType in [GRB.SEMICONT, GRB.SEMIINT]:
                score = 4
            elif var.VType == GRB.BINARY:
                score = 3
            elif var.VType == GRB.INTEGER:
                if math.isclose(ub - lb, 1.0, abs_tol=1e-9):
                    score = 3  # treat as binary
                else:
                    score = 2
            elif var.VType == GRB.CONTINUOUS:
                score = 1
            else:
                score = 0
            scores.append(score * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule is only concerned with variable types.
        return np.zeros(len(constraints), dtype=int)
    
    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a score for a single variable as a tuple, so it is compatible with lexicographic ordering.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a score for a single constraint as a tuple.
        Since this rule does not affect constraints, always return (0,).
        """
        return (0,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the given block (specified by var_indices and constr_indices) based on variable type.
        
        The process is:
          1. Construct sub-lists corresponding to the current block.
          2. Compute the variable scores for this block by calling score_variables.
          3. Group the original variable indices by the computed score.
          4. Since constraints are unaffected, put all constraint indices in one group.
          5. Form the partition as the Cartesian product of these groups.
        
        Returns a dictionary mapping a label to a tuple:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Extract sub-lists for the current block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        # For constraints, we could extract a sub-list; but here the rule always returns 0.
        constr_sub = [constraints[i] for i in constr_indices]
        # rhs is passed as-is if provided.
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # We pass the original A; the rule does not depend on A.
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, A, constr_sub, rhs_sub)
        
        # Group the original variable indices by their computed score.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)
        
        # All constraints are scored 0 by this rule.
        constr_groups = {0: list(constr_indices)}
        
        # Form the partition map as the Cartesian product of the groups.
        partition_map = {}
        label = 0
        for score_v, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
                
        return partition_map
