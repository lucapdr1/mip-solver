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

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        """
        Vectorized scoring of variables based on their types and bounds.
        
        Scoring:
        - 4 for SEMICONT or SEMIINT.
        - 3 for BINARY.
        - For INTEGER: 3 if (ub - lb) is ~1 (binary-like), else 2.
        - 1 for CONTINUOUS.
        - 0 for any other type.
        
        The final score is multiplied by self.scaling.
        Returns a NumPy array of scores.
        """
        n = len(vars)
        # Create an array of variable types.
        vtypes = np.array([var.VType for var in vars])
        # Create a 2D array for bounds.
        bounds_arr = np.array(bounds, dtype=float)  # shape (n, 2)
        # Compute the difference (ub - lb) for each variable.
        diffs = bounds_arr[:, 1] - bounds_arr[:, 0]
        
        # Initialize scores to zero.
        scores = np.zeros(n, dtype=float)
        
        # Masks for each type:
        mask_semicont = (vtypes == GRB.SEMICONT)
        mask_semiint = (vtypes == GRB.SEMIINT)
        mask_binary   = (vtypes == GRB.BINARY)
        mask_integer  = (vtypes == GRB.INTEGER)
        mask_cont     = (vtypes == GRB.CONTINUOUS)
        
        # Apply scores based on type.
        scores[mask_semicont] = 4
        scores[mask_semiint] = 4
        scores[mask_binary] = 3
        scores[mask_cont] = 1
        
        # For INTEGER, check if the bound difference is close to 1.
        mask_int_binary_like = mask_integer & (np.abs(diffs - 1.0) < 1e-9)
        mask_int_nonbinary   = mask_integer & ~mask_int_binary_like
        scores[mask_int_binary_like] = 3
        scores[mask_int_nonbinary] = 2

        # Multiply by scaling factor.
        scores *= self.scaling
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule is only concerned with variable types.
        return np.zeros(len(constraints), dtype=int)
    
    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a score for a single variable as a tuple, so it is compatible with lexicographic ordering.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a score for a single constraint as a tuple.
        Since this rule does not affect constraints, always return (0,).
        """
        return (0,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
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
        
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None    

        # We pass the original A; the rule does not depend on A.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub))
            
        # Group the original variable indices by their computed score.
        unique_scores = np.unique(sub_var_scores)
        var_groups = {}
        for score in unique_scores:
            mask = (sub_var_scores == score)
            var_groups[score] = var_indices[mask]
        
        # All constraints are scored 0 by this rule.
        constr_groups = {0: constr_indices}
        
        # Form the partition map as the Cartesian product of the variable groups and constraint groups.
        partition_map = {}
        label = 0
        for score, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map