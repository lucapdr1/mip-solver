from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB
from collections import defaultdict
import math

class ConstraintCompositionRule(OrderingRule):
    """
    Assigns a score to each constraint based on whether it involves only integral/binary
    variables, only continuous variables, or a mix.
    
    The scoring logic is:
      - Score 3 if the constraint involves only integral (or binary/semi-integer) variables.
      - Score 2 if the constraint involves only continuous (or semi-continuous) variables.
      - Score 1 if the constraint involves a mix of both.
    
    (You may adjust the scores if needed.)
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule focuses solely on constraint composition,
        # so variable scores remain unaffected.
        return [0] * len(vars)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Iterates over each constraint (row) in A, examines which variables (columns)
        appear, and then assigns a score:
          - 3 if the constraint involves only integral/binary/semi-integer variables.
          - 2 if it involves only continuous/semi-continuous variables.
          - 1 if it involves a mix.
        """
        scores = []
        for row_idx, constr in enumerate(constraints):
            row_data = A[row_idx, :]
            nz_indices = row_data.nonzero()[0]

            has_integral = False
            has_continuous = False

            for col_idx in nz_indices:
                var = vars[col_idx]  # Use the provided variable list.
                if var.VType in [GRB.BINARY, GRB.INTEGER, GRB.SEMIINT]:
                    has_integral = True
                elif var.VType in [GRB.CONTINUOUS, GRB.SEMICONT]:
                    has_continuous = True

                if has_integral and has_continuous:
                    break

            if has_integral and not has_continuous:
                cat = 3
            elif has_continuous and not has_integral:
                cat = 2
            else:
                cat = 1

            scores.append(cat * self.scaling)
        return scores

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        This rule does not affect variable ordering. To be compatible with the block
        reordering framework, we return a fixed tuple.
        """
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the score for a single constraint as a tuple.
        This method leverages the existing score_constraints method by applying it
        to a singleton list.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars, obj_coeffs, bounds, A, [constraints[idx]], rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block defined by var_indices and constr_indices based on the 
        constraint composition score computed on the submatrix.
        
        The process is:
          1. Construct sub-lists for variables, bounds, and constraints corresponding to the current block.
          2. Extract the submatrix of A corresponding to these indices.
          3. Call score_constraints on the sub-lists and submatrix to compute scores relative to the block.
          4. Group the original constraint indices by these computed scores.
          5. Since this rule does not differentiate variables (they always score 0), 
             group all variable indices together.
          6. Form the partition map as the Cartesian product of these groups.
        
        Returns a dictionary mapping labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Construct sub-lists for the current block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]  # Not used by this rule, but included for consistency.
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the current block.
        submatrix = A[constr_indices, :][:, var_indices]
        
        # Compute constraint scores on the submatrix using the existing score_constraints method.
        sub_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        
        # Group the original constraint indices by their computed scores.
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_scores):
            constr_groups[score].append(idx)
            
        # All variables are grouped together, as this rule does not differentiate them.
        var_groups = {0: list(var_indices)}
        
        # Form the partition map as the Cartesian product of variable groups and constraint groups.
        partition_map = {}
        label = 0
        for vscore, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
                
        return partition_map
