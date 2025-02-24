import numpy as np
import math
from collections import defaultdict
from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB

class ConstraintIntegerCountRule(OrderingRule):
    """
    Counts the number of integer variables (BINARY, INTEGER, or SEMIINT) appearing
    in each constraint.
    """
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule does not affect variable scores.
       return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        For each constraint (row) in A, count the number of variables that appear with a
        nonzero coefficient and whose VType is one of GRB.BINARY, GRB.INTEGER, or GRB.SEMIINT.
        """
        scores = []
        num_constraints = A.shape[0]
        # Convert A to CSR format if possible.
        A = A.tocsr() if hasattr(A, "tocsr") else A

        for row_idx in range(num_constraints):
            # Get the row as a flat numpy array.
            row_data = A[row_idx, :]
            if hasattr(row_data, "toarray"):
                row_data = row_data.toarray().flatten()
            else:
                row_data = np.ravel(row_data)
            
            # Find nonzero coefficient indices.
            nz_indices = row_data.nonzero()[0]
            integer_count = 0
            for j in nz_indices:
                # Check variable type using GRB constants.
                if vars[j].VType in [GRB.BINARY, GRB.INTEGER, GRB.SEMIINT]:
                    integer_count += 1
            scores.append(integer_count)
        return scores

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a tuple containing the score for a single variable.
        (For this rule, variables are not scored, so it is always 0.)
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a tuple containing the score for a single constraint.
        Here the score is the number of integer variables (BINARY, INTEGER, SEMIINT)
        that appear in the constraint.
        """
        A_mod = A.tocsr() if hasattr(A, "tocsr") else A
        row_data = A_mod[idx, :]
        if hasattr(row_data, "toarray"):
            row_data = row_data.toarray().flatten()
        else:
            row_data = np.ravel(row_data)
        nz_indices = row_data.nonzero()[0]
        integer_count = 0
        for j in nz_indices:
            if vars[j].VType in [GRB.BINARY, GRB.INTEGER, GRB.SEMIINT]:
                integer_count += 1
        return (integer_count,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) using the integer count.
        
        Process:
          1. Construct sub-lists for variables, bounds, constraints, and (optionally) rhs.
          2. Extract the submatrix corresponding to these indices.
          3. Compute variable and constraint scores on the submatrix.
          4. Group the original indices by their computed scores.
          5. Form the partition as the Cartesian product of the variable and constraint groups.
        
        Returns a dictionary mapping labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Sub-lists for the block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the block.
        submatrix = A[constr_indices, :][:, var_indices]

        # Compute scores on the submatrix.
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

        # Group original variable indices by their computed scores.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)

        # Group original constraint indices by their computed scores.
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_groups[score].append(idx)

        # Form the partition as the Cartesian product of the groups.
        partition_map = {}
        label = 0
        for vscore, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map


class ConstraintContinuousCountRule(OrderingRule):
    """
    Counts the number of continuous variables (CONTINUOUS or SEMICONT) appearing
    in each constraint.
    """
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule does not affect variable scores.
       return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        For each constraint (row) in A, count the number of variables that appear with a
        nonzero coefficient and whose VType is either GRB.CONTINUOUS or GRB.SEMICONT.
        """
        scores = []
        num_constraints = A.shape[0]
        A = A.tocsr() if hasattr(A, "tocsr") else A

        for row_idx in range(num_constraints):
            row_data = A[row_idx, :]
            if hasattr(row_data, "toarray"):
                row_data = row_data.toarray().flatten()
            else:
                row_data = np.ravel(row_data)
            
            nz_indices = row_data.nonzero()[0]
            continuous_count = 0
            for j in nz_indices:
                if vars[j].VType in [GRB.CONTINUOUS, GRB.SEMICONT]:
                    continuous_count += 1
            scores.append(continuous_count)
        return scores

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a tuple containing the score for a single variable.
        (For this rule, variables are not scored, so it is always 0.)
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a tuple containing the score for a single constraint.
        Here the score is the number of continuous variables (CONTINUOUS, SEMICONT)
        that appear in the constraint.
        """
        A_mod = A.tocsr() if hasattr(A, "tocsr") else A
        row_data = A_mod[idx, :]
        if hasattr(row_data, "toarray"):
            row_data = row_data.toarray().flatten()
        else:
            row_data = np.ravel(row_data)
        nz_indices = row_data.nonzero()[0]
        continuous_count = 0
        for j in nz_indices:
            if vars[j].VType in [GRB.CONTINUOUS, GRB.SEMICONT]:
                continuous_count += 1
        return (continuous_count,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) using the continuous count.
        
        Process:
          1. Construct sub-lists for variables, bounds, constraints, and (optionally) rhs.
          2. Extract the submatrix corresponding to these indices.
          3. Compute variable and constraint scores on the submatrix.
          4. Group the original indices by their computed scores.
          5. Form the partition as the Cartesian product of the variable and constraint groups.
        
        Returns a dictionary mapping labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Sub-lists for the current block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the current block.
        submatrix = A[constr_indices, :][:, var_indices]

        # Compute scores on the submatrix.
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

        # Group original variable indices by their computed scores.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)

        # Group original constraint indices by their computed scores.
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_groups[score].append(idx)

        # Form the partition map as the Cartesian product of the groups.
        partition_map = {}
        label = 0
        for vscore, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map
class BothBoundsFiniteCountRule(OrderingRule):
    """
    Counts variables that have both bounds finite.
    
    For a single variable:
      - Score 1 if both lower and upper bounds are finite.
      - Otherwise, score 0.
    For a constraint, the rule sums the scores (over nonzero coefficients).
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]
            score = 1 if (not math.isinf(lb) and not math.isinf(ub)) else 0
            scores.append(score * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        num_constraints = A.shape[0]
        A = A.tocsr() if hasattr(A, "tocsr") else A
        for i in range(num_constraints):
            # Get the row as a flat array.
            row = (A[i, :].toarray().ravel() 
                   if hasattr(A[i, :], "toarray") 
                   else np.ravel(A[i, :]))
            count = 0
            for j in range(len(vars)):
                if float(row[j]) != 0:
                    lb, ub = bounds[j]
                    if not math.isinf(lb) and not math.isinf(ub):
                        count += 1
            scores.append(count * self.scaling)
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        score = self.score_variables(
                    [vars[idx]],
                    obj_coeffs[idx:idx+1],
                    [bounds[idx]],
                    A, constraints, rhs
                )[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        A_mod = A.tocsr() if hasattr(A, "tocsr") else A
        row = (A_mod[idx, :].toarray().ravel() 
               if hasattr(A_mod[idx, :], "toarray") 
               else np.ravel(A_mod[idx, :]))
        count = 0
        for j in range(len(vars)):
            if float(row[j]) != 0:
                lb, ub = bounds[j]
                if not math.isinf(lb) and not math.isinf(ub):
                    count += 1
        return (count * self.scaling,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        # Build sub-lists for the block.
        vars_sub   = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub    = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix.
        submatrix = A[constr_indices, :][:, var_indices]

        # Compute scores on the submatrix.
        sub_var_scores   = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

        # Group original indices by computed scores.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_groups[score].append(idx)

        # Form the partition as the Cartesian product.
        partition_map = {}
        label = 0
        for score_v, vgroup in var_groups.items():
            for score_c, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map
class OneBoundFiniteCountRule(OrderingRule):
    """
    Counts variables that have exactly one finite bound.
    
    For a single variable:
      - Score 1 if exactly one of the bounds is finite.
      - Otherwise, score 0.
    For a constraint, the rule sums the scores over variables with nonzero coefficients.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]
            finite_count = (0 if math.isinf(lb) else 1) + (0 if math.isinf(ub) else 1)
            score = 1 if finite_count == 1 else 0
            scores.append(score * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        num_constraints = A.shape[0]
        A = A.tocsr() if hasattr(A, "tocsr") else A
        for i in range(num_constraints):
            row = (A[i, :].toarray().ravel() 
                   if hasattr(A[i, :], "toarray") 
                   else np.ravel(A[i, :]))
            count = 0
            for j in range(len(vars)):
                if float(row[j]) != 0:
                    lb, ub = bounds[j]
                    finite_count = (0 if math.isinf(lb) else 1) + (0 if math.isinf(ub) else 1)
                    if finite_count == 1:
                        count += 1
            scores.append(count * self.scaling)
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        score = self.score_variables(
                    [vars[idx]],
                    obj_coeffs[idx:idx+1],
                    [bounds[idx]],
                    A, constraints, rhs
                )[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        A_mod = A.tocsr() if hasattr(A, "tocsr") else A
        row = (A_mod[idx, :].toarray().ravel() 
               if hasattr(A_mod[idx, :], "toarray") 
               else np.ravel(A_mod[idx, :]))
        count = 0
        for j in range(len(vars)):
            if float(row[j]) != 0:
                lb, ub = bounds[j]
                finite_count = (0 if math.isinf(lb) else 1) + (0 if math.isinf(ub) else 1)
                if finite_count == 1:
                    count += 1
        return (count * self.scaling,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        vars_sub   = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub    = [rhs[i] for i in constr_indices] if rhs is not None else None

        submatrix = A[constr_indices, :][:, var_indices]
        sub_var_scores    = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

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
class BothBoundsInfiniteCountRule(OrderingRule):
    """
    Counts variables that have both bounds infinite.
    
    For a single variable:
      - Score 1 if both lower and upper bounds are infinite.
      - Otherwise, score 0.
    For a constraint, the rule sums the scores over variables with nonzero coefficients.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]
            score = 1 if (math.isinf(lb) and math.isinf(ub)) else 0
            scores.append(score * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        num_constraints = A.shape[0]
        A = A.tocsr() if hasattr(A, "tocsr") else A
        for i in range(num_constraints):
            row = (A[i, :].toarray().ravel() 
                   if hasattr(A[i, :], "toarray") 
                   else np.ravel(A[i, :]))
            count = 0
            for j in range(len(vars)):
                if float(row[j]) != 0:
                    lb, ub = bounds[j]
                    if math.isinf(lb) and math.isinf(ub):
                        count += 1
            scores.append(count * self.scaling)
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        score = self.score_variables(
                    [vars[idx]],
                    obj_coeffs[idx:idx+1],
                    [bounds[idx]],
                    A, constraints, rhs
                )[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        A_mod = A.tocsr() if hasattr(A, "tocsr") else A
        row = (A_mod[idx, :].toarray().ravel() 
               if hasattr(A_mod[idx, :], "toarray") 
               else np.ravel(A_mod[idx, :]))
        count = 0
        for j in range(len(vars)):
            if float(row[j]) != 0:
                lb, ub = bounds[j]
                if math.isinf(lb) and math.isinf(ub):
                    count += 1
        return (count * self.scaling,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        vars_sub   = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub    = [rhs[i] for i in constr_indices] if rhs is not None else None

        submatrix = A[constr_indices, :][:, var_indices]
        sub_var_scores    = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

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
