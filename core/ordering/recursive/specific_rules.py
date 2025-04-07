from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class AllCoefficientsOneRule(OrderingRule):
    """
    Scores variables and constraints based on all non-zero coefficients being 1.
    - Variables: Score is scaling if all non-zero coefficients in the column are 1.
    - Constraints: Score is scaling if all non-zero coefficients in the row are 1.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling
        self.tol = tol

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        num_vars = A.shape[1]
        scores = np.zeros(num_vars, dtype=int)
        
        if hasattr(A, "tocsc"):
            # Get column pointers and compute number of nonzeros per column.
            indptr = A_csc.indptr
            counts = np.diff(indptr)  # length = num_vars
            # Compute absolute deviation from 1 for all nonzero elements.
            diff = np.abs(A_csc.data - 1.0)
            # Preallocate an array for the maximum deviation in each column.
            max_diff = np.empty(num_vars, dtype=float)
            # Create a mask for non-empty columns.
            nonempty_mask = counts > 0
            nonempty_idx = np.where(nonempty_mask)[0]
            # Loop over non-empty columns to compute maximum deviation.
            for j in nonempty_idx:
                max_diff[j] = np.max(diff[indptr[j]:indptr[j+1]])
            # For empty columns, set the maximum deviation to np.inf.
            max_diff[~nonempty_mask] = np.inf
                
            # A column scores 'scaling' if its maximum deviation is within tolerance.
            scores = np.where(max_diff <= self.tol, self.scaling, 0)
        else:
            # Fallback using a Python loop if A does not support toscsc.
            for j in range(num_vars):
                column = A[:, j].flatten()
                non_zero = np.abs(column) > self.tol
                non_zero_data = column[non_zero]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[j] = self.scaling
        return scores


    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        num_constraints = A.shape[0]
        scores = np.zeros(num_constraints, dtype=int)
        
        if hasattr(A, "tocsr"):
            
            indptr = A_csr.indptr
            counts = np.diff(indptr)
            diff = np.abs(A_csr.data - 1.0)
            max_diff = np.empty(num_constraints, dtype=float)
            nonempty_mask = counts > 0
            nonempty_idx = np.where(nonempty_mask)[0]
            for i in nonempty_idx:
                max_diff[i] = np.max(diff[indptr[i]:indptr[i+1]])
            max_diff[~nonempty_mask] = np.inf
            scores = np.where(max_diff <= self.tol, self.scaling, 0)
        else:
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_data = row[non_zero]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[i] = self.scaling
        return scores


    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the (constant) score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the sign-based score for a single constraint (row) as a one-element tuple.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Compute scores on the sub-block.
        sub_var_scores = np.array(self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        sub_constr_scores = np.array(self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # Ensure original indices are numpy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        
        # Group variable indices by their score.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {}
        for score in unique_var_scores:
            mask = (sub_var_scores == score)
            var_partitions[score] = var_indices[mask]
        
        # Group constraint indices by their score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {}
        for score in unique_constr_scores:
            mask = (sub_constr_scores == score)
            constr_partitions[score] = constr_indices[mask]
        
        # Form the partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v, var_group in var_partitions.items():
            for score_c, constr_group in constr_partitions.items():
                partition_map[label] = (var_group, constr_group)
                label += 1
        return partition_map
    

class AllBinaryVariablesRule(OrderingRule):
    """
    Scores variables and constraints based on binary variables.
    - Variables: Score is scaling if the variable is binary (bounds [0,1]).
    - Constraints: Score is scaling if all variables in the constraint are binary.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling
        self.tol = tol


    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # Convert bounds to a NumPy array (shape: [num_vars, 2])
        bounds_arr = np.array(bounds, dtype=float)
        # Check if lower bound is close to 0 and upper bound close to 1
        lb_close = np.isclose(bounds_arr[:, 0], 0.0, atol=self.tol)
        ub_close = np.isclose(bounds_arr[:, 1], 1.0, atol=self.tol)
        is_binary = lb_close & ub_close
        # Score is scaling if binary, 0 otherwise.
        scores = np.where(is_binary, self.scaling, 0)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Number of constraints
        num_constraints = A.shape[0]
        # Convert bounds to a NumPy array (shape: [num_vars, 2])
        bounds_arr = np.array(bounds, dtype=float)
        # Precompute a boolean array for binary variables.
        binary_vars = (np.isclose(bounds_arr[:, 0], 0.0, atol=self.tol) &
                       np.isclose(bounds_arr[:, 1], 1.0, atol=self.tol))
        
        scores = np.zeros(num_constraints, dtype=int)
        
        if hasattr(A, "tocsr"):
            
            for i in range(num_constraints):
                start = A_csr.indptr[i]
                end = A_csr.indptr[i+1]
                # Get the indices of nonzero entries in row i.
                row_indices = A_csr.indices[start:end]
                if row_indices.size > 0 and np.all(binary_vars[row_indices]):
                    scores[i] = self.scaling
        else:
            # Fallback for dense or other formats.
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_indices = np.where(non_zero)[0]
                if non_zero_indices.size > 0 and np.all(binary_vars[non_zero_indices]):
                    scores[i] = self.scaling
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the (constant) score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the sign-based score for a single constraint (row) as a one-element tuple.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Compute scores on the sub-block.
        sub_var_scores = np.array(self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        sub_constr_scores = np.array(self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # Group the original variable indices by their computed score using NumPy.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {}
        for score in unique_var_scores:
            mask = (sub_var_scores == score)
            var_partitions[score] = var_indices[mask]
        
        # Similarly group the constraint indices.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {}
        for score in unique_constr_scores:
            mask = (sub_constr_scores == score)
            constr_partitions[score] = constr_indices[mask]
        
        # Form the partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_partitions[score_v], constr_partitions[score_c])
                label += 1
                    
        return partition_map
class SetPackingRHSRule(OrderingRule):
    """
    Scores variables and constraints for set-packing problems.
    
    - Variables: Score is set to `scaling` if the variable is binary (bounds [0,1]).
    - Constraints: If all variables in the constraint are binary, the score is computed
      as `scaling * rhs`, using the RHS coefficient since scaling the constraint is not meaningful.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling
        self.tol = tol

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Convert bounds to NumPy array (shape: [num_vars, 2])
        bounds_arr = np.array(bounds, dtype=float)
        # Check if lower bound is close to 0 and upper bound is close to 1
        lb_close = np.isclose(bounds_arr[:, 0], 0.0, atol=self.tol)
        ub_close = np.isclose(bounds_arr[:, 1], 1.0, atol=self.tol)
        is_binary = lb_close & ub_close
        # For binary variables, assign the scaling factor as score; else, 0.
        scores = np.where(is_binary, self.scaling, 0)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        num_constraints = A.shape[0]
        # Determine binary status for each variable using the bounds.
        bounds_arr = np.array(bounds, dtype=float)
        binary_vars = np.isclose(bounds_arr[:, 0], 0.0, atol=self.tol) & \
                      np.isclose(bounds_arr[:, 1], 1.0, atol=self.tol)
        
        scores = np.zeros(num_constraints, dtype=float)
        
        # Use CSR format if available.
        if hasattr(A, "tocsr"):
            for i in range(num_constraints):
                start = A_csr.indptr[i]
                end = A_csr.indptr[i+1]
                row_indices = A_csr.indices[start:end]
                # If the constraint involves any variables and all are binary,
                # score it as scaling * rhs[i].
                if row_indices.size > 0 and np.all(binary_vars[row_indices]):
                    scores[i] = self.scaling * rhs[i]
        else:
            # Fallback for dense or other matrix representations.
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_indices = np.where(non_zero)[0]
                if non_zero_indices.size > 0 and np.all(binary_vars[non_zero_indices]):
                    scores[i] = self.scaling * rhs[i]
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Score a single variable.
        score = self.score_variables(
            vars, 
            obj_coeffs, 
            bounds, 
            A, A_csc, A_csr, constraints, rhs
        )[idx]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Score a single constraint.
        score = self.score_constraints(
            vars, 
            obj_coeffs, 
            bounds, 
            A, A_csc, A_csr, constraints, rhs
        )[idx]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Compute scores over all variables and constraints.
        sub_var_scores = np.array(self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        sub_constr_scores = np.array(self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # Group variable indices by their computed score.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}
        
        # Group constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form a partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_partitions[score_v], constr_partitions[score_c])
                label += 1
        return partition_map
class UnscaledObjectiveOrderingRule(OrderingRule):
    """
    Orders variables based on their objective coefficients, but only when the constraint
    matrix is in its unscaled (canonical) form—that is, every nonzero coefficient in A is 1.
    
    This rule is particularly useful in set-packing problems, where all variables are binary
    and a higher objective coefficient indicates a higher priority. The rule first checks 
    that the instance is unscaled; if so, each variable is scored as:
    
        score = scaling * obj_coefficient.
    
    If the instance is scaled (i.e. some coefficients in A differ from 1), the rule returns
    a zero score for all variables.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling
        self.tol = tol

    def is_unscaled(self, A, A_csr):
        """
        Returns True if all nonzero coefficients in the matrix A are 1 (within a given tolerance).
        Works for both CSR (sparse) and dense matrices.
        """
        # Check for sparse matrix (assume CSR format if possible)
        if hasattr(A, "tocsr"):
            # If there are no nonzero entries, consider it unscaled.
            if A_csr.nnz == 0:
                return True
            # Check that all nonzero entries are 1 (up to tolerance)
            return np.all(np.isclose(A_csr.data, 1.0, atol=self.tol))
        else:
            # For a dense numpy array
            non_zero = np.abs(A) > self.tol
            if not np.any(non_zero):
                return True
            return np.all(np.isclose(A[non_zero], 1.0, atol=self.tol))
    
    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        If the instance is unscaled (all nonzero A coefficients are 1), score each variable
        using its objective coefficient multiplied by a scaling factor. Otherwise, return 0.
        """
        if self.is_unscaled(A, A_csr):
            scores = self.scaling * np.array(obj_coeffs, dtype=float)
        else:
            scores = np.zeros(len(obj_coeffs), dtype=float)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule is designed only to order variables.
        num_constraints = A.shape[0]
        return np.zeros(num_constraints, dtype=float)

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        score = self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)[idx]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        score = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)[idx]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Computes scores for all variables and constraints, then partitions the indices based
        on the unique scores. Since constraints are not scored by this rule, they all fall
        into a single group.
        """
        sub_var_scores = np.array(self.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        sub_constr_scores = np.array(self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # Partition variables by unique scores.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}
        
        # Partition constraints by unique scores (likely just one partition of zeros).
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form a partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_partitions[score_v], constr_partitions[score_c])
                label += 1
        return partition_map
