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

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_vars = A.shape[1]
        scores = np.zeros(num_vars, dtype=int)
        
        if hasattr(A, "tocsc"):
            A_csc = A.tocsc()
            for j in range(num_vars):
                start = A_csc.indptr[j]
                end = A_csc.indptr[j+1]
                data = A_csc.data[start:end]
                mask = np.abs(data) > self.tol
                non_zero_data = data[mask]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[j] = self.scaling
        else:
            for j in range(num_vars):
                column = A[:, j].flatten()
                non_zero = np.abs(column) > self.tol
                non_zero_data = column[non_zero]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[j] = self.scaling
        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = np.zeros(num_constraints, dtype=int)
        
        if hasattr(A, "tocsr"):
            A_csr = A.tocsr()
            for i in range(num_constraints):
                start = A_csr.indptr[i]
                end = A_csr.indptr[i+1]
                data = A_csr.data[start:end]
                mask = np.abs(data) > self.tol
                non_zero_data = data[mask]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[i] = self.scaling
        else:
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_data = row[non_zero]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[i] = self.scaling
        return scores.tolist()

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the (constant) score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
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

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constraints_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None
        
        submatrix = A[constr_indices, :][:, var_indices]
        
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constraints_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constraints_sub, rhs_sub)
        
        var_partitions = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_partitions[score].append(idx)
            
        constr_partitions = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_partitions[score].append(idx)
            
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

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_vars = len(vars)
        scores = np.zeros(num_vars, dtype=int)
        for j in range(num_vars):
            lb, ub = bounds[j]
            if np.isclose(lb, 0.0, atol=self.tol) and np.isclose(ub, 1.0, atol=self.tol):
                scores[j] = self.scaling
        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = np.zeros(num_constraints, dtype=int)
        binary_vars = np.array([np.isclose(b[0], 0.0, atol=self.tol) and np.isclose(b[1], 1.0, atol=self.tol) for b in bounds])
        
        if hasattr(A, "tocsr"):
            A_csr = A.tocsr()
            for i in range(num_constraints):
                start = A_csr.indptr[i]
                end = A_csr.indptr[i+1]
                data = A_csr.data[start:end]
                indices = A_csr.indices[start:end]
                mask = np.abs(data) > self.tol
                non_zero_indices = indices[mask]
                if non_zero_indices.size > 0 and np.all(binary_vars[non_zero_indices]):
                    scores[i] = self.scaling
        else:
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_indices = np.where(non_zero)[0]
                if non_zero_indices.size > 0 and np.all(binary_vars[non_zero_indices]):
                    scores[i] = self.scaling
        return scores.tolist()

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the (constant) score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
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

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constraints_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None
        
        submatrix = A[constr_indices, :][:, var_indices]
        
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constraints_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constraints_sub, rhs_sub)
        
        var_partitions = defaultdict(list)
        constr_partitions = defaultdict(list)
        
        for idx, score in zip(var_indices, sub_var_scores):
            var_partitions[score].append(idx)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_partitions[score].append(idx)
            
        partition_map = {}
        label = 0
        for score_v in var_partitions:
            for score_c in constr_partitions:
                partition_map[label] = (var_partitions[score_v], constr_partitions[score_c])
                label += 1
        return partition_map