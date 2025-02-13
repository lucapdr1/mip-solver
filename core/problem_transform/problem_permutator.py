import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import coo_matrix
from utils.logging_handler import LoggingHandler

class ProblemPermutator:
    def __init__(self, gp_env, original_model):
        self.gp_env = gp_env
        self.logger = LoggingHandler().get_logger()
        self.original_model = original_model

    def permutation_matrix(self, perm, sparse=True):
        """
        Build a permutation matrix P from a permutation array perm,
        where perm[i] = j indicates that row i of P has a 1 in column j.
        
        If sparse=True, return a SciPy COO sparse matrix;
        otherwise, return a NumPy dense array.
        """
        n = len(perm)
        row_idx = np.arange(n)
        col_idx = perm
        data = np.ones(n, dtype=np.float64)
        
        if sparse:
            return coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            P = np.zeros((n, n))
            P[row_idx, col_idx] = 1
            return P 

    def create_permuted_problem(self):
        """
        Create a permuted Gurobi model, returning:
          - permuted_model: a new Gurobi model with permuted variables & constraints
          - var_permutation: the permutation array for columns
          - constr_permutation: the permutation array for rows
          - P_col: the column-permutation matrix
          - P_row: the row-permutation matrix
        """
        # --- Generate random permutations ---
        num_vars = self.original_model.NumVars
        num_constrs = self.original_model.NumConstrs
        var_permutation = np.random.permutation(num_vars)
        constr_permutation = np.random.permutation(num_constrs)

        # --- Apply permutation using the helper method ---
        permuted_model = self.apply_permutation(self.original_model, var_permutation, constr_permutation)

        # Build permutation matrices (for analysis or reconstruction if needed)
        P_col = self.permutation_matrix(var_permutation, sparse=True)
        P_row = self.permutation_matrix(constr_permutation, sparse=True)
        return permuted_model, var_permutation, constr_permutation, P_col, P_row

    def apply_permutation(self, model, var_permutation, constr_permutation):
        """
        Reorder the given model according to the provided variable and constraint permutations.
        
        Parameters:
            model : gurobipy.Model
                The model to be reordered.
            var_permutation : array-like
                The new ordering for variables (indices referring to model.getVars()).
            constr_permutation : array-like
                The new ordering for constraints (indices referring to model.getConstrs()).
        
        Returns:
            new_model : gurobipy.Model
                A new model with variables and constraints reordered accordingly.
        """
        original_vars = model.getVars()
        constrs = model.getConstrs()
        A = model.getA()  # Original constraint matrix

        num_vars = len(original_vars)
        num_constrs = len(constrs)

        # Create a new model to hold the permuted variables and constraints.
        new_model = gp.Model(env=self.gp_env)

        # --- Reorder variables ---
        new_vars = []
        for new_idx, old_idx in enumerate(var_permutation):
            old_var = original_vars[old_idx]
            new_var = new_model.addVar(
                lb=old_var.LB,
                ub=old_var.UB,
                obj=old_var.Obj,
                vtype=old_var.VType,
                name=f"x{new_idx+1}"
            )
            new_vars.append(new_var)
        new_model.update()

        # --- Reorder constraints ---
        # Convert A to CSC format for easier slicing.
        A_csc = A.tocsc()
        # First, reorder the columns (variables)
        A_tmp = A_csc[:, var_permutation]
        # Then, reorder the rows (constraints)
        A_perm = A_tmp[constr_permutation, :]
        A_perm_csr = A_perm.tocsr()

        for new_row_idx in range(num_constrs):
            old_constr = constrs[constr_permutation[new_row_idx]]
            expr = gp.LinExpr()
            row = A_perm_csr.getrow(new_row_idx)
            for col_idx, coeff in zip(row.indices, row.data):
                expr.add(new_vars[col_idx], float(coeff))
            rhs_value = old_constr.RHS
            if old_constr.Sense == GRB.LESS_EQUAL:
                new_model.addConstr(expr <= rhs_value)
            elif old_constr.Sense == GRB.GREATER_EQUAL:
                new_model.addConstr(expr >= rhs_value)
            else:  # GRB.EQUAL
                new_model.addConstr(expr == rhs_value)
        new_model.ModelSense = model.ModelSense
        new_model.update()
        return new_model

    def permutation_distance(self, row_perm1, col_perm1, row_perm2, col_perm2,
                             row_dist_method="kendall_tau",
                             col_dist_method="kendall_tau",
                             alpha=1.0, beta=1.0):
        """
        Compute a combined distance between two pairs of permutations.
        row_dist_method and col_dist_method can be 'kendall_tau', 'hamming', etc.
        """
        if row_dist_method == "kendall_tau":
            dist_fn_rows = self.kendall_tau_distance
        else:
            dist_fn_rows = self.hamming_distance

        if col_dist_method == "kendall_tau":
            dist_fn_cols = self.kendall_tau_distance
        else:
            dist_fn_cols = self.hamming_distance

        d_rows = dist_fn_rows(row_perm1, row_perm2)
        d_cols = dist_fn_cols(col_perm1, col_perm2)
        return alpha * d_rows + beta * d_cols

    def hamming_distance(self, perm1, perm2):
        if len(perm1) != len(perm2):
            raise ValueError("Permutations must be the same length.")
        return np.sum(np.array(perm1) != np.array(perm2))

    def kendall_tau_distance(self, perm1, perm2):
        n = len(perm1)
        if len(perm2) != n:
            raise ValueError("Permutations must be the same length.")

        pos_in_perm2 = [0] * n
        for i, val in enumerate(perm2):
            pos_in_perm2[val] = i

        distance = 0
        for i in range(n):
            for j in range(i+1, n):
                if pos_in_perm2[perm1[i]] > pos_in_perm2[perm1[j]]:
                    distance += 1
        return distance
