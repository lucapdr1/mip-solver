# core/problem_permutator.py

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
            # Build a COO-format sparse matrix
            return coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            # Build a dense NumPy array
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
        # --- Prepare references to original data ---
        original_vars = self.original_model.getVars()
        constrs = self.original_model.getConstrs()
        A = self.original_model.getA()  # constraint matrix in CSR or CSC form

        num_vars = len(original_vars)
        num_constrs = len(constrs)

        # --- Generate random permutations ---
        var_permutation = np.random.permutation(num_vars)
        constr_permutation = np.random.permutation(num_constrs)

        # --- Build permutation matrices (sparse by default) ---
        P_col = self.permutation_matrix(var_permutation, sparse=True)   # For columns - Q
        P_row = self.permutation_matrix(constr_permutation, sparse=True)  # For rows - P

        # Make sure A is in CSC (or CSR) for easy slicing
        A_csc = A.tocsc()

        # --- Apply column permutation: reorder columns of A by var_permutation ---
        # A_tmp will be the same number of rows, but columns are now permuted
        A_tmp = A_csc[:, var_permutation]

        # --- Apply row permutation: reorder rows of A by constr_permutation ---
        # A_perm is now the fully permuted matrix
        A_perm = A_tmp[constr_permutation, :]

        # Now we build a new model with these permutations
        permuted_model = gp.Model(env=self.gp_env)

        # ------------------------------------------------------
        # 1) Create new variables in the permuted order
        # ------------------------------------------------------
        new_vars = []
        for new_idx, old_idx in enumerate(var_permutation):
            old_var = original_vars[old_idx]
            # Copy the variable bounds, objective, and type
            new_var = permuted_model.addVar(
                lb=old_var.LB,
                ub=old_var.UB,
                obj=old_var.Obj,
                vtype=old_var.VType,
                name=f"x{new_idx}"
            )
            new_vars.append(new_var)

        # ------------------------------------------------------
        # 2) Create new constraints in the permuted order
        # ------------------------------------------------------
        # Convert A_perm to CSR to iterate rows easily
        A_perm_csr = A_perm.tocsr()

        for new_row_idx in range(num_constrs):
            # This row in A_perm corresponds to the constraint
            # whose old index is constr_permutation[new_row_idx].
            old_constr_idx = constr_permutation[new_row_idx]
            old_constr = constrs[old_constr_idx]

            expr = gp.LinExpr()
            row = A_perm_csr.getrow(new_row_idx)  # row in permuted matrix

            for col_idx, val in zip(row.indices, row.data):
                # col_idx is the new variable index, since we've permuted columns
                # new_vars[col_idx] is the actual Gurobi var object
                expr.add(new_vars[col_idx], float(val))

            # Add the constraint with the same sense and RHS as in original
            rhs_value = old_constr.RHS
            if old_constr.Sense == GRB.LESS_EQUAL:
                permuted_model.addConstr(expr <= rhs_value)
            elif old_constr.Sense == GRB.GREATER_EQUAL:
                permuted_model.addConstr(expr >= rhs_value)
            else:  # old_constr.Sense == GRB.EQUAL
                permuted_model.addConstr(expr == rhs_value)

        # ------------------------------------------------------
        # 3) Copy model sense (Minimize/Maximize)
        # ------------------------------------------------------
        permuted_model.ModelSense = self.original_model.ModelSense

        # Finalize the model
        permuted_model.update()

        # ------------------------------------------------------
        # Return everything needed to analyze or reconstruct
        # ------------------------------------------------------
        return permuted_model, var_permutation, constr_permutation, P_col, P_row
    
    def permutation_distance(self, row_perm1, col_perm1, row_perm2, col_perm2,
                             row_dist_method="kendall_tau",
                             col_dist_method="kendall_tau",
                             alpha=1.0, beta=1.0):
        """
        Compute a combined distance between two pairs of permutations.
        row_dist_method and col_dist_method can be 'kendall_tau', 'hamming', etc.
        """
        # pick the actual function to use for rows
        if row_dist_method == "kendall_tau":
            dist_fn_rows = self.kendall_tau_distance
        else:
            dist_fn_rows = self.hamming_distance  # or others

        # pick the actual function for columns
        if col_dist_method == "kendall_tau":
            dist_fn_cols = self.kendall_tau_distance
        else:
            dist_fn_cols = self.hamming_distance

        d_rows = dist_fn_rows(row_perm1, row_perm2)
        d_cols = dist_fn_cols(col_perm1, col_perm2)
        return alpha * d_rows + beta * d_cols
    

    def hamming_distance(perm1, perm2):
        if len(perm1) != len(perm2):
            raise ValueError("Permutations must be the same length.")
        return np.sum(np.array(perm1) != np.array(perm2))

    def kendall_tau_distance(perm1, perm2):
        n = len(perm1)
        if len(perm2) != n:
            raise ValueError("Permutations must be the same length.")

        # Map elements in perm2 to their positions
        pos_in_perm2 = [0]*n
        for i, val in enumerate(perm2):
            pos_in_perm2[val] = i

        # Count inversions
        distance = 0
        for i in range(n):
            for j in range(i+1, n):
                if pos_in_perm2[perm1[i]] > pos_in_perm2[perm1[j]]:
                    distance += 1
        return distance



