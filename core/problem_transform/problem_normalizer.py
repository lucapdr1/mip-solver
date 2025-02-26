import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from math import gcd
from utils.logging_handler import LoggingHandler

class ProblemNormalizer:
    def __init__(self):
        """
        Initialize the ProblemNormalizer with a logger instance.
        """
        self.logger = LoggingHandler().get_logger()

    def _extract_problem_data(self, model):
        """
        Extract problem data from a Gurobi model.

        Returns:
            tuple: (A, obj_coeffs, rhs, bounds, vtypes)
                - A is a scipy.sparse.csr_matrix of the constraints,
                - obj_coeffs is a NumPy array of objective coefficients,
                - rhs is a NumPy array of constraint right-hand sides,
                - bounds is a list of tuples (lb, ub) for each variable,
                - vtypes is a list of variable types.
        """
        self.logger.lazy_debug("Extracting problem data from the input Gurobi model.")
        constrs = model.getConstrs()
        vars = model.getVars()
        n_vars = len(vars)
        n_constrs = len(constrs)

        rows = []
        cols = []
        data = []
        rhs = []
        constraint_senses = []

        var_to_index = {var.VarName: i for i, var in enumerate(vars)}

        for i, constr in enumerate(constrs):
            row_expr = model.getRow(constr)
            for j in range(row_expr.size()):
                var = row_expr.getVar(j)
                coeff = row_expr.getCoeff(j)
                idx = var_to_index[var.VarName]
                rows.append(i)
                cols.append(idx)
                data.append(coeff)
            rhs.append(constr.RHS)
            constraint_senses.append(constr.Sense)

        A = sp.csr_matrix((data, (rows, cols)), shape=(n_constrs, n_vars))
        obj_coeffs = np.array([var.Obj for var in vars])
        bounds = [(var.LB, var.UB) for var in vars]
        rhs = np.array(rhs)
        vtypes = [var.VType for var in vars]

        self.logger.lazy_debug("Extraction complete: %d constraints, %d variables.", n_constrs, n_vars)
        return A, obj_coeffs, rhs, bounds, vtypes, constraint_senses

    def _compute_scaling_factors(self, A, vtypes, tol=1e-9):
        """
        Compute row and column scaling factors from A.
        For discrete columns (and rows) corresponding to integer/binary variables,
        we use the GCD of the (rounded) absolute coefficients from discrete variables only.
        For continuous ones (or rows/columns with no discrete variables), we use the L2 norm.

        Returns:
            tuple: (row_scaling, col_scaling, row_norms, col_norms)
        """
        A_csr = A.tocsr()
        n_constrs, n_vars = A.shape
        row_scaling = np.ones(n_constrs)
        col_scaling = np.ones(n_vars)
        row_norms = np.zeros(n_constrs)
        col_norms = np.zeros(n_vars)

        # Process each column.
        for j in range(n_vars):
            col_data = A_csr.getcol(j).toarray().flatten()
            nonzero = col_data[np.abs(col_data) > tol]
            if len(nonzero) == 0:
                norm = 1.0
            else:
                if vtypes[j] in [GRB.INTEGER, GRB.BINARY]:
                    # For discrete columns, compute GCD over the rounded absolute values.
                    int_vals = [int(round(abs(x))) for x in nonzero if abs(x) > tol]
                    if len(int_vals) > 0:
                        g = int_vals[0]
                        for val in int_vals[1:]:
                            g = gcd(g, val)
                        norm = g if g != 0 else max(int_vals)
                    else:
                        norm = 1.0
                else:
                    norm = np.sqrt(np.sum(np.square(nonzero)))
            col_norms[j] = norm
            col_scaling[j] = 1.0 / norm if norm != 0 else 1.0

        # Process each row.
        for i in range(n_constrs):
            row_data = A_csr.getrow(i).toarray().flatten()
            indices = np.nonzero(np.abs(row_data) > tol)[0]
            if len(indices) == 0:
                norm = 1.0
            else:
                # If there are any discrete variables in this row, compute the GCD only over those coefficients.
                discrete_vals = []
                for j in indices:
                    if vtypes[j] in [GRB.INTEGER, GRB.BINARY]:
                        discrete_vals.append(int(round(abs(row_data[j]))))
                if len(discrete_vals) > 0:
                    g = discrete_vals[0]
                    for val in discrete_vals[1:]:
                        g = gcd(g, val)
                    norm = g if g != 0 else max(discrete_vals)
                else:
                    # Otherwise, use the L2 norm over all coefficients in the row.
                    norm = np.sqrt(np.sum(np.square(row_data[indices])))
            row_norms[i] = norm
            row_scaling[i] = 1.0 / norm if norm != 0 else 1.0

        self.logger.lazy_debug("Row norms (modified): %s", row_norms)
        self.logger.lazy_debug("Column norms (modified): %s", col_norms)
        self.logger.lazy_debug("Row scaling factors (modified): %s", row_scaling)
        self.logger.lazy_debug("Column scaling factors (modified): %s", col_scaling)
        return row_scaling, col_scaling, row_norms, col_norms


    def _normalize_matrix(self, A, row_scaling, col_scaling):
        """
        Apply scaling to A via diagonal matrices.
        """
        D_row = sp.diags(row_scaling)
        D_col = sp.diags(col_scaling)
        A_normalized = D_row @ A.tocsr() @ D_col
        return A_normalized

    def normalize(self, A, obj_coeffs, rhs, bounds, vtypes):
        """
        Normalize the problem data.

        For continuous variables, L2 normalization is used.
        For integer/binary variables, scaling factors are computed via a GCD-based approach.
        The objective is adjusted as: normalized_obj = original_obj / (col_scaling),
        and the RHS is adjusted by row_scaling.
        Bounds are adjusted by dividing by the column scaling.

        Returns:
            tuple: (A_normalized, normalized_obj_coeffs, normalized_rhs, normalized_bounds, scaling_factors)
        """
        self.logger.lazy_debug("Starting normalization of problem data.")
        row_scaling, col_scaling, row_norms, col_norms = self._compute_scaling_factors(A, vtypes)
        A_normalized = self._normalize_matrix(A, row_scaling, col_scaling)

        normalized_obj_coeffs = obj_coeffs / col_scaling
        normalized_rhs = rhs * row_scaling

        normalized_bounds = []
        for i, (lb, ub) in enumerate(bounds):
            s = col_scaling[i]
            if s > 0:
                new_lb = lb / s if lb != -GRB.INFINITY else -GRB.INFINITY
                new_ub = ub / s if ub != GRB.INFINITY else GRB.INFINITY
            else:
                new_lb = ub / s if ub != GRB.INFINITY else -GRB.INFINITY
                new_ub = lb / s if lb != -GRB.INFINITY else GRB.INFINITY
            normalized_bounds.append((new_lb, new_ub))
        
        scaling_factors = (row_scaling, col_scaling, row_norms, col_norms)
        self.logger.lazy_debug("Normalization complete.")
        return A_normalized, normalized_obj_coeffs, normalized_rhs, normalized_bounds, scaling_factors

    def _build_model_from_data(self, A_normalized, obj_coeffs, rhs, bounds, vtypes, constraint_senses, original_model):
        """
        Build a new Gurobi model from normalized problem data.
        """
        self.logger.lazy_debug("Building new Gurobi model from normalized data.")
        new_model = gp.Model("normalized_model")
        n_vars = len(obj_coeffs)
        var_list = []
        for i in range(n_vars):
            lb, ub = bounds[i]
            vtype = vtypes[i]
            var = new_model.addVar(lb=lb, ub=ub, vtype=vtype, name=f"x{i+1}")
            var_list.append(var)
        new_model.update()

        # Insert the objective function using the normalized objective coefficients.
        obj_expr = gp.LinExpr()
        for j, coeff in enumerate(obj_coeffs):
            obj_expr.addTerms(coeff, var_list[j])
        # Use the same ModelSense (minimize or maximize) as in the original model.
        new_model.setObjective(obj_expr, original_model.ModelSense)
        new_model.update()

        m, n = A_normalized.shape
        # Build constraints preserving the original constraint sense.
        for i in range(m):
            row_start = A_normalized.indptr[i]
            row_end = A_normalized.indptr[i+1]
            indices = A_normalized.indices[row_start:row_end]
            values = A_normalized.data[row_start:row_end]
            expr = gp.LinExpr()
            for j, coeff in zip(indices, values):
                expr.addTerms(coeff, var_list[j])
            # Retrieve the original sense (assumed stored earlier in self.original_constraint_senses)
            sense = constraint_senses[i]  # This should have been saved during extraction.
            if sense == GRB.LESS_EQUAL:
                new_model.addConstr(expr <= rhs[i], name=f"c{i+1}")
            elif sense == GRB.GREATER_EQUAL:
                new_model.addConstr(expr >= rhs[i], name=f"c{i+1}")
            else:
                new_model.addConstr(expr == rhs[i], name=f"c{i+1}")
        new_model.update()

        self.logger.lazy_debug("New Gurobi model built successfully.")
        return new_model


    def normalize_model(self, model):
        """
        Normalize an input Gurobi model and return a new normalized Gurobi model.
        The model to be normalized is passed as a parameter.
        """
        self.logger.lazy_debug("Starting normalization of the Gurobi model.")
        A, obj_coeffs, rhs, bounds, vtypes, constraint_senses = self._extract_problem_data(model)
        A_norm, obj_coeffs_norm, rhs_norm, bounds_norm, scaling_factors = self.normalize(A, obj_coeffs, rhs, bounds, vtypes)
        new_model = self._build_model_from_data(A_norm, obj_coeffs_norm, rhs_norm, bounds_norm, vtypes, constraint_senses, model)
        new_model._scaling_factors = scaling_factors
        self.logger.lazy_debug("Gurobi model normalization complete.")
        return new_model
