import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
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

        The method builds a sparse matrix representation of the constraint matrix A,
        extracts the objective coefficients, the right-hand side (RHS) of the constraints,
        and the variable bounds.

        Args:
            model (gp.Model): The original Gurobi model.

        Returns:
            tuple: (A, obj_coeffs, rhs, bounds) where:
                - A is a scipy.sparse.csr_matrix of the constraints,
                - obj_coeffs is a NumPy array of objective coefficients,
                - rhs is a NumPy array of constraint right-hand sides,
                - bounds is a list of tuples (lb, ub) for each variable.
        """
        self.logger.debug("Extracting problem data from the input Gurobi model.")
        constrs = model.getConstrs()
        vars = model.getVars()
        n_vars = len(vars)
        n_constrs = len(constrs)

        rows = []
        cols = []
        data = []
        rhs = []

        # Build a dictionary to map variable names to indices.
        var_to_index = {var.VarName: i for i, var in enumerate(vars)}

        # Loop over constraints to extract coefficients and RHS.
        for i, constr in enumerate(constrs):
            # getRow returns a LinExpr representing the constraint’s left-hand side.
            row_expr = model.getRow(constr)
            for j in range(row_expr.size()):
                var = row_expr.getVar(j)
                coeff = row_expr.getCoeff(j)
                idx = var_to_index[var.VarName]
                rows.append(i)
                cols.append(idx)
                data.append(coeff)
            # Assume the constraint’s RHS is stored in the attribute 'RHS'.
            rhs.append(constr.RHS)

        # Build the sparse constraint matrix.
        A = sp.csr_matrix((data, (rows, cols)), shape=(n_constrs, n_vars))
        obj_coeffs = np.array([var.Obj for var in vars])
        bounds = [(var.LB, var.UB) for var in vars]
        rhs = np.array(rhs)
        vtypes = [var.VType for var in vars]

        self.logger.debug("Extraction complete: %d constraints, %d variables.", n_constrs, n_vars)
        return A, obj_coeffs, rhs, bounds, vtypes

    def _compute_scaling_factors(self, A):
        """
        Compute row and column scaling factors from the constraint matrix A.

        The goal is to scale the rows and columns so that the matrix has unit
        row and column norms (to the extent possible).

        Args:
            A (scipy.sparse.spmatrix): The constraint matrix.

        Returns:
            tuple: (row_scaling, col_scaling, row_norms, col_norms)
        """
        A_csr = A.tocsr()
        row_norms = np.sqrt(A_csr.power(2).sum(axis=1).A.flatten())
        col_norms = np.sqrt(A_csr.power(2).sum(axis=0).A.flatten())

        self.logger.debug("Row norms: %s", row_norms)
        self.logger.debug("Column norms: %s", col_norms)

        # Prevent division by zero by replacing zero norms with 1.
        row_norms[row_norms == 0] = 1.0
        col_norms[col_norms == 0] = 1.0

        row_scaling = 1.0 / row_norms
        col_scaling = 1.0 / col_norms

        self.logger.debug("Row scaling factors: %s", row_scaling)
        self.logger.debug("Column scaling factors: %s", col_scaling)

        return row_scaling, col_scaling, row_norms, col_norms

    def _normalize_matrix(self, A, row_scaling, col_scaling):
        """
        Apply the row and column scaling to matrix A.

        Args:
            A (scipy.sparse.spmatrix): The original constraint matrix.
            row_scaling (np.ndarray): Scaling factors for rows.
            col_scaling (np.ndarray): Scaling factors for columns.

        Returns:
            scipy.sparse.csr_matrix: The normalized constraint matrix.
        """
        D_row = sp.diags(row_scaling)
        D_col = sp.diags(col_scaling)
        A_normalized = D_row @ A.tocsr() @ D_col
        return A_normalized

    def normalize(self, A, obj_coeffs, rhs, bounds):
        """
        Normalize the problem data.

        This method computes scaling factors, applies them to the constraint matrix,
        and adjusts the objective coefficients, RHS, and variable bounds accordingly.

        Args:
            A (scipy.sparse.spmatrix): Constraint matrix.
            obj_coeffs (np.ndarray): Objective coefficients.
            rhs (np.ndarray): Right-hand side vector.
            bounds (list of tuples): Variable bounds for each variable.

        Returns:
            tuple: (A_normalized, normalized_obj_coeffs, normalized_rhs, normalized_bounds, scaling_factors)
        """
        self.logger.debug("Starting normalization of problem data.")
        row_scaling, col_scaling, row_norms, col_norms = self._compute_scaling_factors(A)
        A_normalized = self._normalize_matrix(A, row_scaling, col_scaling)

        # Adjust the objective coefficients and RHS to account for scaling.
        normalized_obj_coeffs = obj_coeffs / col_scaling  # col_norms is 1 / col_scaling
        normalized_rhs = rhs * row_scaling

        # Correct bounds handling
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

        self.logger.debug("Normalization complete.")
        return A_normalized, normalized_obj_coeffs, normalized_rhs, normalized_bounds, scaling_factors

    def _build_model_from_data(self, A_normalized, obj_coeffs, rhs, bounds, vtypes):
        """
        Build a new Gurobi model from normalized problem data.

        Args:
            A_normalized (scipy.sparse.spmatrix): Normalized constraint matrix.
            obj_coeffs (np.ndarray): Normalized objective coefficients.
            rhs (np.ndarray): Normalized RHS.
            bounds (list of tuples): Normalized variable bounds.

        Returns:
            gp.Model: The new, normalized Gurobi model.
        """
        self.logger.debug("Building new Gurobi model from normalized data.")
        new_model = gp.Model("normalized_model")
        n_vars = len(obj_coeffs)
        var_list = []

        # Add variables with normalized bounds and original types.
        for i in range(n_vars):
            lb, ub = bounds[i]
            vtype = vtypes[i]
            var = new_model.addVar(lb=lb, ub=ub, vtype=vtype, name=f"x{i+1}")
            var_list.append(var)
        new_model.update()

        # Add constraints.
        m, n = A_normalized.shape
        for i in range(m):
            row_start = A_normalized.indptr[i]
            row_end = A_normalized.indptr[i+1]
            indices = A_normalized.indices[row_start:row_end]
            values = A_normalized.data[row_start:row_end]
            expr = gp.LinExpr()
            for j, coeff in zip(indices, values):
                expr.addTerms(coeff, var_list[j])
            # For illustration, we assume constraints are equalities.
            new_model.addConstr(expr == rhs[i], name=f"c{i+1}")

        # Set the normalized objective.
        obj_expr = gp.LinExpr()
        for j, coeff in enumerate(obj_coeffs):
            obj_expr.addTerms(coeff, var_list[j])
        new_model.setObjective(obj_expr, GRB.MINIMIZE)
        new_model.update()

        self.logger.debug("New Gurobi model built successfully.")
        return new_model

    def normalize_model(self, model):
        """
        Normalize an input Gurobi model and return a new normalized Gurobi model.

        This method extracts the problem data from the given model, applies normalization,
        and then builds a new model using the normalized data.

        Args:
            model (gp.Model): The original Gurobi model.

        Returns:
            gp.Model: A new Gurobi model with normalized data.
        """
        self.logger.debug("Starting normalization of the Gurobi model.")
        # Extract the problem data.
        A, obj_coeffs, rhs, bounds, vtypes = self._extract_problem_data(model)

        # Normalize the data.
        A_norm, obj_coeffs_norm, rhs_norm, bounds_norm, scaling_factors = self.normalize(
            A, obj_coeffs, rhs, bounds
        )

        # Build a new model from the normalized data.
        new_model = self._build_model_from_data(A_norm, obj_coeffs_norm, rhs_norm, bounds_norm, vtypes)

        # Optionally, store the scaling factors in the new model for reference.
        new_model._scaling_factors = scaling_factors
        self.logger.debug("Gurobi model normalization complete.")

        return new_model
