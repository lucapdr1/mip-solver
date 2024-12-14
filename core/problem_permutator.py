# core/problem_permutator.py

import gurobipy as gp
import numpy as np
from core.logging_handler import LoggingHandler

class ProblemPermutator:
    def __init__(self, file_path):
        """
        Initialize the ProblemPermutator.

        Args:
            file_path (str): Path to the LP or MPS file.
        """
        self.file_path = file_path
        self.logger = LoggingHandler().get_logger()

    def create_permuted_problem(self):
        """
        Create a randomly permuted version of the original problem.

        Returns:
            gurobipy.Model: Permuted Gurobi model
        """
        # Load the original problem
        original_model = gp.read(self.file_path)

        # Get variable list
        variables = original_model.getVars()
        num_vars = len(variables)

        # Create a random permutation of variable indices
        permutation = np.random.permutation(num_vars)
        self.logger.info("Creating permuted problem:")
        self.logger.info(f"Permutation: {permutation}")

        # Extract data from the original model
        obj_coeffs = [v.obj for v in variables]
        lower_bounds = [v.lb for v in variables]
        upper_bounds = [v.ub for v in variables]
        var_types = [v.vtype for v in variables]

        # Apply permutation to objective, bounds, and types
        permuted_obj = [obj_coeffs[i] for i in permutation]
        permuted_lb = [lower_bounds[i] for i in permutation]
        permuted_ub = [upper_bounds[i] for i in permutation]
        permuted_types = [var_types[i] for i in permutation]

        # Create a new model
        permuted_model = gp.Model()

        # Add permuted variables to the new model
        permuted_variables = permuted_model.addVars(
            num_vars,
            obj=permuted_obj,
            lb=permuted_lb,
            ub=permuted_ub,
            vtype=permuted_types,
            name="x"
        )
        permuted_variables_list = list(permuted_variables.values())

        # Map original variables to permuted variables
        var_map = {variables[i]: permuted_variables_list[permutation.tolist().index(i)] for i in range(num_vars)}

        # Add constraints from the original model
        for constraint in original_model.getConstrs():
            expr = original_model.getRow(constraint)  # Linear expression for the constraint
            coeffs = [expr.getCoeff(i) for i in range(expr.size())]
            vars_ = [expr.getVar(i) for i in range(expr.size())]

            # Map variables and coefficients to the permuted variables
            permuted_expr = gp.LinExpr(
                [coeffs[i] for i in range(len(coeffs))],
                [var_map[vars_[i]] for i in range(len(vars_))]
            )

            # Add the permuted constraint based on its sense
            if constraint.sense == gp.GRB.LESS_EQUAL:
                permuted_model.addConstr(permuted_expr <= constraint.rhs, name=constraint.ConstrName)
            elif constraint.sense == gp.GRB.GREATER_EQUAL:
                permuted_model.addConstr(permuted_expr >= constraint.rhs, name=constraint.ConstrName)
            elif constraint.sense == gp.GRB.EQUAL:
                permuted_model.addConstr(permuted_expr == constraint.rhs, name=constraint.ConstrName)

        self.logger.info("Permuted problem created successfully.")
        return permuted_model


