# core/problem_permutator.py

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from utils.logging_handler import LoggingHandler

class ProblemPermutator:
    def __init__(self, gp_env, original_model):
        self.gp_env = gp_env
        self.logger = LoggingHandler().get_logger()
        self.original_model = original_model

    def create_permuted_problem(self):
        """Create permuted problem with structural validation"""
        # Create a new model instead of copying to avoid reference issues
        permuted_model = gp.Model(env=self.gp_env)
        original_vars = self.original_model.getVars()
        num_vars = len(original_vars)

        # Generate permutation
        permutation = np.random.permutation(num_vars)
        
        # Create new variables with permuted properties
        var_map = {}
        for i, orig_idx in enumerate(permutation):
            orig_var = original_vars[orig_idx]
            new_var = permuted_model.addVar(
                lb=orig_var.LB,
                ub=orig_var.UB,
                obj=orig_var.Obj,
                vtype=orig_var.VType,
                name=f"x{i}"
            )
            var_map[orig_idx] = new_var

        # Get constraint matrix
        A = self.original_model.getA()
        constrs = self.original_model.getConstrs()

        # Rebuild constraints with permuted variables
        for i, constr in enumerate(constrs):
            expr = gp.LinExpr()
            row = A.getrow(i)
            
            # Iterate through non-zero elements
            for j, val in zip(row.indices, row.data):
                expr.add(var_map[j], float(val))
            
            # Correct syntax for addConstr
            if constr.Sense == GRB.LESS_EQUAL:
                permuted_model.addConstr(expr <= constr.RHS)
            elif constr.Sense == GRB.GREATER_EQUAL:
                permuted_model.addConstr(expr >= constr.RHS)
            else:  # GRB.EQUAL
                permuted_model.addConstr(expr == constr.RHS)

        # Set objective sense
        permuted_model.ModelSense = self.original_model.ModelSense
        
        permuted_model.update()
        return permuted_model


