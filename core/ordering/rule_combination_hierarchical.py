import numpy as np
from gurobipy import Var
from core.ordering.ordering_rule_interface import OrderingRule

class HierarchicalRuleComposition(OrderingRule):
    def __init__(self, var_block_rules, var_intra_rules, constr_block_rules, constr_intra_rules):
        self.var_block_rules = var_block_rules or []
        self.var_intra_rules = var_intra_rules or []
        self.constr_block_rules = constr_block_rules or []
        self.constr_intra_rules = constr_intra_rules or []

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # 1. Compute block identifiers from block rules, and flatten them to ensure 1D.
        block_components = [
            np.ravel(rule.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
            for rule in self.var_block_rules
        ]
        
        # 2. Compute intra-block scores (assuming these already return 1D arrays).
        intra_scores = np.zeros(len(vars))
        for rule in self.var_intra_rules:
            intra_scores += np.ravel(rule.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # 3. Combine into tuples: (block1, block2, ..., intra_score)
        var_scores = np.empty(len(vars), dtype=object)
        for i in range(len(vars)):
            block_part = tuple(comp[i] for comp in block_components)
            var_scores[i] = block_part + (intra_scores[i],)
        return var_scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # 1. Compute block identifiers from block rules for constraints, flattening each to 1D.
        block_components = [
            np.ravel(rule.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
            for rule in self.constr_block_rules
        ]
        
        # 2. Compute intra-block scores (assumed to be 1D).
        intra_scores = np.zeros(len(constraints))
        for rule in self.constr_intra_rules:
            intra_scores += np.ravel(rule.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # 3. Combine into tuples for each constraint and store in a NumPy array.
        constr_scores = np.empty(len(constraints), dtype=object)
        for i in range(len(constraints)):
            block_part = tuple(comp[i] for comp in block_components)
            constr_scores[i] = block_part + (intra_scores[i],)
        
        print("ok2")
        return constr_scores

