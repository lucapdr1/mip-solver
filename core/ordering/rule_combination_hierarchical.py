import numpy as np
from gurobipy import Var
from core.ordering.ordering_rule_interface import OrderingRule

class HierarchicalRuleComposition(OrderingRule):
    def __init__(self, var_block_rules, var_intra_rules, constr_block_rules, constr_intra_rules):
        self.var_block_rules = var_block_rules or []
        self.var_intra_rules = var_intra_rules or []
        self.constr_block_rules = constr_block_rules or []
        self.constr_intra_rules = constr_intra_rules or []

    # ==========================
    # Variable Scoring
    # ==========================
    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # 1. Compute block identifiers from block rules
        block_components = [
            rule.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
            for rule in self.var_block_rules
        ]

        # 2. Compute intra-block scores
        intra_scores = np.zeros(len(vars))
        for rule in self.var_intra_rules:
            intra_scores += rule.score_variables(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)

        # 3. Combine into tuples: (block1, block2, ..., intra_score)
        var_scores = []
        for i in range(len(vars)):
            block_part = tuple(comp[i] for comp in block_components)
            var_scores.append(block_part + (intra_scores[i],))

        return var_scores

    # ==========================
    # Constraint Scoring
    # ==========================
    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Similar logic for constraints
        block_components = [
            rule.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
            for rule in self.constr_block_rules
        ]

        intra_scores = np.zeros(len(constraints))
        for rule in self.constr_intra_rules:
            intra_scores += rule.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)

        constr_scores = []
        for i in range(len(constraints)):
            block_part = tuple(comp[i] for comp in block_components)
            constr_scores.append(block_part + (intra_scores[i],))

        return constr_scores