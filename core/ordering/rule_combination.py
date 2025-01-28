from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class RuleComposition(OrderingRule):
    def __init__(self, rules):
        self.rules = rules

    def score_variables(self, vars, obj_coeffs, bounds):
        # Combine scores from all rules (e.g., weighted sum)
        scores = np.zeros(len(vars))
        for rule in self.rules:
            scores += np.array(rule.score_variables(vars, obj_coeffs, bounds))
        return scores.tolist()

    def score_constraints(self, constraints, A, rhs):
        # Combine scores from all rules
        scores = np.zeros(len(constraints))
        for rule in self.rules:
            scores += np.array(rule.score_constraints(constraints, A, rhs))
        return scores.tolist()