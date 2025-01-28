from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class RuleComposition(OrderingRule):
    def __init__(self, rules):
        self.rules = rules

    def score_variables(self, vars, obj_coeffs, A, bounds):
        # Combine scores from all rules (e.g., weighted sum)
        scores = np.zeros(len(vars))
        for rule in self.rules:
            scores += np.array(rule.score_variables(vars, obj_coeffs, A, bounds))
        return scores.tolist()

    def score_constraints(self, constraints, A, rhs):
        # Initialize scores array
        scores = np.zeros(len(constraints))
        
        for rule in self.rules:
            # Assuming each rule's score_constraints method returns a list or array of scores
            rule_scores = rule.score_constraints(constraints, A, rhs)
            
            # Convert rule_scores to numpy array if it's not already
            if not isinstance(rule_scores, np.ndarray):
                rule_scores = np.array(rule_scores)
            
            # Check if shapes match or need adjustment
            if rule_scores.shape != scores.shape:
                # If shapes don't match, you might need to reshape or handle this case specifically
                raise ValueError(f"Shape mismatch: rule_scores {rule_scores.shape} vs expected {scores.shape}")
            
            # Add rule's scores to total scores
            scores += rule_scores

        return scores.tolist()