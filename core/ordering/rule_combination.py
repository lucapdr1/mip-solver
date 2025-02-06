from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np

class RuleComposition(OrderingRule):
    def __init__(self, rules):
        self.rules = rules

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # Validate 'vars' is a sequence
        if not isinstance(vars, (list, tuple, np.ndarray)):
            raise ValueError(f"Expected 'vars' to be a list, tuple, or array, but got {type(vars)}")
        
        # Initialize scores array
        num_vars = len(vars)
        scores = np.zeros(num_vars, dtype=float)

        # Loop through all rules to combine scores
        for rule in self.rules:
            rule_scores = rule.score_variables(vars, obj_coeffs, bounds, A, constraints, rhs)
            if not isinstance(rule_scores, (list, np.ndarray)):
                raise ValueError(f"Rule {rule} returned invalid type for scores: {type(rule_scores)}")
            
            # Ensure rule_scores is a NumPy array
            rule_scores = np.array(rule_scores, dtype=float)
            
            # Accumulate scores
            scores += rule_scores

        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # Initialize scores array
        scores = np.zeros(len(constraints))
        
        for rule in self.rules:
            # Assuming each rule's score_constraints method returns a list or array of scores
            rule_scores = rule.score_constraints(vars, obj_coeffs, bounds, A, constraints, rhs)
            
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