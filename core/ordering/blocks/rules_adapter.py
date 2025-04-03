from core.ordering.blocks.block_rules import BlockOrderingRule
import numpy as np

class OrderingRuleBlockAdapter(BlockOrderingRule):
    """
    Generic adapter that transforms any OrderingRule into a BlockOrderingRule.
    
    This adapter allows any existing ordering rule to be used for block ordering
    in the RecursiveHierarchicalRuleComposition framework without modification.
    
    It works by extracting scores for variables and/or constraints within each block
    and aggregating them according to the specified method.
    """
    
    def __init__(self, ordering_rule, aggregation='sum', component='both', descending=True):
        """
        Initialize the adapter with an ordering rule.
        
        :param ordering_rule: Any OrderingRule implementation
        :param aggregation: How to aggregate scores within a block - 'average', 'max', 'sum', or 'median'
        :param component: Which component to score - 'variables', 'constraints', or 'both'
        :param descending: If True, higher scores come first; if False, lower scores come first
        """
        self.ordering_rule = ordering_rule
        self.aggregation = aggregation
        self.component = component
        self.descending = descending
        
        # Cache for problem data and pre-computed scores
        self.problem_data = None
        self.var_scores = None
        self.constr_scores = None
    
    def set_problem_data(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Set the problem data and pre-compute scores.
        
        :param vars: List of variables
        :param obj_coeffs: List of objective coefficients
        :param bounds: List of bounds
        :param A: Constraint matrix
        :param A_csc: Constraint matrix in CSC format
        :param A_csr: Constraint matrix in CSR format
        :param constraints: List of constraints
        :param rhs: List of right-hand sides
        """
        self.problem_data = {
            'vars': vars,
            'obj_coeffs': obj_coeffs,
            'bounds': bounds,
            'A': A,
            'A_csc': A_csc,
            'A_csr': A_csr,
            'constraints': constraints,
            'rhs': rhs
        }
        
        # Pre-compute scores
        self._compute_scores()
    
    def _compute_scores(self):
        """
        Compute and cache the scores from the underlying ordering rule.
        """
        if not self.problem_data:
            return
            
        if self.component in ['variables', 'both']:
            self.var_scores = self.ordering_rule.score_variables(
                self.problem_data['vars'],
                self.problem_data['obj_coeffs'],
                self.problem_data['bounds'],
                self.problem_data['A'],
                self.problem_data['A_csc'],
                self.problem_data['A_csr'],
                self.problem_data['constraints'],
                self.problem_data['rhs']
            )
        
        if self.component in ['constraints', 'both']:
            self.constr_scores = self.ordering_rule.score_constraints(
                self.problem_data['vars'],
                self.problem_data['obj_coeffs'],
                self.problem_data['bounds'],
                self.problem_data['A'],
                self.problem_data['A_csc'],
                self.problem_data['A_csr'],
                self.problem_data['constraints'],
                self.problem_data['rhs']
            )
    
    def _aggregate_scores(self, scores):
        """
        Aggregate a list of scores based on the specified aggregation method.
        
        :param scores: List of scores
        :return: Aggregated score
        """
        if len(scores) == 0:
            return 0
            
        if self.aggregation == 'average':
            return np.mean(scores)
        elif self.aggregation == 'max':
            return np.max(scores)
        elif self.aggregation == 'sum':
            return np.sum(scores)
        elif self.aggregation == 'median':
            return np.median(scores)
        else:
            return np.mean(scores)  # Default to average
    
    def score_blocks(self, partition_map, level, is_parent_rule):
        """
        Score blocks based on the underlying ordering rule.
        
        :param partition_map: Dictionary {label: (var_indices, constr_indices)}
        :param level: Current recursion depth
        :param is_parent_rule: True if partition was created by a parent rule
        :return: Dictionary {label: score} with scores for each block
        """
        scores = {}
        
        # If we don't have scores yet, we can't score properly
        if (self.component in ['variables', 'both'] and self.var_scores is None) or \
           (self.component in ['constraints', 'both'] and self.constr_scores is None):
            if not self.problem_data:
                # If we don't have problem data either, return default scores
                return {label: 0 for label in partition_map}
            else:
                # Try to compute scores
                self._compute_scores()
                
                # If still no scores, return default
                if (self.component in ['variables', 'both'] and self.var_scores is None) or \
                   (self.component in ['constraints', 'both'] and self.constr_scores is None):
                    return {label: 0 for label in partition_map}
        
        for label, (var_indices, constr_indices) in partition_map.items():
            block_score = 0
            
            if self.component in ['variables', 'both'] and len(var_indices) > 0:
                var_block_scores = self.var_scores[var_indices]
                var_score = self._aggregate_scores(var_block_scores)
                block_score += var_score
            
            if self.component in ['constraints', 'both'] and len(constr_indices) > 0:
                constr_block_scores = self.constr_scores[constr_indices]
                constr_score = self._aggregate_scores(constr_block_scores)
                if self.component == 'both':
                    block_score = (block_score + constr_score) / 2
                else:
                    block_score = constr_score
            
            # Invert score if we want ascending order
            if not self.descending:
                block_score = -block_score
                
            scores[label] = block_score
        
        return scores