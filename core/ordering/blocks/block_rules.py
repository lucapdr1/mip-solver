from abc import ABC, abstractmethod
import numpy as np

class BlockOrderingRule(ABC):
    """
    Interface for rules that define how to order blocks in a partition.
    """
    
    @abstractmethod
    def score_blocks(self, partition_map, level, is_parent_rule):
        """
        Score blocks from a partition map. Higher scores indicate blocks that should come first.
        
        :param partition_map: Dictionary {label: (var_indices, constr_indices)}
        :param level: Current recursion depth
        :param is_parent_rule: True if partition was created by a parent rule
        :return: Dictionary {label: score} with scores for each block
        """
        pass


class SizeBlockOrderingRule(BlockOrderingRule):
    """
    Orders blocks by their size (var_count + constr_count) in descending order.
    This is the default behavior in the original implementation.
    """
    
    def __init__(self, descending=True):
        """
        :param descending: If True, larger blocks come first; if False, smaller blocks come first
        """
        self.descending = descending
    
    def score_blocks(self, partition_map, level, is_parent_rule):
        scores = {}
        for label, (var_indices, constr_indices) in partition_map.items():
            # Score is the total size of variables and constraints
            score = len(var_indices) + len(constr_indices)
            if not self.descending:
                score = -score  # Invert the score for ascending order
            scores[label] = score
        return scores


class IdentityBlockOrderingRule(BlockOrderingRule):
    """
    Preserves the original order of blocks as given by their labels.
    Assumes labels can be sorted (e.g., integers or strings).
    """
    
    def score_blocks(self, partition_map, level, is_parent_rule):
        # Use the negative label as score to preserve original order
        # (assuming labels are sortable)
        scores = {}
        for label in partition_map.keys():
            try:
                # Try to use the label directly as a score
                scores[label] = -float(label)
            except (ValueError, TypeError):
                # If label is not a number, use its hash
                scores[label] = -hash(label)
        return scores


class DensityBlockOrderingRule(BlockOrderingRule):
    """
    Orders blocks by their density (nonzeros / (var_count * constr_count)) in descending order.
    Requires access to the constraint matrix to calculate the actual density.
    """
    
    def __init__(self, descending=True):
        """
        :param descending: If True, denser blocks come first; if False, sparser blocks come first
        """
        self.descending = descending
        self.matrix_cache = None
    
    def score_blocks(self, partition_map, level, is_parent_rule, A=None):
        """
        Score blocks based on their density.
        
        :param partition_map: Dictionary {label: (var_indices, constr_indices)}
        :param level: Current recursion depth
        :param is_parent_rule: True if partition was created by a parent rule
        :param A: The constraint matrix (optional, if already provided to the rule)
        :return: Dictionary {label: score} with scores for each block
        """
        scores = {}
        
        # Use provided matrix or cached matrix
        matrix = A or self.matrix_cache
        
        for label, (var_indices, constr_indices) in partition_map.items():
            var_count = len(var_indices)
            constr_count = len(constr_indices)
            block_size = var_count * constr_count
            
            if block_size == 0:
                density = 0
            else:
                # Extract the submatrix for this block
                if matrix is not None:
                    submatrix = matrix[constr_indices, :][:, var_indices]
                    nnz = submatrix.count_nonzero()
                    density = nnz / block_size
                else:
                    # If no matrix is available, use a placeholder
                    density = 1.0
                
            score = density
            if not self.descending:
                score = -score
                
            scores[label] = score
        
        return scores
    
    def set_matrix(self, A):
        """
        Set the constraint matrix for density calculations.
        
        :param A: The constraint matrix in a sparse format
        """
        self.matrix_cache = A


class HierarchicalBlockOrderingRule(BlockOrderingRule):
    """
    Combines multiple block ordering rules in a hierarchical manner.
    For blocks with the same score from the first rule, the second rule is used, and so on.
    """
    
    def __init__(self, rules):
        """
        :param rules: List of BlockOrderingRule objects to apply in order
        """
        self.rules = rules
    
    def score_blocks(self, partition_map, level, is_parent_rule):
        if not self.rules:
            return {label: 0 for label in partition_map}
            
        # Get scores from each rule
        all_scores = []
        for rule in self.rules:
            scores = rule.score_blocks(partition_map, level, is_parent_rule)
            all_scores.append(scores)
        
        # Combine scores into tuples
        combined_scores = {}
        for label in partition_map:
            score_tuple = tuple(scores[label] for scores in all_scores)
            combined_scores[label] = score_tuple
            
        return combined_scores