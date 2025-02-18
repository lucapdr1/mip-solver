import numpy as np
from collections import defaultdict
from utils.logging_handler import LoggingHandler
from core.ordering.ordering_rule_interface import OrderingRule

logger = LoggingHandler().get_logger()

def rotate(lst):
    """Rotate a list by one element: [a, b, c] -> [b, c, a]."""
    return lst[1:] + lst[:1] if lst else lst

class RecursiveHierarchicalRuleComposition(OrderingRule):
    """
    Recursively partitions the entire (row, column) space using block rules and intra rules.
    
    Two separate lists of block rules are provided:
      - matrix_block_rules_parent: Rules to apply (in order) at the current block as long as they are effective.
      - matrix_block_rules_child: If no parent's rule partitions the block, then these are tried (in order).
      
    Intra rules (matrix_intra_rules) are used to order indices when no rule (neither parent's nor child's)
    produces an effective partition.
    
    If all rules have been tried on a block and no change happens (i.e. each returns only one block),
    the recursion stops for that block.
    
    Detailed debug logs are provided to trace the recursion and rule application.
    """
    
    def __init__(self, matrix_block_rules_parent=None, matrix_block_rules_child=None,
                 matrix_intra_rules=None, max_depth=10):
        """
        :param matrix_block_rules_parent: List of block rules to apply at the current block.
        :param matrix_block_rules_child: List of block rules to try if parent's rules fail.
        :param matrix_intra_rules: List of intra rules used to order indices when no partitioning occurs.
        :param max_depth: Maximum recursion depth.
        """
        self.matrix_block_rules_parent = matrix_block_rules_parent or []
        self.matrix_block_rules_child = matrix_block_rules_child or []
        self.matrix_intra_rules = matrix_intra_rules or []
        self.max_depth = max_depth

        # Cached computed ordering (if desired, you can disable caching between different problem instances)
        self.cached_var_order = None
        self.cached_constr_order = None
        self.granularity_data = []  # To track sizes of leaf blocks

    def reset_cache(self):
        self.cached_var_order = None
        self.cached_constr_order = None

    def _compute_ordering(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes and caches the ordering for variables and constraints.
        """
        logger.debug("Computing ordering for the entire problem.")
        if self.cached_var_order is None or self.cached_constr_order is None: 
            self.cached_var_order, self.cached_constr_order = self._recursive_block_matrix(
                level=0,
                var_indices=np.arange(len(vars)),
                constr_indices=np.arange(len(constraints)),
                vars=vars,
                obj_coeffs=obj_coeffs,
                bounds=bounds,
                A=A,
                constraints=constraints,
                rhs=rhs,
                parent_rules=self.matrix_block_rules_parent,
                child_rules=self.matrix_block_rules_child,
                intra_rules=self.matrix_intra_rules
            )
            logger.debug("Computed ordering: variables: %s, constraints: %s",
                         self.cached_var_order, self.cached_constr_order)

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a score for each variable equal to its position in the computed ordering.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, constraints, rhs)
        scores = [0] * len(vars)
        for pos, var_idx in enumerate(self.cached_var_order):
            scores[var_idx] = pos
        logger.debug("Final variable ordering (positions): %s", scores)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a score for each constraint equal to its position in the computed ordering.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, constraints, rhs)
        scores = [0] * len(constraints)
        for pos, constr_idx in enumerate(self.cached_constr_order):
            scores[constr_idx] = pos
        logger.debug("Final constraint ordering (positions): %s", scores)
        #reset chaches before ending this iteration
        self.reset_cache()
        return scores

    def _recursive_block_matrix(self, level, var_indices, constr_indices,
                                vars, obj_coeffs, bounds, A, constraints, rhs,
                                parent_rules, child_rules, intra_rules):
        """
        Recursively partitions the block (var_indices, constr_indices) using the provided block rules.
        
        First, all parent's rules are tried (in order). If any parent's rule produces an effective partition 
        (i.e. more than one block), that partition is used. Otherwise, child's rules are tried.
        
        If neither set of rules partitions the block (i.e. each returns a single block), recursion stops 
        for that block (intra rules or identity ordering are applied).
        """
        logger.debug("Recursion level %d: var_indices: %s, constr_indices: %s",
                     level, var_indices, constr_indices)
        
        # Stop if max depth reached.
        if level >= self.max_depth:
            logger.debug("Level %d: Maximum depth reached; applying intra rules.", level)
            return self._apply_intra_rules_matrix(var_indices, constr_indices,
                                                  vars, obj_coeffs, bounds, A, constraints, rhs,
                                                  intra_rules)
        
        # First try parent's rules.
        effective_partition = None
        used_rule = None
        for rule in parent_rules:
            partition_map = self._partition_by_rule_matrix(rule, var_indices, constr_indices,
                                                           vars, obj_coeffs, bounds, A, constraints, rhs)
            logger.debug("Level %d: Parent rule %s produced partition_map with %d blocks.",
                         level, rule.__class__.__name__, len(partition_map))
            if len(partition_map) > 1:
                effective_partition = partition_map
                used_rule = rule
                break
        if effective_partition is not None:
            logger.debug("Level %d: Using parent's rule %s that produced an effective partition.", level, used_rule.__class__.__name__)
            sorted_blocks = sorted(effective_partition.items(),
                                   key=lambda x: (len(x[1][0]) + len(x[1][1])),
                                   reverse=True)
            ordered_vars = []
            ordered_constr = []
            # For each subblock, restart parent's rule list (i.e. parent's rules can be applied again).
            for label, (sub_var_indices, sub_constr_indices) in sorted_blocks:
                logger.debug("Level %d: Recursing on parent's partition block %s with var_indices: %s, constr_indices: %s",
                             level, label, sub_var_indices, sub_constr_indices)
                sub_ordered_vars, sub_ordered_constr = self._recursive_block_matrix(
                    level + 1,
                    np.array(sub_var_indices),
                    np.array(sub_constr_indices),
                    vars, obj_coeffs, bounds, A, constraints, rhs,
                    parent_rules, child_rules, intra_rules
                )
                ordered_vars.extend(sub_ordered_vars)
                ordered_constr.extend(sub_ordered_constr)
            logger.debug("Level %d: Concatenated ordering from parent's partition: vars: %s, constr: %s",
                         level, ordered_vars, ordered_constr)
            return ordered_vars, ordered_constr
        
        # If no parent's rule was effective, try child's rules.
        effective_partition = None
        used_rule = None
        for rule in child_rules:
            partition_map = self._partition_by_rule_matrix(rule, var_indices, constr_indices,
                                                           vars, obj_coeffs, bounds, A, constraints, rhs)
            logger.debug("Level %d: Child rule %s produced partition_map with %d blocks.",
                         level, rule.__class__.__name__, len(partition_map))
            if len(partition_map) > 1:
                effective_partition = partition_map
                used_rule = rule
                break
        if effective_partition is not None:
            logger.debug("Level %d: Using child's rule %s that produced an effective partition.", level, used_rule.__class__.__name__)
            sorted_blocks = sorted(effective_partition.items(),
                                   key=lambda x: (len(x[1][0]) + len(x[1][1])),
                                   reverse=True)
            ordered_vars = []
            ordered_constr = []
            # Rotate the child's rules so that the rule just used is not applied twice in a row.
            rotated_child_rules = rotate(child_rules)
            for label, (sub_var_indices, sub_constr_indices) in sorted_blocks:
                logger.debug("Level %d: Recursing on child's partition block %s with var_indices: %s, constr_indices: %s",
                             level, label, sub_var_indices, sub_constr_indices)
                sub_ordered_vars, sub_ordered_constr = self._recursive_block_matrix(
                    level + 1,
                    np.array(sub_var_indices),
                    np.array(sub_constr_indices),
                    vars, obj_coeffs, bounds, A, constraints, rhs,
                    rotated_child_rules, rotated_child_rules, intra_rules
                )
                ordered_vars.extend(sub_ordered_vars)
                ordered_constr.extend(sub_ordered_constr)
            logger.debug("Level %d: Concatenated ordering from child's partition: vars: %s, constr: %s",
                         level, ordered_vars, ordered_constr)
            return ordered_vars, ordered_constr
        
        # If neither parent's nor child's rules produce an effective partition, stop recursion.
        logger.debug("Level %d: No effective partition from parent's or child's rules; applying intra rules.", level)
        return self._apply_intra_rules_matrix(var_indices, constr_indices,
                                              vars, obj_coeffs, bounds, A, constraints, rhs,
                                              intra_rules)

    def _partition_by_rule_matrix(self, rule, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Uses the given block rule's score_matrix method to partition the current block.
        """
        partition_map = rule.score_matrix(var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs)
        logger.debug("Partition by rule %s: partition_map: %s", rule.__class__.__name__, partition_map)
        return partition_map

    def _apply_intra_rules_matrix(self, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, constraints, rhs, intra_rules):
        """
        When no further block partitioning is possible, apply intra rules to order the indices.
        """
        # Record the size of this leaf block
        var_count = len(var_indices)
        constr_count = len(constr_indices)
        self.granularity_data.append((var_count, constr_count))
        logger.debug("Leaf block size: %d variables, %d constraints", var_count, constr_count)

        logger.debug("Applying intra rules on var_indices: %s, constr_indices: %s", var_indices, constr_indices)
        if not intra_rules:
            logger.debug("No intra rules provided; returning identity ordering for this block.")
            return list(var_indices), list(constr_indices)
    
        var_scores = []
        for idx in var_indices:
            scores = tuple(rule.score_matrix_for_variable(idx, vars, obj_coeffs, bounds, A, constraints, rhs)
                           for rule in intra_rules)
            logger.debug("Variable %s intra scores: %s", idx, scores)
            var_scores.append((idx, scores))
        ordered_vars = [idx for idx, _ in sorted(var_scores, key=lambda x: x[1])]
    
        constr_scores = []
        for i in constr_indices:
            scores = tuple(rule.score_matrix_for_constraint(i, vars, obj_coeffs, bounds, A, constraints, rhs)
                           for rule in intra_rules)
            logger.debug("Constraint %s intra scores: %s", i, scores)
            constr_scores.append((i, scores))
        ordered_constr = [i for i, _ in sorted(constr_scores, key=lambda x: x[1])]
    
        logger.debug("Intra ordering: vars: %s, constr: %s", ordered_vars, ordered_constr)
        return ordered_vars, ordered_constr

    def get_granularity_data(self):
        """Return the sizes of all leaf blocks as a list of (var_count, constr_count) tuples."""
        return self.granularity_data.copy()

    def get_granularity_statistics(self):
        """Compute and return statistics on the granularity of leaf blocks."""
        if not self.granularity_data:
            return None
        
        var_sizes = [v for v, c in self.granularity_data]
        constr_sizes = [c for v, c in self.granularity_data]
        
        stats = {
            'variables': {
                'average': np.mean(var_sizes),
                'min': np.min(var_sizes),
                'max': np.max(var_sizes),
                'total_blocks': len(var_sizes),
                'total_vars': sum(var_sizes)
            },
            'constraints': {
                'average': np.mean(constr_sizes),
                'min': np.min(constr_sizes),
                'max': np.max(constr_sizes),
                'total_blocks': len(constr_sizes),
                'total_constrs': sum(constr_sizes)
            }
        }
        return stats