import numpy as np
from collections import defaultdict
from utils.logging_handler import LoggingHandler
from core.ordering.ordering_rule_interface import OrderingRule

logger = LoggingHandler().get_logger()

class RecursiveHierarchicalRuleComposition(OrderingRule):
    """
    Recursively applies rectangular block rules to partition the entire (row, column) space,
    prioritizing larger sub-blocks, until no further partitioning occurs.
    
    The computed order is cached to avoid redundant recomputation when calling
    `score_variables` and `score_constraints`.
    """
    
    def __init__(self, matrix_block_rules=None, matrix_intra_rules=None, max_depth=10):
        """
        :param matrix_block_rules: List of block rules for rectangular blocks.
        :param matrix_intra_rules: List of intra rules to order indices when no block partitioning occurs.
        :param max_depth: Maximum recursion depth (to prevent infinite recursion).
        """
        self.matrix_block_rules = matrix_block_rules or []
        self.matrix_intra_rules = matrix_intra_rules or []
        self.max_depth = max_depth
        
        # Caching the computed order
        self.cached_var_order = None
        self.cached_constr_order = None

    def reset_cache(self):
        self.cached_var_order = None
        self.cached_constr_order = None


    def _compute_ordering(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes the ordering for both variables and constraints once and stores the results.
        """
        logger.debug("Computing ordering for entire problem.")
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
                block_rules=self.matrix_block_rules,
                intra_rules=self.matrix_intra_rules
            )
            logger.debug("Computed ordering: vars: %s, constraints: %s", 
                          self.cached_var_order, self.cached_constr_order)

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a scoring based on the computed variable ordering.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, constraints, rhs)
        scores = [0] * len(vars)
        for pos, var_idx in enumerate(self.cached_var_order):
            scores[var_idx] = pos  # Assign position as score
        logger.debug("Final variable ordering (as positions): %s", scores)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a scoring based on the computed constraint ordering.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, constraints, rhs)
        scores = [0] * len(constraints)
        for pos, constr_idx in enumerate(self.cached_constr_order):
            scores[constr_idx] = pos  # Assign position as score
        logger.debug("Final constraint ordering (as positions): %s", scores)
        return scores

    def _recursive_block_matrix(self, level, var_indices, constr_indices,
                                vars, obj_coeffs, bounds, A, constraints, rhs,
                                block_rules, intra_rules):
        """
        Recursively partitions the matrix block defined by (var_indices, constr_indices)
        using the available block rules.
        """
        logger.debug("Recursion level %d: Processing var_indices: %s, constr_indices: %s",
                      level, var_indices, constr_indices)
        if level >= self.max_depth or not block_rules:
            logger.debug("Level %d: Reached max depth or no block rules left. Applying intra rules.", level)
            return self._apply_intra_rules_matrix(var_indices, constr_indices,
                                                  vars, obj_coeffs, bounds, A, constraints, rhs,
                                                  intra_rules)
        current_rule = block_rules[0]
        partition_map = self._partition_by_rule_matrix(
            current_rule, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs
        )
        logger.debug("Level %d: Using rule %s produced partition_map with %d blocks.",
                      level, current_rule.__class__.__name__, len(partition_map))

        if len(partition_map) == 1:
            logger.debug("Level %d: Partition map has one block; discarding current rule and recursing with remaining rules.", level)
            return self._recursive_block_matrix(
                level + 1, var_indices, constr_indices,
                vars, obj_coeffs, bounds, A, constraints, rhs,
                block_rules[1:], intra_rules
            )

        # Sort blocks by (size of var group + size of constr group), descending.
        sorted_blocks = sorted(partition_map.items(),
                               key=lambda x: (len(x[1][0]) + len(x[1][1])),
                               reverse=True)
        logger.debug("Level %d: Sorted blocks: %s", level,
                      [(label, (len(sub_var_indices), len(sub_constr_indices)))
                       for label, (sub_var_indices, sub_constr_indices) in sorted_blocks])
        
        ordered_vars = []
        ordered_constr = []
        for label, (sub_var_indices, sub_constr_indices) in sorted_blocks:
            logger.debug("Level %d: Recursing on block label %s with var_indices: %s and constr_indices: %s",
                          level, label, sub_var_indices, sub_constr_indices)
            sub_ordered_vars, sub_ordered_constr = self._recursive_block_matrix(
                level + 1,
                np.array(sub_var_indices),
                np.array(sub_constr_indices),
                vars, obj_coeffs, bounds, A, constraints, rhs,
                block_rules[1:], intra_rules
            )
            ordered_vars.extend(sub_ordered_vars)
            ordered_constr.extend(sub_ordered_constr)
        logger.debug("Level %d: Concatenated ordering for this block: vars: %s, constraints: %s",
                      level, ordered_vars, ordered_constr)
        return ordered_vars, ordered_constr

    def _partition_by_rule_matrix(self, rule, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Uses the given rectangular block rule to partition the current block.
        """
        partition_map = rule.score_matrix(var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs)
        logger.debug("Partition by rule %s: partition_map: %s", rule.__class__.__name__, partition_map)
        return partition_map

    def _apply_intra_rules_matrix(self, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, constraints, rhs, intra_rules):
        """
        When no further partitioning is possible, apply the intra rules to order the indices.
        """
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

        logger.debug("Intra rules ordering: vars: %s, constraints: %s", ordered_vars, ordered_constr)
        return ordered_vars, ordered_constr
