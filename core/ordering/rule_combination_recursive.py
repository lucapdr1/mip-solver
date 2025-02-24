import numpy as np
from collections import defaultdict
from utils.logging_handler import LoggingHandler
from core.ordering.ordering_rule_interface import OrderingRule

logger = LoggingHandler().get_logger()

def rotate(lst):
    """Rotate a list by one element: [a, b, c] -> [b, c, a]."""
    return lst
    #return lst[1:] + lst[:1] if lst else lst

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
        self.original_var_count = 0
        self.original_constr_count = 0

    def reset_cache(self):
        self.cached_var_order = None
        self.cached_constr_order = None

    def _compute_ordering(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes and caches the ordering for variables and constraints.
        """
        self.original_var_count = len(vars)
        self.original_constr_count = len(constraints)
        logger.lazy_debug("Computing ordering for the entire problem.")
        if self.cached_var_order is None or self.cached_constr_order is None: 
            self.granularity_data = []
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
            logger.lazy_debug("Computed ordering: variables: %s, constraints: %s",
                         self.cached_var_order, self.cached_constr_order)

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a score for each variable equal to its position in the computed ordering.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, constraints, rhs)
        scores = [0] * len(vars)
        for pos, var_idx in enumerate(self.cached_var_order):
            scores[var_idx] = pos
        logger.lazy_debug("Final variable ordering (positions): %s", scores)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a score for each constraint equal to its position in the computed ordering.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, constraints, rhs)
        scores = [0] * len(constraints)
        for pos, constr_idx in enumerate(self.cached_constr_order):
            scores[constr_idx] = pos
        logger.lazy_debug("Final constraint ordering (positions): %s", scores)
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
        logger.lazy_debug("Recursion level %d: var_indices: %s, constr_indices: %s",
                    level, var_indices, constr_indices)
        
        # Stop if max depth reached.
        if level >= self.max_depth:
            logger.lazy_debug("Level %d: Maximum depth reached; applying intra rules.", level)
            return self._apply_intra_rules_matrix(var_indices, constr_indices,
                                                vars, obj_coeffs, bounds, A, constraints, rhs,
                                                intra_rules)
        
        # First try parent's rules.
        effective_partition = None
        used_rule = None
        for rule in parent_rules:
            partition_map = self._partition_by_rule_matrix(rule, var_indices, constr_indices,
                                                        vars, obj_coeffs, bounds, A, constraints, rhs)
            logger.lazy_debug("Level %d: Parent rule %s produced partition_map with %d blocks.",
                        level, rule.__class__.__name__, len(partition_map))
            if len(partition_map) > 1:
                effective_partition = partition_map
                used_rule = rule
                break
        if effective_partition is not None:
            logger.lazy_debug("Level %d: Using parent's rule %s that produced an effective partition.", 
                        level, used_rule.__class__.__name__)
            # Convert dictionary items into a NumPy array with dtype=object.
            blocks = np.array(list(effective_partition.items()), dtype=object)
            # Compute an array of block sizes.
            sizes = np.array([len(block[1][0]) + len(block[1][1]) for block in blocks])
            # Get the indices that sort the sizes in descending order.
            sorted_indices = np.argsort(sizes)[::-1]
            # Index the blocks array to obtain a sorted array.
            sorted_blocks = blocks[sorted_indices]

            # Instead of extending Python lists, accumulate sub-results in a list of NumPy arrays.
            ordered_vars_list = []
            ordered_constr_list = []
            # For each subblock, restart parent's rule list.
            for label, (sub_var_indices, sub_constr_indices) in sorted_blocks:
                logger.lazy_debug("Level %d: Recursing on parent's partition block %s with var_indices: %s, constr_indices: %s",
                            level, label, sub_var_indices, sub_constr_indices)
                sub_ordered_vars, sub_ordered_constr = self._recursive_block_matrix(
                    level + 1,
                    np.array(sub_var_indices),
                    np.array(sub_constr_indices),
                    vars, obj_coeffs, bounds, A, constraints, rhs,
                    parent_rules, child_rules, intra_rules
                )
                # Convert sub-results to numpy arrays if they aren’t already.
                ordered_vars_list.append(np.array(sub_ordered_vars))
                ordered_constr_list.append(np.array(sub_ordered_constr))
            # Concatenate all sub-results using NumPy's optimized routines.
            if ordered_vars_list:
                ordered_vars = np.concatenate(ordered_vars_list)
                ordered_constr = np.concatenate(ordered_constr_list)
            else:
                ordered_vars = np.array([])
                ordered_constr = np.array([])
            logger.lazy_debug("Level %d: Concatenated ordering from parent's partition: vars: %s, constr: %s",
                        level, ordered_vars, ordered_constr)
            return ordered_vars, ordered_constr

        # --- (Child branch would be modified similarly) ---
        # If no parent's rule was effective, try child's rules.
        effective_partition = None
        used_rule = None
        for rule in child_rules:
            partition_map = self._partition_by_rule_matrix(rule, var_indices, constr_indices,
                                                        vars, obj_coeffs, bounds, A, constraints, rhs)
            logger.lazy_debug("Level %d: Child rule %s produced partition_map with %d blocks.",
                        level, rule.__class__.__name__, len(partition_map))
            if len(partition_map) > 1:
                effective_partition = partition_map
                used_rule = rule
                break
        if effective_partition is not None:
            logger.lazy_debug("Level %d: Using child's rule %s that produced an effective partition.", 
                        level, used_rule.__class__.__name__)
            
            # Convert dictionary items into a NumPy array with dtype=object.
            blocks = np.array(list(effective_partition.items()), dtype=object)
            # Compute an array of block sizes.
            sizes = np.array([len(block[1][0]) + len(block[1][1]) for block in blocks])
            # Get the indices that sort the sizes in descending order.
            sorted_indices = np.argsort(sizes)[::-1]
            # Index the blocks array to obtain a sorted array.
            sorted_blocks = blocks[sorted_indices]
            

            ordered_vars_list = []
            ordered_constr_list = []
            rotated_child_rules = rotate(child_rules)
            for label, (sub_var_indices, sub_constr_indices) in sorted_blocks:
                logger.lazy_debug("Level %d: Recursing on child's partition block %s with var_indices: %s, constr_indices: %s",
                            level, label, sub_var_indices, sub_constr_indices)
                sub_ordered_vars, sub_ordered_constr = self._recursive_block_matrix(
                    level + 1,
                    np.array(sub_var_indices),
                    np.array(sub_constr_indices),
                    vars, obj_coeffs, bounds, A, constraints, rhs,
                    rotated_child_rules, rotated_child_rules, intra_rules
                )
                ordered_vars_list.append(np.array(sub_ordered_vars))
                ordered_constr_list.append(np.array(sub_ordered_constr))
            if ordered_vars_list:
                ordered_vars = np.concatenate(ordered_vars_list)
                ordered_constr = np.concatenate(ordered_constr_list)
            else:
                ordered_vars = np.array([])
                ordered_constr = np.array([])
            logger.lazy_debug("Level %d: Concatenated ordering from child's partition: vars: %s, constr: %s",
                        level, ordered_vars, ordered_constr)
            return ordered_vars, ordered_constr

        # If no effective partition from parent's or child's rules, apply intra rules.
        logger.lazy_debug("Level %d: No effective partition from parent's or child's rules; applying intra rules.", level)
        return self._apply_intra_rules_matrix(var_indices, constr_indices,
                                            vars, obj_coeffs, bounds, A, constraints, rhs,
                                            intra_rules)


    def _partition_by_rule_matrix(self, rule, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Uses the given block rule's score_matrix method to partition the current block.
        """
        partition_map = rule.score_matrix(var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs)
        logger.lazy_debug("Partition by rule %s: partition_map: %s", rule.__class__.__name__, partition_map)
        return partition_map

    def _apply_intra_rules_matrix(self, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, constraints, rhs, intra_rules):
        """
        When no further block partitioning is possible, apply intra rules to order the indices.
        """
        # Append the product of current block dimensions
        block_size = len(var_indices) * len(constr_indices)
        self.granularity_data.append(block_size)
        logger.lazy_debug(f"Leaf block size: {block_size} (vars: {len(var_indices)}, constrs: {len(constr_indices)})")

        logger.lazy_debug("Applying intra rules on var_indices: %s, constr_indices: %s", var_indices, constr_indices)
        if not intra_rules:
            logger.lazy_debug("No intra rules provided; returning identity ordering for this block.")
            return list(var_indices), list(constr_indices)
    
        var_scores = []
        for idx in var_indices:
            scores = tuple(rule.score_matrix_for_variable(idx, vars, obj_coeffs, bounds, A, constraints, rhs)
                           for rule in intra_rules)
            logger.lazy_debug("Variable %s intra scores: %s", idx, scores)
            var_scores.append((idx, scores))
        # Assume var_scores is a list of tuples: (idx, score_tuple)
        # Extract indices and scores into NumPy arrays.
        indices = np.array([idx for idx, score in var_scores])
        # Convert the score tuples into a 2D array.
        score_array = np.array([score for idx, score in var_scores])
        # To sort lexicographically in the order of the tuple (first element most significant),
        # we need to reverse the order of the columns (because np.lexsort uses the last key as primary).
        order = np.lexsort(tuple(score_array.T[::-1]))
        # Now get the ordered indices as a NumPy array.
        ordered_vars = indices[order]
    
        constr_scores = []
        for i in constr_indices:
            scores = tuple(rule.score_matrix_for_constraint(i, vars, obj_coeffs, bounds, A, constraints, rhs)
                           for rule in intra_rules)
            logger.lazy_debug("Constraint %s intra scores: %s", i, scores)
            constr_scores.append((i, scores))
        # Extract indices and score tuples into NumPy arrays.
        indices = np.array([i for i, score in constr_scores])
        score_array = np.array([score for i, score in constr_scores])
        # np.lexsort sorts by the last key first, so reverse the columns
        order = np.lexsort(tuple(score_array.T[::-1]))
        # Get the sorted constraint indices as a NumPy array.
        ordered_constr = indices[order]
    
        logger.lazy_debug("Intra ordering: vars: %s, constr: %s", ordered_vars, ordered_constr)
        return ordered_vars, ordered_constr

    def get_granularity_data(self):
        """Return the sizes of all leaf blocks as a list of (var_count, constr_count) tuples."""
        return self.granularity_data.copy()

    def get_granularity_statistics(self):
        """Compute statistics on block sizes (var x constr products)."""
        if not self.granularity_data:
            return None
        
        stats = {
            'block_sizes': {
                'average': np.mean(self.granularity_data),
                'min': np.min(self.granularity_data),
                'max': np.max(self.granularity_data),
                'total_blocks': len(self.granularity_data),
                'sum_of_block_sizes': sum(self.granularity_data),
                'original_matrix_size': self.original_var_count * self.original_constr_count
            }
        }
        return stats