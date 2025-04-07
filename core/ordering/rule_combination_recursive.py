import numpy as np
from collections import defaultdict
from utils.logging_handler import LoggingHandler
from core.ordering.ordering_rule_interface import OrderingRule
from core.ordering.recursive.ladder_intra_rule import LadderIntraRule

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

    def _compute_ordering(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
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
                A_csc=A_csc,
                A_csr=A_csr,
                constraints=constraints,
                rhs=rhs,
                parent_rules=self.matrix_block_rules_parent,
                child_rules=self.matrix_block_rules_child,
                intra_rules=self.matrix_intra_rules
            )
            logger.lazy_debug("Computed ordering: variables: %s, constraints: %s",
                         self.cached_var_order, self.cached_constr_order)

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a score for each variable equal to its position in the computed ordering.
        If the cached ordering is partial, only those indices are updated.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        n = len(vars)
        scores = np.zeros(n, dtype=int)
        cached_order = np.array(self.cached_var_order, dtype=int)
        # Assign positions to only the indices in cached_order.
        scores[cached_order] = np.arange(cached_order.size)
        logger.lazy_debug("Final variable ordering (positions): %s", scores)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a score for each constraint equal to its position in the computed ordering.
        If the cached ordering is partial, only those indices are updated.
        """
        self._compute_ordering(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        n = len(constraints)
        scores = np.zeros(n, dtype=int)
        cached_order = np.array(self.cached_constr_order, dtype=int)
        scores[cached_order] = np.arange(cached_order.size)
        logger.lazy_debug("Final constraint ordering (positions): %s", scores)
        self.reset_cache()
        return scores

    def _recursive_block_matrix(self, level, var_indices, constr_indices,
                            vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs,
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
        
        (var_indices, constr_indices, vars_sub, obj_coeffs_sub, bounds_sub,
            constr_sub, rhs_sub, sub_csr, sub_csc) = self.extract_block_data(
                vars, obj_coeffs, bounds, constraints, rhs, A_csr, var_indices, constr_indices)

        # Stop if max depth reached.
        if level >= self.max_depth:
            logger.lazy_debug("Level %d: Maximum depth reached; applying intra rules.", level)
            return self._apply_intra_rules_matrix(var_indices, constr_indices,
                                                vars_sub, obj_coeffs_sub, bounds_sub, sub_csr, sub_csc, sub_csr, constr_sub, rhs_sub,
                                                intra_rules)
        # First try parent's rules.
        effective_partition = None
        used_rule = None
        for rule in parent_rules:
            partition_map = self._partition_by_rule_matrix(rule, var_indices, constr_indices,
                                                        vars_sub, obj_coeffs_sub, bounds_sub, sub_csr, sub_csc, sub_csr, constr_sub, rhs_sub)
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
                    vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs,
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
                                                        vars_sub, obj_coeffs_sub, bounds_sub, sub_csr, sub_csc, sub_csr, constr_sub, rhs_sub)
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
                    vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs,
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
                                                vars_sub, obj_coeffs_sub, bounds_sub, sub_csr, sub_csc, sub_csr, constr_sub, rhs_sub,
                                                intra_rules)


    def _partition_by_rule_matrix(self, rule, var_indices, constr_indices,
                                  vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Uses the given block rule's score_matrix method to partition the current block.
        """
        partition_map = rule.score_matrix(var_indices, constr_indices, vars, obj_coeffs, bounds, A_csr, A_csc, A_csr, constraints, rhs)
        logger.lazy_debug("Partition by rule %s: partition_map: %s", rule.__class__.__name__, partition_map)
        return partition_map

    def _apply_intra_rules_matrix(self, global_var_indices, global_constr_indices,
                              vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub, intra_rules):
        """
        Applies intra rules on a sub-block that has already been extracted, using a hierarchical composition
        of intra rules similar to HierarchicalRuleComposition. For each variable and constraint in the sub-block,
        it computes a combined score as a tuple (by concatenating the scores from each intra rule).
        The local indices are then sorted lexicographically by these score tuples, and the resulting ordering is
        mapped back to the corresponding global indices.

        Parameters:
        - global_var_indices, global_constr_indices: Global indices corresponding to the sub-block.
        - vars_sub, bounds_sub, constr_sub, rhs_sub, etc.: The sub-block arrays.
        - intra_rules: A list of intra rules to apply.

        Returns:
        ordered_vars, ordered_constr -- the final global ordering of indices.
        """
        # Create local indices for the sub-block.
        num_vars = len(vars_sub)
        num_constr = len(constr_sub)
        local_var_indices = np.arange(num_vars)
        local_constr_indices = np.arange(num_constr)

        # Record granularity data.
        block_size = num_vars * num_constr
        self.granularity_data.append(block_size)
        logger.lazy_debug(f"Leaf block size: {block_size} (local vars: {num_vars}, local constrs: {num_constr})")


        # If no intra rules are provided, return identity ordering.
        if not intra_rules:
            logger.lazy_debug("No intra rules provided; returning identity ordering for this block.")
            return global_var_indices.tolist(), global_constr_indices.tolist()

        # --- Process variable scores on the local sub-block ---
        var_scores = []
        for idx in local_var_indices:
            flat_scores = []
            for rule in intra_rules:
                if isinstance(rule, LadderIntraRule):
                    score = rule.score_matrix_for_variable(
                        idx, vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub,
                        original_var_count=self.original_var_count, original_constr_count=self.original_constr_count
                    )
                else:
                    score = rule.score_matrix_for_variable(
                        idx, vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub
                    )
                # Flatten the score if it is a tuple.
                if isinstance(score, tuple):
                    flat_scores.extend(score)
                else:
                    flat_scores.append(score)
            var_scores.append((idx, tuple(flat_scores)))
        if len(var_scores) == 0:
            ordered_vars = global_var_indices.tolist()
        else:
            # Build a 2D array for lexicographic sorting.
            score_len = len(var_scores[0][1])
            score_array = np.empty((len(var_scores), score_len), dtype=float)
            for i, (l_idx, score_tuple) in enumerate(var_scores):
                score_array[i, :] = np.array(score_tuple, dtype=float)
            if score_array.size == 0:
                ordered_local_vars = local_var_indices
            else:
                order_vars_local = np.lexsort(score_array.T)
                ordered_local_vars = local_var_indices[order_vars_local]
            # Map local variable ordering back to global indices.
            ordered_vars = global_var_indices[ordered_local_vars]

        # --- Process constraint scores on the local sub-block ---
        constr_scores = []
        for idx in local_constr_indices:
            flat_scores = []
            for rule in intra_rules:
                if isinstance(rule, LadderIntraRule):
                    score = rule.score_matrix_for_constraint(
                        idx, vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub,
                        original_var_count=self.original_var_count, original_constr_count=self.original_constr_count
                    )
                else:
                    score = rule.score_matrix_for_constraint(
                        idx, vars_sub, obj_coeffs, bounds_sub, A, A_csc, A_csr, constr_sub, rhs_sub
                    )
                if isinstance(score, tuple):
                    flat_scores.extend(score)
                else:
                    flat_scores.append(score)
            constr_scores.append((idx, tuple(flat_scores)))
        if len(constr_scores) == 0:
            ordered_constr = global_constr_indices.tolist()
        else:
            score_len = len(constr_scores[0][1])
            constr_score_array = np.empty((len(constr_scores), score_len), dtype=float)
            for i, (l_idx, score_tuple) in enumerate(constr_scores):
                constr_score_array[i, :] = np.array(score_tuple, dtype=float)
            if constr_score_array.size == 0:
                ordered_local_constr = local_constr_indices
            else:
                order_constr_local = np.lexsort(constr_score_array.T)
                ordered_local_constr = local_constr_indices[order_constr_local]
            # Map local constraint ordering back to global indices.
            ordered_constr = global_constr_indices[ordered_local_constr]

        logger.lazy_debug("Final intra ordering: global vars: %s, global constr: %s", ordered_vars, ordered_constr)
        return ordered_vars, ordered_constr


    def extract_block_data(self, vars, obj_coeffs, bounds, constraints, rhs, A_csr, var_indices, constr_indices):
        """
        Prepares block data by converting indices and slicing arrays and submatrix.
        
        Returns:
        - var_indices: NumPy array of variable indices.
        - constr_indices: NumPy array of constraint indices.
        - vars_sub: Sliced array of variables.
        - bounds_sub: Sliced array of bounds.
        - constr_sub: Sliced array of constraints.
        - rhs_sub: Sliced array of right-hand sides (or None).
        - submatrix_csr: Submatrix in CSR format.
        - submatrix_csc: Submatrix in CSC format (converted once).
        """
        # Convert indices to NumPy arrays.
        var_indices = np.asarray(var_indices)
        constr_indices = np.asarray(constr_indices)
        
        # Slice the arrays.
        vars_sub = np.asarray(vars)[var_indices]
        bounds_sub = np.asarray(bounds)[var_indices]
        constr_sub = np.asarray(constraints)[constr_indices]
        rhs_sub = np.asarray(rhs)[constr_indices] if rhs is not None else None
        obj_coeffs_sub = np.asarray(obj_coeffs)[var_indices]
        
        # Extract the submatrix from A in CSR format and convert to CSC once.
        submatrix_csr = A_csr[constr_indices, :][:, var_indices]
        submatrix_csc = submatrix_csr.tocsc()
        
        return (var_indices, constr_indices, vars_sub, obj_coeffs_sub, bounds_sub,
                constr_sub, rhs_sub, submatrix_csr, submatrix_csc)

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