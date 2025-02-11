import numpy as np
from collections import defaultdict
from core.ordering.ordering_rule_interface import OrderingRule

class RecursiveHierarchicalRuleComposition(OrderingRule):
    """
    Recursively applies block rules until no further partitioning occurs.
    Then applies intra-block rules for ranking.
    """

    def __init__(self, var_block_rules=None, var_intra_rules=None,
                       constr_block_rules=None, constr_intra_rules=None,
                       max_depth=10):
        self.var_block_rules = var_block_rules or []
        self.var_intra_rules = var_intra_rules or []
        self.constr_block_rules = constr_block_rules or []
        self.constr_intra_rules = constr_intra_rules or []
        self.max_depth = max_depth  # Prevents infinite recursion

    # ----------------------------------------------------------------
    # Variable Scoring
    # ----------------------------------------------------------------
    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        all_var_indices = np.arange(len(vars))  # Use NumPy array for efficient indexing
        results = self._recursive_block_vars(
            level=0,
            var_indices=all_var_indices,
            vars=vars,
            obj_coeffs=obj_coeffs,
            bounds=bounds,
            A=A,
            constraints=constraints,
            rhs=rhs,
            block_rules=self.var_block_rules,
            intra_rules=self.var_intra_rules
        )

        results.sort(key=lambda x: x[0])
        return [score for (_, score) in results]

    def _recursive_block_vars(self, level, var_indices,
                              vars, obj_coeffs, bounds, A, constraints, rhs,
                              block_rules, intra_rules):
        if level >= self.max_depth or not block_rules:
            return self._apply_intra_rules_vars(var_indices, vars, obj_coeffs,
                                                bounds, A, constraints, rhs, intra_rules)

        current_block_rule = block_rules[0]
        partition_map = self._partition_by_rule_vars(
            current_block_rule, var_indices, vars, obj_coeffs, bounds, A, constraints, rhs
        )

        if len(partition_map) == 1:
            return self._recursive_block_vars(
                level+1, var_indices, vars, obj_coeffs, bounds, A, constraints, rhs,
                block_rules[1:], intra_rules
            )

        results = []
        sorted_block_labels = sorted(partition_map.keys(), reverse=True)
        for label in sorted_block_labels:
            sub_indices = np.array(partition_map[label])
            if len(sub_indices) == len(var_indices) and np.array_equal(sub_indices, var_indices):
                return self._recursive_block_vars(
                    level+1, var_indices, vars, obj_coeffs, bounds, A, constraints, rhs,
                    block_rules[1:], intra_rules
                )

            sub_results = self._recursive_block_vars(
                level+1, sub_indices, vars, obj_coeffs, bounds, A, constraints, rhs,
                block_rules[1:], intra_rules
            )

            adjusted_sub_results = [(vidx, (label,) + score_tuple) for vidx, score_tuple in sub_results]
            results.extend(adjusted_sub_results)

        return results

    def _partition_by_rule_vars(self, rule, var_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions variables into blocks based on the given block rule.
        Ensures that labels are properly assigned as scalars.
        """
        block_labels = np.array(rule.score_variables(
            [vars[i] for i in var_indices],
            obj_coeffs[var_indices] if obj_coeffs is not None else None,
            [bounds[i] for i in var_indices],
            A, constraints, rhs
        ))

        partition_map = defaultdict(list)
        for idx, lbl in zip(var_indices, block_labels):
            partition_map[int(lbl)].append(idx)  # Ensure label is an integer

        return partition_map


    def _apply_intra_rules_vars(self, var_indices, vars, obj_coeffs, bounds, A, constraints, rhs, intra_rules):
        """
        Apply intra-block rules to score variables within a final block.
        """
        # Fix: Use len() instead of direct boolean check
        if intra_rules is None or len(intra_rules) == 0:
            return [(idx, (0.0,)) for idx in var_indices]

        results = []
        for idx in var_indices:
            rule_scores = [
                rule.score_variables([vars[idx]], obj_coeffs[idx:idx+1], [bounds[idx]],
                                    A, constraints, rhs)[0]  # Extract scalar value
                for rule in intra_rules
            ]
            final_score_tuple = tuple(rule_scores)
            results.append((idx, final_score_tuple))

        return results


    # ----------------------------------------------------------------
    # Constraint Scoring
    # ----------------------------------------------------------------
    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        all_constr_indices = np.arange(len(constraints))
        results = self._recursive_block_constraints(
            level=0,
            constr_indices=all_constr_indices,
            vars=vars,
            obj_coeffs=obj_coeffs,
            bounds=bounds,
            A=A,
            constraints=constraints,
            rhs=rhs,
            block_rules=self.constr_block_rules,
            intra_rules=self.constr_intra_rules
        )

        results.sort(key=lambda x: x[0])
        return [score for (_, score) in results]

    def _recursive_block_constraints(self, level, constr_indices,
                                     vars, obj_coeffs, bounds, A, constraints, rhs,
                                     block_rules, intra_rules):
        if level >= self.max_depth or not block_rules:
            return self._apply_intra_rules_constraints(constr_indices, vars, obj_coeffs,
                                                       bounds, A, constraints, rhs, intra_rules)

        current_block_rule = block_rules[0]
        partition_map = self._partition_by_rule_constraints(
            current_block_rule, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs
        )

        if len(partition_map) == 1:
            return self._recursive_block_constraints(
                level+1, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs,
                block_rules[1:], intra_rules
            )

        results = []
        sorted_block_labels = sorted(partition_map.keys(), reverse=True)
        for label in sorted_block_labels:
            sub_indices = np.array(partition_map[label])
            if len(sub_indices) == len(constr_indices) and np.array_equal(sub_indices, constr_indices):
                return self._recursive_block_constraints(
                    level+1, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs,
                    block_rules[1:], intra_rules
                )

            sub_results = self._recursive_block_constraints(
                level+1, sub_indices, vars, obj_coeffs, bounds, A, constraints, rhs,
                block_rules[1:], intra_rules
            )

            adjusted_sub_results = [(cidx, (label,) + score_tuple) for cidx, score_tuple in sub_results]
            results.extend(adjusted_sub_results)

        return results

    def _partition_by_rule_constraints(self, rule, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions constraints into blocks based on the given block rule.
        """
        block_labels = np.array(rule.score_constraints(
            vars,
            obj_coeffs,
            bounds,
            A,
            [constraints[i] for i in constr_indices],
            rhs[constr_indices] if rhs is not None else None
        ))

        partition_map = defaultdict(list)
        for i, lbl in zip(constr_indices, block_labels):
            partition_map[int(lbl)].append(i)  # Ensure label is an integer

        return partition_map


    def _apply_intra_rules_constraints(self, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs, intra_rules):
        if intra_rules is None or len(intra_rules) == 0:
            return [(i, (0.0,)) for i in constr_indices]

        results = []
        for i in constr_indices:
            rule_scores = [rule.score_constraints(
                vars,
                obj_coeffs,
                bounds,
                A,
                [constraints[i]],  # Single constraint
                np.array([rhs[i]]) if rhs is not None else None
            )[0] for rule in intra_rules]
            final_score_tuple = tuple(rule_scores)
            results.append((i, final_score_tuple))

        return results
