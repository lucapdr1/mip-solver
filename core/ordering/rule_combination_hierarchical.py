import numpy as np
from core.ordering.ordering_rule_interface import OrderingRule

class HierarchicalRuleComposition(OrderingRule):
    """
    - var_block_rules: list[OrderingRule]
         Each rule's `score_variables()` produces a numeric array for blocking.
    - var_intra_rules: list[OrderingRule]
         Each rule's `score_variables()` produces a numeric array for ordering within a block.
    - constr_block_rules: list[OrderingRule]
    - constr_intra_rules: list[OrderingRule]
    - block_rank_factor: large constant ensuring different blocks don't mix in the final sort.
    """
    def __init__(self,
                 var_block_rules,
                 var_intra_rules,
                 constr_block_rules,
                 constr_intra_rules,
                 block_rank_factor=1e6):
        self.var_block_rules = var_block_rules or []
        self.var_intra_rules = var_intra_rules or []
        self.constr_block_rules = constr_block_rules or []
        self.constr_intra_rules = constr_intra_rules or []
        self.block_rank_factor = block_rank_factor

    # ==========================
    #  Variable scoring
    # ==========================
    def score_variables(self, vars, obj_coeffs, A, bounds):
        """
        Return final scores that can be used in a single .argsort(descending).
        We do:
            (1) gather block scores from all block rules => a tuple per variable
            (2) unique them, assign block ranks
            (3) sum up intra-block rule scores
            (4) final_score = (MAX_RANK - rank) * block_rank_factor + (intra_score)
        """
        # 1) Gather block-scores for each variable from each block rule
        #    result => shape (nvars, n_block_rules)
        block_scores_arrays = [
            rule.score_variables(vars, obj_coeffs, A, bounds)
            for rule in self.var_block_rules
        ]
        # transpose => list of tuples
        # block_scores_arrays is a list of lists, each length nvars
        # so we zip them up
        block_scores_arrays = np.array(block_scores_arrays)  # shape (n_block_rules, nvars)
        block_scores_arrays = block_scores_arrays.T          # shape (nvars, n_block_rules)

        # Convert each row to a tuple => label
        block_score_tuples = [tuple(row) for row in block_scores_arrays]

        # 2) Determine unique tuples, assign block rank
        unique_tuples = list(set(block_score_tuples))
        unique_tuples.sort()  # lexicographic ascending order (customize if needed)

        # map each tuple -> rank
        tuple_to_rank = {tpl: i for i, tpl in enumerate(unique_tuples)}
        block_ranks = np.array([tuple_to_rank[tpl] for tpl in block_score_tuples])
        nblocks = len(unique_tuples)
        
        # 3) Sum intra-block scores
        intra_sum = np.zeros(len(vars))
        for rule in self.var_intra_rules:
            intra_sum += rule.score_variables(vars, obj_coeffs, A, bounds)

        # 4) Combine: we want block 0 to appear *first* in a descending sort,
        #    so do (nblocks - rank) * big_constant + intra_score
        #    That ensures block 0 has the largest final_score, block 1 next, etc.
        final_scores = (nblocks - block_ranks) * self.block_rank_factor + intra_sum
        return final_scores

    # ==========================
    # Constraint scoring
    # ==========================
    def score_constraints(self, constraints, A, rhs):
        """
        Same logic as for variables.
        """
        block_scores_arrays = [
            rule.score_constraints(constraints, A, rhs)
            for rule in self.constr_block_rules
        ]
        block_scores_arrays = np.array(block_scores_arrays)  # shape (n_block_rules, nconstraints)
        block_scores_arrays = block_scores_arrays.T          # shape (nconstraints, n_block_rules)

        block_score_tuples = [tuple(row) for row in block_scores_arrays]

        unique_tuples = list(set(block_score_tuples))
        unique_tuples.sort()  # lexicographic ascending
        tuple_to_rank = {tpl: i for i, tpl in enumerate(unique_tuples)}
        block_ranks = np.array([tuple_to_rank[tpl] for tpl in block_score_tuples])
        nblocks = len(unique_tuples)

        intra_sum = np.zeros(len(constraints))
        for rule in self.constr_intra_rules:
            intra_sum += rule.score_constraints(constraints, A, rhs)

        final_scores = (nblocks - block_ranks) * self.block_rank_factor + intra_sum
        return final_scores
