import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from gurobipy import GRB

class BlockDecomposition:
    def __init__(self, A_matrix, logger=None):
        """
        Parameters
        ----------
        A_matrix : scipy.sparse.csr_matrix
            The coefficient matrix (constraints × variables).
        logger : logging.Logger, optional
            Logger for debugging or info messages.
        """
        self.A = A_matrix
        self.logger = logger

    def _decompose_variable_blocks(self):
        """
        Group variables into connected components based on the adjacency
        A^T × A. Each component is a "block" of variable indices.
        """
        if self.logger:
            self.logger.debug("Decomposing variable blocks...")

        A_var_adj = self.A.T @ self.A  # (vars × vars)
        # Convert to boolean adjacency
        A_var_adj_bool = A_var_adj.astype(bool).astype(int)

        n_components, labels = connected_components(A_var_adj_bool, directed=False)
        blocks = [[] for _ in range(n_components)]
        for var_idx, label in enumerate(labels):
            blocks[label].append(var_idx)

        if self.logger:
            self.logger.debug(f"Found {n_components} variable blocks.")
        return blocks

    def _decompose_constraint_blocks(self):
        """
        Group constraints into connected components based on the adjacency
        A × A^T. Each component is a "block" of constraint indices.
        """
        if self.logger:
            self.logger.debug("Decomposing constraint blocks...")

        A_constr_adj = self.A @ self.A.T  # (constrs × constrs)
        # Convert to boolean adjacency
        A_constr_adj_bool = A_constr_adj.astype(bool).astype(int)

        n_components, labels = connected_components(A_constr_adj_bool, directed=False)
        blocks = [[] for _ in range(n_components)]
        for constr_idx, label in enumerate(labels):
            blocks[label].append(constr_idx)

        if self.logger:
            self.logger.debug(f"Found {n_components} constraint blocks.")
        return blocks

    def _compute_block_scores(self, blocks, scores):
        """
        Compute the block scores by summing the *scores* for each index in a block.
        No extra weighting or priorities here—just the raw scores.
        """
        block_scores = []
        for block in blocks:
            block_score = sum(scores[idx] for idx in block)
            block_scores.append(block_score)
        return block_scores

    def _sort_blocks_by_score(self, blocks, block_scores):
        """
        Sort blocks in descending order based on block_scores.
        Returns the sorted blocks (dropping scores).
        """
        blocks_with_scores = zip(blocks, block_scores)
        sorted_tuples = sorted(blocks_with_scores, key=lambda x: -x[1])
        sorted_blocks = [block for block, _ in sorted_tuples]
        return sorted_blocks

    def _sort_indices_within_blocks(self, blocks, scores):
        """
        Sort each block's indices by descending 'scores' and
        return a single combined ordering (1D array).
        """
        sorted_order = []
        for block in blocks:
            # Sort by descending order of 'scores'
            sorted_block = sorted(block, key=lambda idx: -scores[idx])
            sorted_order.extend(sorted_block)
        return np.array(sorted_order)

    def get_variable_order(self, gurobi_vars, var_scores):
        """
        Decompose variable blocks, assign priorities, compute block scores,
        and return a final variable ordering (array of indices).

        Parameters
        ----------
        gurobi_vars : list of gurobipy.Var
            The list of variables from the Gurobi model.
        var_scores : array-like
            The scores for each variable.

        Returns
        -------
        var_order : numpy.ndarray
            The variable indices in the desired (sorted) order.
        """
        # Decompose into variable blocks
        var_blocks = self._decompose_variable_blocks()

        # Compute block scores
        block_scores = self._compute_block_scores(var_blocks, var_scores)
        # Sort blocks by scores
        sorted_blocks = self._sort_blocks_by_score(var_blocks, block_scores)
        # Sort indices within blocks by var_scores
        var_order = self._sort_indices_within_blocks(sorted_blocks, var_scores)
        return var_order

    def get_constraint_order(self, gurobi_constrs, constr_scores):
        """
        Decompose constraint blocks, assign priorities, compute block scores,
        and return a final constraint ordering (array of indices).

        Parameters
        ----------
        gurobi_constrs : list of gurobipy.Constr
            The list of constraints from the Gurobi model.
        constr_scores : array-like
            The scores for each constraint.

        Returns
        -------
        constr_order : numpy.ndarray
            The constraint indices in the desired (sorted) order.
        """
        # Decompose into constraint blocks
        constr_blocks = self._decompose_constraint_blocks()

        # Compute block scores
        block_scores = self._compute_block_scores(constr_blocks, constr_scores)
        # Sort blocks by scores
        sorted_blocks = self._sort_blocks_by_score(constr_blocks, block_scores)
        # Sort indices within blocks by constr_scores
        constr_order = self._sort_indices_within_blocks(sorted_blocks, constr_scores)
        return constr_order
