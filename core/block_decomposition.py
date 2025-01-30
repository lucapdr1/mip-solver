import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from gurobipy import GRB
from collections import deque

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

    def _decompose_variable_blocks(self):
        """
        Group variables into connected components by a bipartite BFS
        in the original A matrix, avoiding (A^T @ A).
        """
        if self.logger:
            self.logger.debug("Decomposing variable blocks (bipartite BFS)...")

        A_csr = self.A  # shape = (m, n) => m constraints, n variables
        m, n = A_csr.shape

        # -- Build adjacency lists for the bipartite graph:
        #    var -> list of constraint indices
        #    constr -> list of variable indices
        var_to_constr = [[] for _ in range(n)]
        constr_to_var = [[] for _ in range(m)]

        # We'll use the CSR structure to iterate row-by-row:
        for i in range(m):  # each constraint i
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            # The columns in A_csr.indices[row_start:row_end] are the variables with nonzero in row i
            for idx in range(row_start, row_end):
                j = A_csr.indices[idx]  # variable index
                var_to_constr[j].append(i)
                constr_to_var[i].append(j)

        # -- We'll do a BFS over the bipartite graph, but track only variable connectivity.
        visited_vars = [False] * n
        blocks = []  # each element will be a list of variable indices

        for start_var in range(n):
            if not visited_vars[start_var]:
                # Start a new block
                new_block = []
                visited_vars[start_var] = True
                queue = deque([start_var])

                while queue:
                    var = queue.popleft()
                    new_block.append(var)
                    # For each constraint connected to 'var'
                    for c in var_to_constr[var]:
                        # Then for each variable connected to 'c'
                        for other_var in constr_to_var[c]:
                            if not visited_vars[other_var]:
                                visited_vars[other_var] = True
                                queue.append(other_var)

                blocks.append(new_block)

        # 'blocks' now contains the connected sets of variables.
        n_components = len(blocks)
        if self.logger:
            self.logger.info(f"Found {n_components} variable blocks via bipartite BFS.")
        return blocks
    def _decompose_constraint_blocks(self):
        """
        Group constraints into connected components via a bipartite BFS
        in the original A matrix, avoiding (A @ A^T).
        Each block is a list of constraint indices.
        """
        if self.logger:
            self.logger.debug("Decomposing constraint blocks (bipartite BFS)...")

        A_csr = self.A  # shape = (m, n),  m constraints × n variables
        m, n = A_csr.shape

        # -- Build adjacency lists for the bipartite graph:
        #    constr -> list of variable indices
        #    var -> list of constraint indices
        constr_to_var = [[] for _ in range(m)]
        var_to_constr = [[] for _ in range(n)]

        for i in range(m):  # each constraint i
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            for idx in range(row_start, row_end):
                j = A_csr.indices[idx]  # variable index
                constr_to_var[i].append(j)
                var_to_constr[j].append(i)

        visited_constr = [False] * m
        blocks = []

        # BFS over constraints to find connected components
        for start_constr in range(m):
            if not visited_constr[start_constr]:
                new_block = []
                visited_constr[start_constr] = True
                queue = deque([start_constr])

                while queue:
                    c = queue.popleft()
                    new_block.append(c)
                    # For each variable in this constraint
                    for v in constr_to_var[c]:
                        # Then for each constraint in that variable
                        for other_c in var_to_constr[v]:
                            if not visited_constr[other_c]:
                                visited_constr[other_c] = True
                                queue.append(other_c)

                blocks.append(new_block)

        n_components = len(blocks)
        if self.logger:
            self.logger.info(f"Found {n_components} constraint blocks via bipartite BFS.")
        return blocks

