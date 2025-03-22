from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components

class ReverseCuthillMcKeeRule(OrderingRule):
    """
    An ordering rule based on the Reverse Cuthill-McKee (RCM) algorithm.
    
    This rule computes a permutation of constraints (rows) by:
      1. Constructing a graph of constraints using the nonzero structure of A (via A_csr * A_csrᵀ).
      2. Running the RCM algorithm to compute an ordering that reduces matrix bandwidth.
      3. Assigning a score to each constraint so that a lower (RCM) rank produces a higher score.
      
    In the score_matrix method, the entire block of constraints is reordered
    according to the RCM permutation.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not affect variable ordering.
        return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Computes scores for constraints using RCM ordering.
        
        Steps:
          1. Construct the constraint graph as G = A_csr * A_csrᵀ.
          2. Zero out the diagonal (self-connections) and eliminate explicit zeros.
          3. Compute the RCM permutation.
          4. Assign a score to each constraint such that the first element in the permutation
             gets score = scaling * (n), the second scaling*(n-1), etc.
        """
        # Construct the constraint graph.
        # Each constraint is a node; an edge exists if two constraints share a variable.
        G = A_csr.dot(A_csr.transpose())
        G.setdiag(0)
        G.eliminate_zeros()
        
        # Compute the RCM ordering.
        rcm_order = reverse_cuthill_mckee(G)
        n = len(rcm_order)
        
        # Assign scores: higher score for constraints appearing earlier in the RCM order.
        scores = np.empty(n, dtype=int)
        for rank, idx in enumerate(rcm_order):
            scores[idx] = self.scaling * (n - rank)
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not affect the ordering of variables.
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Compute the overall constraint scores and return the score for the specified constraint.
        scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        return (scores[idx],)

    from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee

class ReverseCuthillMcKeeRule(OrderingRule):
    """
    An ordering rule based on the Reverse Cuthill-McKee (RCM) algorithm.
    
    This rule computes a permutation of constraints (rows) by:
      1. Constructing a symmetric constraint graph from A_csr (using A_csr * A_csrᵀ).
      2. Running the RCM algorithm to compute an ordering that reduces matrix bandwidth.
      3. Assigning a score to each constraint so that a lower RCM rank (i.e. closer to the front)
         produces a higher score.
         
    The score_matrix method then partitions the block using the scores computed by score_constraints.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not affect variable ordering.
        return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Computes scores for constraints using RCM ordering.
        
        Steps:
          1. Construct the symmetric constraint graph: G = A_csr * A_csrᵀ.
          2. Zero out the diagonal (self-connections) and eliminate explicit zeros.
          3. Compute the RCM ordering.
          4. Assign a score to each constraint so that the first element in the RCM ordering
             gets score = scaling * n, the second gets scaling * (n-1), etc.
        """
        # Build the symmetric graph: each constraint is a node.
        G = A_csr.dot(A_csr.transpose())
        G = G.tolil()         # Convert to LIL for efficient modifications
        G.setdiag(0)
        G = G.tocsr()         # Convert back to CSR if needed
        G.eliminate_zeros()
        
        # Compute the RCM ordering on the symmetric graph.
        rcm_order = reverse_cuthill_mckee(G)
        n = len(rcm_order)
        
        # Initialize scores array.
        scores = np.empty(n, dtype=int)
        # Assign scores: higher score for constraints appearing earlier in the RCM order.
        for rank, idx in enumerate(rcm_order):
            scores[idx] = self.scaling * (n - rank)
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not affect variable ordering.
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the score for a single constraint by leveraging score_constraints.
        """
        scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        return (scores[idx],)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block defined by var_indices and constr_indices based on the
        RCM-based score computed on the corresponding A.
        
        Steps:
          1. Compute constraint scores by calling score_constraints.
          2. Group the original constraint indices by their computed scores using NumPy.
          3. Group all variable indices together (since this rule does not affect variables).
          4. Sort the unique scores in descending order.
          5. Form the partition map as the Cartesian product of these groups.
          
        Returns a dictionary mapping block labels to tuples of NumPy arrays:
            { label: (var_indices_array, constr_indices_array) }
        """
        # Compute constraint scores on the A.
        sub_scores = np.array(self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # Group the original constraint indices by their computed scores.
        unique_scores = np.unique(sub_scores)
        constr_groups = {}
        for score in unique_scores:
            mask = (sub_scores == score)
            constr_groups[score] = constr_indices[mask]
        
        # All variables are grouped together.
        # Sort the unique scores in descending order.
        sorted_scores = np.sort(np.array(list(constr_groups.keys())))[::-1]
        
        # Form the partition map: for each sorted score, assign the entire var_indices and the corresponding constraint group.
        partition_map = {}
        label = 0
        for score in sorted_scores:
            partition_map[label] = (var_indices, constr_groups[score])
            label += 1
        return partition_map

class AdjacencyClusteringRule(OrderingRule):
    """
    An ordering rule that clusters constraints based on their adjacency.
    
    Steps:
      1. Build the constraint adjacency graph via G = A_csr * A_csrᵀ.
         (Convert to LIL to set the diagonal to zero, then back to CSR.)
      2. Compute connected components (clusters) using the graph.
      3. Compute the degree of each constraint (number of adjacent constraints).
      4. Assign each constraint a composite score:
             score[i] = scaling * ( cluster_sizes[comp[i]] * factor + degree[i] )
         where factor is chosen (e.g., max_degree + 1) to make cluster size dominate.
      5. In score_matrix, group constraint indices by their computed scores (in descending order),
         while leaving the variable indices unchanged.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not affect variable ordering.
        return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Computes composite scores for constraints based on clustering and connectivity.
        
        Steps:
          1. Compute the constraint adjacency graph: G = A_csr * A_csrᵀ.
          2. Convert G to LIL for efficient modifications, zero its diagonal, convert back to CSR.
          3. Compute connected components of G to identify clusters.
          4. For each constraint, compute its degree (number of nonzero entries in its row).
          5. Let factor = (max_degree + 1) so that cluster size differences dominate.
          6. For each constraint i, assign:
                   score[i] = scaling * (cluster_sizes[component[i]] * factor + degree[i])
        """
        # Build the constraint adjacency graph.
        G = A_csr.dot(A_csr.transpose())
        G = G.tolil()         # Convert to LIL for efficient diagonal modification.
        G.setdiag(0)          # Remove self-loops.
        G = G.tocsr()         # Convert back to CSR.
        G.eliminate_zeros()
        
        # Compute connected components (clusters) of the constraint graph.
        n_components, labels = connected_components(G, directed=False, connection='weak')
        # Compute cluster sizes: how many constraints belong to each component.
        cluster_sizes = np.bincount(labels)
        
        # Compute the degree for each constraint.
        degrees = np.array(G.getnnz(axis=1)).flatten()
        factor = degrees.max() + 1  # Choose a factor to ensure cluster size differences dominate.
        
        n = len(labels)
        scores = np.empty(n, dtype=int)
        for i in range(n):
            scores[i] = self.scaling * (cluster_sizes[labels[i]] * factor + degrees[i])
        return scores

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule does not affect the ordering of variables.
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a tuple with the score for constraint at index idx by calling score_constraints.
        """
        scores = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs)
        return (scores[idx],)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block defined by var_indices and constr_indices based on the
        composite clustering scores computed on A.
        
        Steps:
          1. Compute the constraint scores via score_constraints.
          2. Group the original constraint indices by these scores.
          3. Since this rule does not differentiate variables, group all var_indices together.
          4. Sort the unique scores in descending order.
          5. Form the partition map as a dictionary where each key maps to a tuple:
             (var_indices, corresponding sorted constraint indices).
        
        Returns:
            A dictionary mapping block labels to tuples of NumPy arrays.
        """
        # Compute constraint scores.
        sub_scores = np.array(self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs))
        
        # Group the original constraint indices by their computed scores.
        unique_scores = np.unique(sub_scores)
        constr_groups = {}
        for score in unique_scores:
            mask = (sub_scores == score)
            constr_groups[score] = constr_indices[mask]
        
        # Sort unique scores in descending order.
        sorted_scores = np.sort(np.array(list(constr_groups.keys())))[::-1]
        
        # Form the partition map: each block gets all var_indices and its associated constraints.
        partition_map = {}
        label = 0
        for score in sorted_scores:
            partition_map[label] = (var_indices, constr_groups[score])
            label += 1
        return partition_map

