import numpy as np

#Base interface
class DistanceMetric:
    def compute(self, perm1, perm2):
        raise NotImplementedError("Subclasses must implement this method")

#----------------------------------------------

class KendallTauDistance(DistanceMetric):
    def compute(self, perm1, perm2):
        # Assume self.kendall_tau_distance is implemented elsewhere.
        return self.kendall_tau_distance(perm1, perm2)

    def kendall_tau_distance(self, perm1, perm2):
        n = len(perm1)
        if len(perm2) != n:
            raise ValueError("Permutations must be the same length.")

        # Map elements of perm2 to their positions
        pos_in_perm2 = [0] * n
        for i, val in enumerate(perm2):
            pos_in_perm2[val] = i

        # Transform perm1 into the positions from perm2
        transformed = [pos_in_perm2[val] for val in perm1]

        # Use a merge sort based inversion count
        _, inv_count = self._count_inversions(transformed)
        return inv_count

    def _count_inversions(self, arr):
        # Base case: a single element has zero inversions
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, inv_left = self._count_inversions(arr[:mid])
        right, inv_right = self._count_inversions(arr[mid:])
        merged, inv_split = self._merge_count(left, right)
        return merged, inv_left + inv_right + inv_split

    def _merge_count(self, left, right):
        merged = []
        inv_count = 0
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i  # Count inversions: all remaining left items are greater
                j += 1
        # Append remaining elements (no inversions added)
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count

class HammingDistance(DistanceMetric):
    def compute(self, perm1, perm2):
        return self.hamming_distance(perm1, perm2)
    
    def hamming_distance(self, perm1, perm2):
        if len(perm1) != len(perm2):
            raise ValueError("Permutations must be the same length.")
        return np.sum(np.array(perm1) != np.array(perm2))

class AdjacencyAwareDistance(DistanceMetric):
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def compute(self, perm1, perm2):
        return self.adjacency_aware_distance(perm1, perm2, self.adjacency)
    
    def adjacency_aware_distance(self, perm1, perm2, adjacency):
        """
        Compare two permutations by how they place 'adjacent' constraints.
        
        For each pair (i, j) in adjacency, we look at the positions of i and j 
        in perm1 and perm2, and sum the difference of their distances.
        
        i.e. sum( abs( (pos1[i] - pos1[j]) - (pos2[i] - pos2[j]) ) ) over all i in [n], j in adjacency[i].
        
        We'll count only j > i to avoid double-counting in an undirected adjacency.
        """
        n = len(perm1)
        if len(perm2) != n:
            raise ValueError("Permutations must be same length.")

        # Build position lookups
        pos1 = [0] * n
        pos2 = [0] * n
        for idx, val in enumerate(perm1):
            pos1[val] = idx
        for idx, val in enumerate(perm2):
            pos2[val] = idx
        
        distance = 0
        # accumulate differences for adjacency pairs
        for i in range(n):
            for j in adjacency[i]:
                if j > i:  # to avoid double-counting i->j and j->i
                    diff_pos1 = abs(pos1[i] - pos1[j])
                    diff_pos2 = abs(pos2[i] - pos2[j])
                    distance += abs(diff_pos1 - diff_pos2)
        return distance

class CompositeDistance(DistanceMetric):
    def __init__(self, cluster_assignments, rcm_adjacency, alpha_cluster=1.0, beta_local=1.0):
        self.cluster_assignments = cluster_assignments
        self.rcm_adjacency = rcm_adjacency
        self.alpha_cluster = alpha_cluster
        self.beta_local = beta_local

    def compute(self, perm1, perm2):
        # Build position lookups.
        pos1 = {val: idx for idx, val in enumerate(perm1)}
        pos2 = {val: idx for idx, val in enumerate(perm2)}
    
        # Group indices by cluster.
        clusters = {}
        for idx, cluster in enumerate(self.cluster_assignments):
            clusters.setdefault(cluster, []).append(idx)
    
        cluster_distance = 0.0
        for cluster, indices in clusters.items():
            positions1 = [pos1[i] for i in indices]
            positions2 = [pos2[i] for i in indices]
            med1 = np.median(positions1)
            med2 = np.median(positions2)
            cluster_distance += abs(med1 - med2)
    
        intra_distance = 0.0
        for cluster, indices in clusters.items():
            for i in indices:
                for j in indices:
                    if i < j and j in self.rcm_adjacency.get(i, set()):
                        diff1 = abs(pos1[i] - pos1[j])
                        diff2 = abs(pos2[i] - pos2[j])
                        intra_distance += abs(diff1 - diff2)
    
        return self.alpha_cluster * cluster_distance + self.beta_local * intra_distance
