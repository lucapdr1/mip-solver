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
    
class WeightedKendallTauDistance:
    """
    Computes a weighted Kendall tau distance in O(n log n) for the special case
    where the weight function is defined as:
        w(i, j) = alpha(i) + alpha(j)
    with a default top-heavy alpha defined as alpha(i) = 1 / (1 + i).
    
    The computed distance is normalized to be in [0, 1].
    """
    
    def __init__(self):
        # Default top-heavy alpha: positions near the top (i.e. low index) get higher weight.
        self.alpha_func = lambda i: 1.0 / (1 + i)
        self.normalize = False

    def compute(self, perm1, perm2):
        """
        Public method to compute the weighted Kendall tau distance
        between perm1 and perm2.
        """
        return self.weighted_kendall_tau_distance(perm1, perm2)

    def weighted_kendall_tau_distance(self, perm1, perm2):
        """
        Transforms perm1 relative to perm2 and counts the weighted inversions
        using an augmented merge sort approach in O(n log n).
        """
        n = len(perm1)
        if len(perm2) != n:
            raise ValueError("Permutations must be the same length.")

        # Map elements of perm2 to their positions.
        pos_in_perm2 = [0] * n
        for idx, val in enumerate(perm2):
            pos_in_perm2[val] = idx

        # Transform perm1 into positions according to perm2.
        transformed = [pos_in_perm2[val] for val in perm1]

        # Prepare list with (transformed value, original index)
        arr_with_idx = [(val, i) for i, val in enumerate(transformed)]

        # Count weighted inversions using merge sort.
        _, weighted_inv_sum = self._merge_sort_count(arr_with_idx)

        if not self.normalize:
            return weighted_inv_sum

        # Compute total weight sum over all pairs (i, j) for i < j.
        total_weight = self._total_weight(n)
        return weighted_inv_sum / total_weight if total_weight > 0 else 0.0

    def _merge_sort_count(self, arr):
        """
        Recursively sorts the array (which contains (value, original_index)) and
        counts weighted inversions using the weight function w(i, j) = alpha(i) + alpha(j).
        """
        length = len(arr)
        if length <= 1:
            return arr, 0.0

        mid = length // 2
        left, inv_left = self._merge_sort_count(arr[:mid])
        right, inv_right = self._merge_sort_count(arr[mid:])
        merged, inv_split = self._merge_count_split(left, right)
        return merged, inv_left + inv_right + inv_split

    def _merge_count_split(self, left, right):
        """
        Merges two sorted arrays, counting weighted inversions between left and right.
        Each element is a tuple (value, original_index).
        """
        merged = []
        i = j = 0
        inv_count = 0.0

        # Precompute prefix sums of alpha for the left subarray.
        left_alpha = [self.alpha_func(item[1]) for item in left]
        prefix_sum_left = [0.0] * (len(left) + 1)
        for idx in range(len(left)):
            prefix_sum_left[idx + 1] = prefix_sum_left[idx] + left_alpha[idx]

        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                # All elements left[i:] are inversions with right[j]
                num_inversions = len(left) - i
                # Sum of alpha for remaining left elements.
                sum_alpha_left = prefix_sum_left[len(left)] - prefix_sum_left[i]
                alpha_right_j = self.alpha_func(right[j][1])
                inv_count += sum_alpha_left + alpha_right_j * num_inversions
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count

    def _total_weight(self, n):
        """
        Computes the total sum of weights for all pairs (i, j) with 0 <= i < j < n,
        where the weight is defined as:
            w(i, j) = alpha(i) + alpha(j)
        """
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += self.alpha_func(i) + self.alpha_func(j)
        return total
    
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
