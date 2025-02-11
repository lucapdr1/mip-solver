Below is a documentation-style explanation that describes the mathematical ideas behind the recursive hierarchical reordering procedure implemented in your code. You can include this in your project documentation to explain both the high‐level intuition and the formal structure of the algorithm.

---

# Mathematical Explanation of the Recursive Hierarchical Reordering Algorithm

The goal of the **RecursiveHierarchicalRuleComposition** is to compute a permutation of the variable (and constraint) indices that reflects a hierarchical (block‐wise) ordering of the matrix. The ordering is computed by recursively partitioning the matrix into “blocks” based on one or more block rules, and—if available—refining the order within each block using intra rules. Mathematically, the algorithm is a divide-and-conquer procedure that constructs a permutation σ (for variables) and a corresponding ordering for constraints.

## 1. Problem Setup

Let:
- **V** be the set of variable indices, \( V = \{0, 1, \dots, n-1\} \),
- **C** be the set of constraint (row) indices, \( C = \{0, 1, \dots, m-1\} \),
- **A** be the matrix (or problem data) representing the relationships between variables and constraints.

Our goal is to find an ordering (permutation) of the variables (and similarly for constraints) that “groups together” indices with similar structural properties.

## 2. Block Rules and Partitioning

A **block rule** \( R \) is a function that, given a subset of variable indices \( V' \subseteq V \) and constraint indices \( C' \subseteq C \), returns a partition of the block \( (V', C') \). Formally,  
\[
R(V', C') = \{ (V_1, C_1), (V_2, C_2), \dots, (V_k, C_k) \}
\]
where the sets \( V_i \) are disjoint (and their union is \( V' \)) and similarly for the constraint sets \( C_i \). In our implementation, block rules are realized by methods such as `score_matrix` in each rule class (e.g., in the **BoundCategoryRule**, **CardinalityRule**, or **SignPatternRule**).

### Partitioning Criterion

A block rule may, for example, group variables by:
- **Cardinality:** Variables with the same number (or range) of nonzero coefficients are grouped together.
- **Sign Pattern:** Variables whose columns have the same sign pattern (all nonnegative, all nonpositive, or mixed) are grouped together.
- **Bound Category:** Variables are grouped by the “category” of their bounds (both finite and nonnegative/nonpositive, straddling zero, one bound infinite, or both infinite).

Mathematically, if the rule \( R \) uses a scoring function \( f: V' \to \mathbb{Z} \) (or more generally, \( f: V' \to \mathbb{R} \)), then one way to partition is to define
\[
V_i = \{ v \in V' : f(v) = s_i \},
\]
with the corresponding constraint group often being the entire \( C' \) (if the rule does not partition constraints). The partition is then the collection of pairs \( \{ (V_i, C') : s_i \text{ is a unique score} \} \).

## 3. Recursive Partitioning

The algorithm defines a function \( F(V', C'; \mathcal{R}, \mathcal{I}) \) that returns an ordering of the indices in \( V' \) and \( C' \), where:
- \( \mathcal{R} \) is an ordered list of block rules.
- \( \mathcal{I} \) is an ordered list of intra rules (for ordering within blocks).

### The Recursive Definition

1. **Base Case:**  
   If either the recursion depth reaches a maximum \( L \) (to avoid infinite recursion) or no block rule remains, then the algorithm applies the intra rules.  
   - **Without intra rules:** The ordering is the identity (i.e., the indices remain in their given order).  
   - **With intra rules:** Each index \( v \) is assigned a score tuple  
     \[
     \text{score}(v) = \bigl( g_1(v), g_2(v), \dots, g_k(v) \bigr)
     \]
     where each \( g_i \) is the score produced by an intra rule. The ordering is then obtained by sorting \( V' \) lexicographically by these tuples.

2. **Recursive Case:**  
   Suppose the current block rule \( R \) (the first rule in the list) partitions \( (V', C') \) into subblocks:
   \[
   R(V', C') = \{ (V_1, C_1), (V_2, C_2), \dots, (V_k, C_k) \}
   \]
   If \( k = 1 \) (i.e., the rule did not actually partition the block), then the algorithm discards \( R \) (or in some variants, uses the remaining rules) and calls  
   \[
   F(V', C'; \mathcal{R}', \mathcal{I}),
   \]
   where \( \mathcal{R}' \) is \( \mathcal{R} \) with \( R \) removed.  
   If \( k > 1 \), then each block is ordered recursively:
   \[
   \sigma_i = F(V_i, C_i; \mathcal{R}', \mathcal{I}),
   \]
   and then the final ordering is the concatenation (in some prescribed order, e.g., descending order of block size) of the \( \sigma_i \)’s.

In our code, the recursion is implemented by the function `_recursive_block_matrix`. The key mathematical operation is:

- **Concatenation of Orderings:**  
  If the blocks are \( (V_1, C_1), \dots, (V_k, C_k) \) and their recursively computed orderings are \( \sigma_1, \dots, \sigma_k \), then the overall ordering is  
  \[
  \sigma(V', C') = \sigma_1 \, \| \, \sigma_2 \, \| \, \cdots \, \| \, \sigma_k,
  \]
  where “\(\|\)” denotes concatenation.

## 4. Intra Rules and Lexicographic Ordering

If intra rules are provided, each index \( i \) in a block is assigned a tuple of scores  
\[
\text{score}(i) = \left( g_1(i), g_2(i), \dots, g_p(i) \right)
\]
by calling methods like `score_matrix_for_variable` (or `score_matrix_for_constraint`). The lexicographic ordering is defined by:

- \( i \) precedes \( j \) if there exists an index \( k \) such that:
  - \( g_l(i) = g_l(j) \) for all \( l < k \), and  
  - \( g_k(i) < g_k(j) \).

In the absence of intra rules, the algorithm simply returns the indices in the order they are passed (the identity ordering within the block).

## 5. Final Ordering and Scoring

Once the full ordering is computed, the final step in the algorithm assigns each variable (or constraint) a “score” equal to its position in the ordering. That is, if the computed permutation is
\[
\sigma = (\sigma(0), \sigma(1), \dots, \sigma(n-1)),
\]
then the final score for variable \( v \) is defined as the index \( k \) such that \( \sigma(k) = v \).

---

# Summary

- **Partitioning:**  
  The algorithm partitions the matrix using block rules \( R \) that group indices according to structural properties (e.g., cardinality, sign pattern, bounds).

- **Recursion:**  
  It then recurses on each subblock, either reapplying the same block rules or—if none remain—using intra rules to refine the ordering via lexicographic tuples.

- **Concatenation:**  
  The final ordering is obtained by concatenating the orderings of the subblocks.

- **Final Score:**  
  The score for each variable (or constraint) is simply its position in the overall ordering.

This recursive hierarchical approach is a divide-and-conquer method that attempts to expose and exploit structure in the problem matrix. The mathematical operations underlying it are partitioning (using equivalence classes defined by the block rules), lexicographic ordering (via intra rules), and concatenation of ordered subblocks.

---

This documentation explains the “math behind the code” at both a conceptual and formal level. You can adjust the details to better match your specific implementation and desired properties.