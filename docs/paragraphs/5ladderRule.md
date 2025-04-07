
# Math Behind the LadderIntraRule Ordering

The goal of this ordering rule is to assign a “score” to each variable (and constraint) based on the pattern of nonzero entries in the constraint matrix. The resulting scores determine the order in which variables appear (or are processed) later in the algorithm. The rule uses a **lexicographic ordering** of the column patterns of the constraint matrix. In addition, it uses an **identity ordering** when the current problem block is “small” relative to the original full matrix.

## 1. Definitions and Preliminaries

- **Constraint Matrix, \( A \):**  
  A sparse matrix representing the constraints in a linear (or mixed-integer) programming model. The matrix is stored in a compressed sparse column (CSC) format, which efficiently represents each column’s nonzero row indices.

- **CSC Format Components:**
  - `A_csc.indptr`: An array of length \( n+1 \) where \( n \) is the number of variables (columns). For each variable \( i \), the nonzero entries are found in `A_csc.indices[A_csc.indptr[i]:A_csc.indptr[i+1]]`.
  - `A_csc.indices`: An array of row indices corresponding to the nonzero entries of the matrix.

- **Original Matrix Size:**  
  Denoted by \( \text{original\_var\_count} \) and \( \text{original\_constr\_count} \). The full matrix size is given by:  
  \[
  \text{Total Matrix Size} = \text{original\_var\_count} \times \text{original\_constr\_count}
  \]

- **Current Block Size:**  
  Denoted by \( n \times m \) (current number of variables and constraints). It is used to decide whether to apply the lexicographic ordering or revert to an identity ordering.

## 2. Threshold Check and Identity Ordering

Before computing the lexicographic keys, the rule checks whether the size of the current block is significantly smaller than the full problem. Specifically, if:
\[
n \times m < \text{threshold} \times (\text{original\_var\_count} \times \text{original\_constr\_count})
\]
then **identity ordering** is used. Identity ordering means that each variable is assigned a score equal to its index:
\[
\text{score}[i] = i
\]
This is mathematically trivial but ensures that if only a small “slice” of the full problem is considered, no reordering is performed.

## 3. Lexicographic Key Computation

When the block size exceeds the threshold, each variable’s column is assigned a key based on its pattern of nonzero entries:

1. **For each variable \( i \):**
   - Extract the subarray:
     \[
     \text{pattern}_i = \text{A\_csc.indices}[\text{A\_csc.indptr}[i] : \text{A\_csc.indptr}[i+1]]
     \]
   - The resulting pattern is a tuple of row indices where the variable has nonzero entries:
     \[
     \text{key}[i] = (\text{row}_{1}, \text{row}_{2}, \dots, \text{row}_{k})
     \]

2. **Handling Empty Columns:**  
   If a variable \( i \) has no nonzeros (i.e., \( \text{A\_csc.indptr}[i+1] - \text{A\_csc.indptr}[i] = 0 \)), it is assigned a key:
   \[
   \text{key}[i] = (m,)
   \]
   where \( m \) is the number of constraints. This key is chosen to be “large” so that variables with no nonzero entries appear later in the sorted order.

## 4. Lexicographic Sorting

Once each variable has an associated key, the keys are sorted **lexicographically**. This means that the keys are compared element by element:
- **Lexicographic Order Definition:**  
  For two tuples \( a = (a_1, a_2, \dots, a_k) \) and \( b = (b_1, b_2, \dots, b_k) \), \( a \) is considered smaller than \( b \) if there exists an index \( j \) such that:
  \[
  a_1 = b_1, \quad a_2 = b_2, \quad \dots, \quad a_{j-1} = b_{j-1} \quad \text{and} \quad a_j < b_j.
  \]
- **Outcome:**  
  Variables with nonzero entries in earlier (lower-indexed) rows will have smaller keys and are therefore ordered to the left (appear earlier in the list).

## 5. Assigning Scores

After sorting, each variable’s score is defined as its **rank** in the sorted order:
\[
\text{score}[i] = \text{rank of variable } i \text{ in the lexicographic order}
\]
These scores are then used by subsequent processing steps to decide the ordering of variables.

## 6. Constraints Ordering

For constraints, the ordering is much simpler: an identity ordering is always applied. Thus, for each constraint:
\[
\text{score}_{\text{constraint}}[i] = i
\]
This means that constraints retain their original ordering regardless of the structure of \( A \).

---

## Summary

- **Threshold Check:**  
  If the current block size is below a threshold fraction of the total problem size, the ordering is identity (i.e., \( \text{score}[i]=i \)).

- **Lexicographic Key for Variables:**  
  For each variable, the key is formed from the nonzero row indices in its corresponding column of \( A \) (or a high value key for empty columns). Variables are then sorted lexicographically.

- **Final Scores:**  
  The score for each variable is its rank in the sorted order. This approach pushes variables with nonzeros in lower-indexed rows (indicating “earlier” constraints) to the left.

This mathematical reasoning ensures that the ordering rule respects the structure of the constraint matrix, leading to a “ladder-style” arrangement where variables are ordered in a manner that may benefit subsequent processing in optimization or other algorithms.