# 1. Variable Score

For **each variable** \(i\), its individual score, denoted \(\text{Score}^v_i\), is computed as:

\[
\boxed{
\text{Score}^v_i 
\;=\; 
1000 \,\cdot\, P(v_i)
\;+\;
100 \,\cdot\, \log\!\bigl(1 + |\text{Obj}_i|\bigr)
\;+\;
10 \,\cdot\, \sum_{j \in \Omega(i)} \log\!\bigl(1 + \bigl|a_{j,i}\bigr|\bigr)
\;+\;
10 \,\cdot\, \#\Omega(i)
}
\]

## 1.1. Parameters for Variable Score

- \(\mathbf{P(v_i)}\): **Type Priority** of variable \(i\).  
  - \(P(v_i) = 3\) if \(v_i\) is **Binary** (B).  
  - \(P(v_i) = 2\) if \(v_i\) is **Integer** (I).  
  - \(P(v_i) = 1\) if \(v_i\) is **Continuous** (C).  
  This term is multiplied by \(1000\) so that the variable type has a dominant impact on the final score.

- \(\mathbf{|\text{Obj}_i|}\): **Absolute value** of the **objective coefficient** for variable \(i\).  
  - The formula uses \(100 \cdot \log(1 + |\text{Obj}_i|)\), so large objective coefficients lead to higher (log-scaled) contributions.

- \(\sum_{j \in \Omega(i)} \log\!\bigl(1 + |a_{j,i}|\bigr)\):  
  - For **each constraint** \(j\) where variable \(i\) has a **nonzero** coefficient \(a_{j,i}\), take \(\log(1 + |a_{j,i}|)\) and **sum** these values.  
  - Multiply by \(10\).  
  - This rewards variables that have large coefficients within constraints (but in a log-scaled way to avoid extreme dominance).

- \(\mathbf{\#\Omega(i)}\): The **count** of constraints in which variable \(i\) appears with a nonzero coefficient (its **occurrence** count).  
  - Multiplied by \(10\), so variables that appear in more constraints get a higher score.

---

# 2. Constraint Score

For **each constraint** \(j\), its individual score, denoted \(\text{Score}^c_j\), is computed as:

\[
\boxed{
\text{Score}^c_j 
\;=\;
1000 \,\cdot\, P(c_j)
\;+\;
100 \,\cdot\, \log\!\bigl(1 + |\text{RHS}_j|\bigr)
\;+\;
10 \,\cdot\, \sum_{i \in \Theta(j)} \log\!\bigl(1 + \bigl|a_{j,i}\bigr|\bigr)
\;+\;
1 \,\cdot\, \log\!\bigl(1 + \bigl|\text{Range}_j\bigr|\bigr)
}
\]

## 2.1. Parameters for Constraint Score

- \(\mathbf{P(c_j)}\): **Type Priority** of constraint \(j\).  
  - \(P(c_j) = 3\) if the constraint sense is **\(\ge\)**.  
  - \(P(c_j) = 2\) if the sense is **\(=\)**.  
  - \(P(c_j) = 1\) if the sense is **\(\le\)**.  
  This factor is multiplied by \(1000\) so that constraint sense has a dominant effect on the final score.

- \(\mathbf{|\text{RHS}_j|}\): **Absolute value** of the **right-hand side** (RHS) of constraint \(j\).  
  - The formula uses \(100 \cdot \log(1 + |\text{RHS}_j|)\), so constraints with large RHS values get a higher (log-scaled) contribution.

- \(\sum_{i \in \Theta(j)} \log\!\bigl(1 + |a_{j,i}|\bigr)\):  
  - For **each variable** \(i\) with a **nonzero** coefficient \(a_{j,i}\) in constraint \(j\), take \(\log(1 + |a_{j,i}|)\) and **sum** these values.  
  - Multiply by \(10\).  
  - This boosts constraints containing large coefficients, again log-scaled.

- \(\log\!\bigl(1 + |\text{Range}_j|\bigr)\):  
  - **Range** might be defined (as in your code) by:  
    \[
      \text{Range}_j = \max_{i \in \Theta(j)}(a_{j,i}) \;-\; \min_{i \in \Theta(j)}(a_{j,i}),
    \]
    or any other measure of row “width.”  
  - This term is multiplied by \(1\). Even so, it can still affect the ordering because constraints with bigger coefficient ranges have a higher \(\log(1+|\text{Range}_j|)\).

---

# 3. Block Decomposition & Block Ordering

Once you have **individual item scores** (for variables or constraints), you:

1. **Form “blocks”** of items, typically by **connected components** in your bipartite graph (\(A\) matrix).  
   - A **variable block** \(B_k\) is a set of variable indices that are all reachable from each other via shared constraints.  
   - A **constraint block** \(C_\ell\) is a set of constraint indices that are all reachable from each other via shared variables.

2. **Compute the “Block Score”** as the **sum** of the **individual** scores of the items in that block.  

   - For a variable block \(B_k\):
     \[
       \text{BlockScore}^v(B_k) 
       \;=\;
       \sum_{i \in B_k} \text{Score}^v_i.
     \]

   - For a constraint block \(C_\ell\):
     \[
       \text{BlockScore}^c(C_\ell)
       \;=\;
       \sum_{j \in C_\ell} \text{Score}^c_j.
     \]

3. **Order the Blocks** in **descending** order of their block score.  
   - The block with the highest sum of item scores is placed first, etc.

4. **Within Each Block**, order the items themselves by their own **descending** item score.  
   - That means, for block \(B_k\), you sort the variables in \(B_k\) by \(\text{Score}^v_i\) from largest to smallest.  
   - The same applies for constraints in \(C_\ell\).

Hence, the final ordering (top to bottom) respects:  
1. **Block rank** (based on summed item scores),  
2. **In-block item rank** (based on each item’s own score).

---

# 4. Summary of the Overall Workflow

1. **Compute individual scores for each variable** (or constraint) using the formulas above.  
2. **Decompose** variables (or constraints) into **blocks**—e.g., by bipartite BFS on \(A\).  
3. **Sum** the per-item scores within each block to get a **block score**.  
4. **Sort** the blocks in descending block score.  
5. **Sort** each block’s items in descending per-item score.  
6. **Concatenate** the blocks (in sorted order), placing each block’s items in their sorted order.  

That final concatenation is your **desired ordering** for variables or constraints.

---

## 5. Glossary of Symbols

- \(m\), \(n\): Number of constraints and variables, respectively.  
- \(a_{j,i}\): Coefficient of variable \(i\) in constraint \(j\).  
- \(\Omega(i)\): Set of constraints where variable \(i\) has a **nonzero** coefficient.  
- \(\Theta(j)\): Set of variables that appear with **nonzero** coefficients in constraint \(j\).  
- \(\text{Obj}_i\): Objective coefficient of variable \(i\).  
- \(\text{RHS}_j\): Right-hand side of constraint \(j\).  
- \(\text{Range}_j\): Some measure of the “span” of coefficients in constraint \(j\). Often \(\max_i(a_{j,i}) - \min_i(a_{j,i})\).  
- \(P(v_i)\): Type-based priority for variable \(i\). (3 for binary, 2 for integer, 1 for continuous.)  
- \(P(c_j)\): Type-based priority for constraint \(j\). (3 for \(\ge\), 2 for \(=\), 1 for \(\le\).)  
- \(\#\Omega(i)\): Count of constraints that contain variable \(i\).  
- \(\log\!\bigl(1 + x\bigr)\): Natural logarithm (or base-10, depending on implementation) of \(1 + |x|\) to dampen extreme values.
