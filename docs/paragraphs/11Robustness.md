
## Problem Normalization
#### **1. Normalization for Scaling Invariance**
- **Objective:** Remove arbitrary scaling effects (i.e. multiplying rows and columns by scalars) while preserving the discrete structure of MILO problems.
- **Steps:**
  - **Column Normalization:**  
    - **Continuous Variables:**  
      For each continuous variable (column), compute the L₂ norm of its nonzero coefficients. Define the scaling factor as the reciprocal of that norm. Adjust the objective coefficient as  
      \[
      c_i \leftarrow \frac{c_i}{s_i}.
      \]
    - **Discrete Variables:**  
      For each integer or binary variable (including those with bounds of the form \([a,a+1]\)), compute the scaling factor using the greatest common divisor (GCD) of the rounded absolute values of its nonzero coefficients. (Ignore any values below a tolerance threshold.) This produces an integer norm, and the scaling factor is then  
      \[
      s_i = \frac{1}{\text{GCD}}.
      \]
      Adjust the objective coefficient as  
      \[
      c_i \leftarrow \text{round}\!\left(\frac{c_i}{s_i}\right),
      \]
      ensuring that the normalized objective coefficient remains an integer.
  - **Row Normalization:**  
    - For each constraint (row), if the row contains any discrete variable, compute the row norm using only the coefficients corresponding to the discrete variables (via the GCD of their rounded values). Then, scale the entire row—including any continuous coefficients—by the reciprocal of this norm. Adjust the RHS as  
      \[
      b_j \leftarrow b_j \cdot r_j,\quad \text{with} \quad r_j = \frac{1}{\text{row norm}}.
      \]
    - If the row does not contain any discrete variable, compute the row norm using the L₂ norm.
- **Outcome:**  
  After normalization, continuous variables are scaled by their L₂ norms, while discrete variables are scaled by factors that are reciprocals of integers (preserving integrality). The objective function and variable bounds are adjusted accordingly.

#### **2. Building the Normalized Model**
- **Objective:** Reconstruct a new MILO model using the normalized data.
- **Steps:**
  - **Variable Creation:**  
    Build a new model with variables having normalized bounds and preserving their original types.
  - **Objective Insertion:**  
    Insert the normalized objective function using the normalized objective coefficients (for discrete variables, these coefficients are forced to be integers).
  - **Constraint Reconstruction:**  
    Add constraints using the normalized constraint matrix and normalized right-hand sides, preserving the original constraint senses (≤, ≥, or =).
- **Outcome:**  
  The resulting normalized model is mathematically equivalent to the original and maintains the discrete properties of integer and binary variables.

#### **3. Limitations**
- **Dependence on Initial Scaling:**  
  Although the normalization procedure removes arbitrary scaling, starting from different equivalent scalings may still produce different canonical forms. In practice, if many rows and columns are scaled, slight numerical differences may persist.

- **Incomplete Invariance:**  
  Even with the discrete-aware adjustments, the final canonical form may not be exactly identical for all equivalent formulations—particularly when a large number of rows and columns are scaled—though the underlying mathematical model remains equivalent.

---

This revised process uses a discrete-aware strategy for normalization by computing scaling factors via the L₂ norm for continuous variables and via the GCD (of rounded values) for discrete variables. The objective and bounds are adjusted accordingly, and the normalized model is built by preserving the original constraint senses. However, due to the inherent sensitivity when many rows and columns are scaled, different equivalent scalings may not always produce an identical canonical form.