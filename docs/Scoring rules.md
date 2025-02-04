# **Rules**

## **Cols Ordering**

\[
\text{Score}_i = 1000 \cdot P + 100 \cdot \log(1 + |\text{Obj}_i|) + 10 \cdot \sum{(\log(1 + |\delta_j|)}  + 10 \cdot \#occurrences
\]

#### **Components**
1. **Type Priority (\( P \)):**
   Assign priorities to variable types:
   - **Binary Variables (\( VType = \text{Binary} \)):** \( P = 3 \)
   - **Integer Variables (\( VType = \text{Integer} \)):** \( P = 2 \)
   - **Continuous Variables (\( VType = \text{Continuous} \)):** \( P = 1 \)

2. **Objective Coefficient (\( |\text{Obj}_i| \)):**
   Variables with larger absolute contributions to the objective function should have a higher score.

## **Rows Ordering**

\[
\text{Score}_j = 1000 \cdot P + 100 \cdot \log(1 + |\text{RHS}_j|) + 10 \cdot \sum{(\log(1 + |\gamma_j|)} + 1 \cdot \log(1 + |\text{Range}_j|) 
\]

#### **Components**
1. **Type Priority (\( P \)):**
   Assign priorities to variable types:
   - **> :** \( P = 3 \)
   - **= :** \( P = 2 \)
   - **< :** \( P = 1 \)

2. **RHS (\( |\text{RHS}_j| \)):**
   RHS variables with larger absolute contribution should have a higher score.
3. **Row Coefficient (\( |\gamma_j| \)):**
   Row variables with larger absolute contributions should have a higher score.

----
Because the approach focuses on relative differences and the structure of constraints rather than on absolute coefficient sizes, it tends to suit problems where the decision variable roles are defined by their interaction with many constraints and where relative cost or benefit differences are crucial. Examples include:

- **Assignment and Matching Problems:**  
  In these problems, the decision to assign a task (or match an element) is driven by cost or profit differences. Here, a variable’s importance comes more from how its objective coefficient compares to others rather than its absolute value.

- **Knapsack and Set Covering/Packing Problems:**  
  For knapsack problems, it’s the profit-to-weight ratio (a relative measure) that drives the optimal selection, not the raw profit numbers. Similarly, in set covering/packing, the “importance” of a column (or variable) is determined by how frequently it appears and how its cost compares with others.

- **Network Design and Facility Location:**  
  In these models, although the absolute sizes of capacities and costs can vary widely, it’s the relative differences (e.g., the ratio of capacity to demand, or the cost difference between opening and not opening a facility) that typically determine which variables are most critical.

- **Production Planning/Scheduling Problems:**  
  Such problems often have constraints with widely varying RHS values. Your row score, which includes log(1+|RHS|) and additional measures for coefficient variation (the γ and Range terms), would capture the “tightness” or variability of these constraints—a factor that is often more important than the absolute magnitudes.