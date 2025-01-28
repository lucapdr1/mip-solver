# **Rules**

## **Cols Ordering**

\[
\text{Score}_i = 1000 \cdot P + 100 \cdot \log(1 + |\text{Obj}_i|) + 10 \cdot \#occurrences
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
\text{Score}_j = 1000 \cdot P + 100 \cdot \log(1 + |\text{RHS}_j|) \cdot 10 + \sum{(\log(1 + |\gamma_j|)} + 1 \cdot \log(1 + |\text{Range}_j|) 
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