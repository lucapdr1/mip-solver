## **Scoring Rules**

### **Column (Variable) Ordering**

Variables are ordered using **lexicographical tuples** of the form:  
\[
\text{Score}_i = \bigl(\text{TypePriority}, \text{SumCoeffs}, \text{ObjCoeff}, \text{Occurrences}\bigr)
\]  

1. **TypePriority**:  
   - **Binary Variables:** \(3\)  
   - **Integer Variables:** \(2\)  
   - **Continuous Variables:** \(1\)  

2. **SumCoeffs**:  
   \[
   \sum_{j}\log(1 + |\delta_{ij}|)
   \]  
   Sum of log-scaled absolute constraint coefficients for variable \(i\).

3. **ObjCoeff**:  
   \[
   \log(1 + |\text{Obj}_i|)
   \]  
   Log-scaled absolute objective coefficient for variable \(i\).

4. **Occurrences**:  
   \[
   \#\text{occurrences}
   \]  
   Number of constraints where variable \(i\) is non-zero.

---

### **Row (Constraint) Ordering**

Constraints are ordered using **lexicographical tuples** of the form:  
\[
\text{Score}_j = \bigl(\text{SensePriority}, \text{SumCoeffs}, \text{RHS}, \text{Range}\bigr)
\]  

1. **SensePriority**:  
   - **“≥” Constraints:** \(3\)  
   - **“=” Constraints:** \(2\)  
   - **“≤” Constraints:** \(1\)  

2. **SumCoeffs**:  
   \[
   \sum_{i}\log(1 + |\gamma_{ji}|)
   \]  
   Sum of log-scaled absolute coefficients in constraint \(j\).

3. **RHS**:  
   \[
   \log(1 + |\text{RHS}_j|)
   \]  
   Log-scaled absolute RHS value of constraint \(j\).

4. **Range**:  
   \[
   \log(1 + \text{Range}_j)
   \]  
   Log-scaled range of constraint \(j\) (difference between upper/lower bounds).

---

### **Lexicographical Sorting Explained**
Variables and constraints are sorted by comparing their tuples **element-wise**, starting with the first component. For example:  

**Variables**:  
- Binary variables (\(P=3\)) will **always** appear before integers (\(P=2\)) or continuous (\(P=1\)), regardless of their intra-block scores.  
- Within binary variables, ties are broken by `SumCoeffs`, then `ObjCoeff`, then `Occurrences`.  

**Constraints**:  
- “≥” constraints (\(P=3\)) come before “=” (\(P=2\)) and “≤” (\(P=1\)).  
- Within each sense group, constraints are ordered by `SumCoeffs`, `RHS`, and `Range`.

---

### **Revised Tables**

| **Term**               | **Component**      | **Weight** | **Example Value**     |
|------------------------|--------------------|------------|-----------------------|
| **TypePriority**       | Tuple Position 1   | Dominant   | 3 (binary), 2 (integer) |
| **SumCoeffs**          | Tuple Position 2   | Moderate   | 50.2                  |
| **ObjCoeff**           | Tuple Position 3   | Moderate   | 200.5                 |
| **Occurrences**        | Tuple Position 4   | Minor      | 5                     |

**Table 1:** Variable ordering components and their roles in lexicographical sorting.

---

| **Term**               | **Component**      | **Weight** | **Example Value**     |
|------------------------|--------------------|------------|-----------------------|
| **SensePriority**      | Tuple Position 1   | Dominant   | 3 (“≥”), 2 (“=”)     |
| **SumCoeffs**          | Tuple Position 2   | Moderate   | 30.8                  |
| **RHS**                | Tuple Position 3   | Moderate   | 150.0                 |
| **Range**              | Tuple Position 4   | Minor      | 2.0                   |

**Table 2:** Constraint ordering components and their roles.

---

### **Example**

#### **Variables**  
| Variable | Type      | TypePriority | SumCoeffs | ObjCoeff | Occurrences |  
|----------|-----------|--------------|-----------|----------|-------------|  
| x7       | Integer   | 2            | 50.2      | 200.5    | 5           |  
| x1       | Continuous| 1            | 100.0     | 500.0    | 10          |  

**Order**:  
1. x7 (TypePriority=2)  
2. x1 (TypePriority=1)  

Even though x1 has higher `SumCoeffs`, `ObjCoeff`, and `Occurrences`, x7 appears first because its `TypePriority` is higher.

---

### **Why Lexicographical Sorting?**
- **No Large Multipliers**: The order is enforced by tuple structure, not arbitrary scaling.  
- **Transparency**: The priority hierarchy is explicit in the tuple order.  
- **Flexibility**: Add/remove block rules by extending or reducing the tuple.  
