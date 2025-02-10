# Variability Metrics in Optimization Experiments

This document explains the mathematical formulation used to measure variability in:
1. **Solve-time variability**: Measures the spread of solve times across permutations.
2. **Permutation distance variability**: Measures the spread of permutation distances before and after canonicalization.

Both metrics use **standard deviation (σ)** to quantify the dispersion of values in their respective sets.

---

## Solve-Time Variability

Solve-time variability captures how much solve times fluctuate across different **permutations of the same problem** and how much **canonicalization** stabilizes this variability.

### **Mathematical Definition**
Let:
- \( t_{\text{orig}} \) be the solve time of the **original** problem.
- \( t_{\text{perm}, i} \) be the solve time of the \( i \)-th **permuted** problem.
- \( t_{\text{canon-orig}} \) be the solve time of the **canonicalized form of the original** problem.
- \( t_{\text{canon-perm}, i} \) be the solve time of the **canonicalized form of the \( i \)-th permuted problem**.
- \( N \) be the number of permutations.

The standard deviation of **solve times across permutations (including the original)** is given by:

\[
\sigma_{\text{perm}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (t_{\text{perm}, i} - \bar{t}_{\text{perm}})^2}
\]

where:

\[
\bar{t}_{\text{perm}} = \frac{1}{N+1} \left( t_{\text{orig}} + \sum_{i=1}^{N} t_{\text{perm}, i} \right)
\]

Similarly, the standard deviation of **canonicalized solve times** is:

\[
\sigma_{\text{canon}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (t_{\text{canon-perm}, i} - \bar{t}_{\text{canon}})^2}
\]

where:

\[
\bar{t}_{\text{canon}} = \frac{1}{N+1} \left( t_{\text{canon-orig}} + \sum_{i=1}^{N} t_{\text{canon-perm}, i} \right)
\]

### **Interpretation**
- **Higher \(\sigma_{\text{perm}}\)** → High variability in solve times across permutations.
- **Higher \(\sigma_{\text{canon}}\)** → High variability even after canonicalization.
- **If \(\sigma_{\text{canon}} < \sigma_{\text{perm}}\)** → Canonicalization improves stability.

---

## Permutation Distance Variability

Permutation distance variability measures how much the **structural differences** (measured by a **permutation distance**) change across permutations.

### **Mathematical Definition**
Let:
- \( d_{\text{perm}, i} \) be the **permutation distance** between the **original and permuted problem**.
- \( d_{\text{canon}, i} \) be the **permutation distance** between the **canonical form of the original and the canonical form of the permuted problem**.
- \( N \) be the number of permutations.

The standard deviation of **permutation distances before canonicalization** is:

\[
\sigma_{\text{perm-dist}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (d_{\text{perm}, i} - \bar{d}_{\text{perm}})^2}
\]

where:

\[
\bar{d}_{\text{perm}} = \frac{1}{N} \sum_{i=1}^{N} d_{\text{perm}, i}
\]

Similarly, the standard deviation of **permutation distances after canonicalization** is:

\[
\sigma_{\text{canon-dist}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (d_{\text{canon}, i} - \bar{d}_{\text{canon}})^2}
\]

where:

\[
\bar{d}_{\text{canon}} = \frac{1}{N} \sum_{i=1}^{N} d_{\text{canon}, i}
\]

### **Interpretation**
- **Higher \(\sigma_{\text{perm-dist}}\)** → Large variability in permutation distances across different problem permutations.
- **Higher \(\sigma_{\text{canon-dist}}\)** → Canonicalization does not effectively eliminate permutation differences.
- **If \(\sigma_{\text{canon-dist}} < \sigma_{\text{perm-dist}}\)** → Canonicalization reduces structural variation.

---

## Summary of Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Solve-Time Variability (Original + Permuted)** | \( \sigma_{\text{perm}} \) | Measures fluctuation in solve times across problem permutations |
| **Solve-Time Variability (Canonical Forms)** | \( \sigma_{\text{canon}} \) | Measures fluctuation in solve times of canonicalized problems |
| **Permutation Distance Variability (Before Canonicalization)** | \( \sigma_{\text{perm-dist}} \) | Measures how much the problem's structure changes across permutations |
| **Permutation Distance Variability (After Canonicalization)** | \( \sigma_{\text{canon-dist}} \) | Measures if canonicalization is making structures more consistent |

---
