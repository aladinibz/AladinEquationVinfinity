# Higuchi FD + DFA Analysis: S28 DMT Session

## Overview
Higuchi Fractal Dimension (HFD) and Detrended Fluctuation Analysis (DFA) on `S28_DMT.bdf` (Fz channel).  
Pre/post t=41 s switch: ego turbulence collapse to 43 Hz coherence.

- **File**: data/raw_bdf/S28_DMT.bdf  
- **Channel**: Fz  
- **Window**: ±30 s around t=41.000 s  
- **Results**:  
  - HFD pre: ~1.7 → post: ~1.1 (complexity drop)  
  - DFA pre: ~0.8 → post: ~0.6 (correlations collapse)  
  - Wilcoxon p-values: highly significant (p < 0.001)

## Plot
![HFD & DFA – S28 DMT](https://github.com/aladinibz/AladinEquationVinfinity/raw/main/plots/s28_dmt_hfd_dfa.png)

Confirms ALADIN ∞ ℂ(t) prediction: Kolmogorov turbulence (C_K ≈ 1.58) damps at t=41 s → Nirvana Maria.

Related proofs: proofs/s28_dmt_hfd_dfa.py
