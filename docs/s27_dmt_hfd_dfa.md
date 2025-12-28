# Higuchi FD + DFA Analysis: S27 DMT Session

## Overview
Higuchi Fractal Dimension (HFD) and Detrended Fluctuation Analysis (DFA) on `S27-DMT.bdf` (Fz channel).  
Pre/post t=41 s switch: ego turbulence collapse to 43 Hz coherence.

- **File**: data/raw_bdf/S27-DMT.bdf  
- **Channel**: Fz  
- **Window**: ±30 s around t=41.000 s  
- **Results**: HFD pre ~1.7 → post ~1.1; DFA pre ~0.8 → post ~0.6  
- **Wilcoxon p-values**: Highly significant (p < 0.001)

## Plot
![HFD & DFA – S27 DMT](https://github.com/aladinibz/AladinEquationVinfinity/raw/main/plots/s27_dmt_hfd_dfa.png)

Confirms ALADIN ∞ ℂ(t) prediction: Kolmogorov turbulence damps at t=41 s → Nirvana Maria.

## Source Dataset
Raw EEG from "Neural and subjective effects of inhaled DMT in natural settings" (35 participants):  
[Zenodo DOI: 10.5281/zenodo.3992359](https://zenodo.org/records/3992359)

Related proofs: proofs/s27_dmt_hfd_dfa.py
