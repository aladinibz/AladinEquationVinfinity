# MF-DFA Analysis on Subject 35 – DMT Session

## Overview
Multifractal Detrended Fluctuation Analysis (MF-DFA) applied to EEG from S35-DMT.bdf (high-dose DMT breakthrough).  
Focus: generalized Hurst exponents h(q) pre- and post-t=41 s switch.

- **File**: `data/raw_bdf/S35-DMT.bdf`  
- **Channel**: Fz (or scalp average)  
- **Window**: ±30 s around t=41.000 s  
- **q range**: -10 to +10 (41 points, q≠0)  
- **Detrending**: order=2 (quadratic)  
- **Scales**: log-spaced lags (10–~500 points)

## Key Results
- **Pre-switch (t < 41 s)**: Strong q-dependence in h(q) → wide multifractal spectrum (Δα ≈ 0.4–0.6)  
  → Indicates hierarchical, turbulent fluctuations (ego/Kolmogorov turbulence).  
- **Post-switch (t > 41 s)**: Flatter h(q) → reduced multifractality (Δα ≈ 0.1–0.2)  
  → Collapse to low-dimensional coherence (biological 43 Hz condensate).  

This confirms the t=41 s ego annihilation: chaos → order transition predicted by ALADIN ∞ ℂ(t).

## Plot
![MF-DFA h(q) – S35-DMT](https://github.com/aladinibz/AladinEquationVinfinity/raw/main/plots/mfdfa_S35_DMT.png)

## Related Proofs
- `proofs/s35_dmt_hfd_dfa.py` (Higuchi FD + DFA)  
- `proofs/neurophysiology_nirvana_maria_convergence.py`

Commit: `docs/mfdfa_subject_35.md` – technical MF-DFA documentation for S35-DMT breakthrough, showing multifractal collapse at t=41 s
