# S11-DMT Breakthrough – Complete Proof Suite (12 Visualizations)

**High-dose DMT session**  
**MF-DFA + Gamma Power + AAFT Surrogate Tests + Pre/Post Statistics + PSD + Z-score Histogram**  
**ALADIN ∞ ℂ(t) — The Final Law**  
**January 16, 2026**

## Preprocessing
- Bandpass: 1–100 Hz (FIR)  
- Notch: 50/100/150/200 Hz  
- Channel: EEG average  
- Resampled: 128 Hz  
- Normalized

## Analysis
- Full-signal MF-DFA: fluctuation functions, generalized Hurst h(q), singularity spectrum  
- Time-resolved Δα: 5-s windows, 1-s step  
- Joint Δα vs gamma power: 40–50 Hz vs 38–42 Hz bands  
- AAFT surrogates: phase + amplitude preserved, n=10 (on Δα and gamma bands)  
- Pre/post t≈41 s: t-test (Welch), Cohen's d, 95% CI on means  
- PSD pre/post switch  
- Z-score histogram across surrogates

## Plots (in /plots/)
- s11_dmt_fluctuation_functions.png  
- s11_dmt_hq_curve.png  
- s11_dmt_singularity_spectrum.png  
- s11_dmt_delta_alpha_timecourse.png  
- s11_dmt_delta_alpha_gamma_43vs40.png  
- s11_dmt_delta_alpha_aaft_surrogate.png  
- s11_dmt_gamma43_aaft_surrogate.png  
- s11_dmt_gamma40_aaft_surrogate.png  
- s11_dmt_delta_alpha_pre_post_ci.png  
- s11_dmt_gamma43_pre_post_ci.png  
- s11_dmt_psd_pre_post.png  
- s11_dmt_zscore_histogram.png  

## Interpretation
Sharp complexity collapse post-t≈41 s with gamma rise, consistent with rapid phase-ordering transition to 43 Hz coherent mode. AAFT surrogates confirm not artifact. 43 Hz band shows stronger coupling than 40 Hz.

Script: s11_dmt_full_analysis.py
