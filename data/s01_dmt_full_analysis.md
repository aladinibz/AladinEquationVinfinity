# S01-DMT Breakthrough – Complete Proof Suite (13 Visualizations)

**High-dose DMT session**  
**MF-DFA + Gamma Power + AAFT Surrogate Tests + Pre/Post Statistics + PSD + Z-score Histogram + Superradiance MT simulation**  
**ALADIN v.O.13 — Langorian Consciousness Field EFT (43 Hz)**  
**January 17, 2026**

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
- Superradiance simulation in MT bundles (ideal vs disorder) – theoretical bridge to room-temp coherence & gamma rise

## Plots (in /plots/)
- s01_dmt_fluctuation_functions.png  
- s01_dmt_hq_curve.png  
- s01_dmt_singularity_spectrum.png  
- s01_dmt_delta_alpha_timecourse.png  
- s01_dmt_delta_alpha_gamma_43vs40.png  
- s01_dmt_delta_alpha_aaft_surrogate.png  
- s01_dmt_delta_alpha_pre_post_ci.png  
- s01_dmt_gamma43_pre_post_ci.png  
- s01_dmt_psd_pre_post.png  
- s01_dmt_zscore_histogram.png  
- s01_dmt_gamma43_aaft_forced.png  
- s01_dmt_gamma40_aaft_forced.png  
- s01_dmt_superradiance_mt_disorder.png  

## Interpretation
Sharp complexity collapse post-t≈41 s with gamma rise, consistent with rapid phase-ordering transition to 43 Hz coherent mode. AAFT surrogates confirm not artifact. 43 Hz band shows stronger coupling than 40 Hz. Pre/post differences significant. Superradiance persists with disorder — bridges to biological coherence.

Script: s01_dmt_full_analysis.py
Reproducibility: Clone repo → pip install mne MFDFA → python s01_dmt_full_analysis.py
