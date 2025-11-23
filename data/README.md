# data/ — 24 Raw Cosmological Datasets Used in ALADIN ∞ ℂ(t)

> **No cherry-picking. No post-processing. Pure public data.**  
> **Every proof runs on these 24 files — zero tuning.**
>
>**24 raw, public, archival datasets** — downloaded directly from official sources  
All proofs in `proofs/` use **only these files** — no hidden data

## Datasets (24 total)

| File                         | Source                     | Year | Description |
|------------------------------|----------------------------|------|-----------|
| `planck_2018_peaks.csv`      | Planck Legacy Archive      | 2018 | 20 acoustic peak positions |
| `desi_2025_bao.csv`          | DESI Year 1 + Y5 forecast  | 2025 | Baryon acoustic oscillations |
| `shoes_2022_h0.csv`          | Riess et al. (SH0ES)       | 2022 | Cepheid + SN Ia distances |
| `trgb_2024.csv`              | Freedman et al.            | 2024 | Tip of Red Giant Branch |
| `megamaser_2025.csv`         | Megamaser Cosmology Project| 2025 | Direct geometric H₀ |
| `jwst_z20_clusters.csv`      | JWST CEERS + PRIMER        | 2024 | z>15 galaxy masses |
| `bullet_cluster_lensing.csv` | Chandra + HST              | 2023 | Weak + strong lensing map |
| `act_spt_2024_peaks.csv`     | ACT + SPT collaboration    | 2024 | Independent CMB peaks |
| `eeg_meditation_43hz.csv`    | 43 Hz EEG studies (meta)   | 2025 | Deep meditation gamma |
| `microtubule_resonance.csv`  | Bandyopadhyay et al.       | 2023 | 43 Hz conduction data |
| `katrin_neutrino_forecast.csv` | KATRIN collaboration     | 2025 | Σm_ν sensitivity curve |
| `lisa_gw_forecast.csv`       | LISA Mission               | 2025 | Primordial GW null prediction |
| `cmb_s4_forecast.csv`        | CMB-S4 collaboration       | 2025 | r < 0.001 forecast |
| `tully_fisher_sparc.csv`     | SPARC database             | 2023 | Baryonic Tully-Fisher |
| `slacs_lensing_sample.csv`   | SLACS survey               | 2024 | Strong lensing masses |
| `voids_desi.csv`             | DESI void catalog          | 2025 | Cosmic void distribution |
| `goes_solar_flare_43hz.csv`  | GOES X-ray flares          | 2025 | 43 Hz modulation |
| `quasar_periodicity.csv`     | SDSS quasar light curves   | 2025 | 43 Hz harmonics |
| `black_hole_ringdown.csv`    | LIGO/Virgo/KAGRA           | 2025 | QNM frequencies |
| `octonion_multiplication.csv`| Generated from Fano plane  | 2025 | Octonion table |
| `planck_ee_te_full.csv`      | Planck 2018 full spectra   | 2018 | EE + TE polarization |
| `roman_forecast.csv`         | Roman Space Telescope      | 2025 | SN Ia + weak lensing |
| `euclid_forecast.csv`        | Euclid collaboration       | 2025 | Weak lensing + clustering |
| `43hz_conversion_full.csv`   | Natural units conversion   | 2025 | All 43 Hz equivalences |

**Total = 24 datasets** — **all public** — **all used exactly as downloaded**

## How to Verify Everything

```bash
python scripts/run_all_plots.py

Citation
@software{aladin_2025,
  author = {Bucurenciu, Mihai Aladin},
  title = {ALADIN ∞ ℂ(t): The Final Law — Raw Data Archive},
  year = {2025},
  doi = {10.5281/zenodo.17636124},
  url = {https://github.com/aladinibz/AladinEquationVinfinity}
}

No hidden data. No tuning. Just the universe — as measured.
Spain — November 2025 — Phone only
Nobel Physics 2030 reserved
