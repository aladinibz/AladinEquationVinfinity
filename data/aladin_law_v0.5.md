# ALADIN Plasma Stability Law v0.5 — Final Foundation

**Invariant QMHD Threshold for Self-Stabilization in Current-Driven Plasma Columns**  
Mihai Alexandru Bucurenciu (Aladin) — January 21, 2026

### Core Law

A current-driven plasma column with uniform axial current density **J₀** exhibits self-stabilization against sausage (m=0) and kink (m=1) instabilities when

**Π = μ₀ J₀² r² / (ρ_eff c²) ≥ 8**

where  
- ρ_eff = total effective inertial density (classical + relativistic + QMHD contributions)  
- Π = 8 is the relativistic Alfvén causality limit (classical v_A → √2 c signals breakdown; physical v_A^{rel} ≤ c/√2)

### Mathematical Origin of Π = 8

From Ampère's law: B_θ = μ₀ J₀ a / 2  
Magnetic pressure: P_mag = μ₀ J₀² a² / 8  
Balance condition: P_mag = ρ_eff c²  
→ ρ_eff = μ₀ J₀² a² / (8 c²)  
→ Π = μ₀ J₀² a² / (ρ_eff c²) = 8

### Falsification

The law is falsified if Π(t) < 4 persists without stabilization for more than 10 Alfvén crossing times (τ > 10 a / v_A) or shows sustained exponential growth consistent with ideal-MHD rates.

### Regime of Validity

Ideal/low-resistivity MHD, β ≲ 1, approximate axial symmetry, resistive diffusion time ≫ instability growth time, subdominant reconnection during collapse phase. Strongly turbulent or reconnection-dominated regimes excluded from core law.

### Scope & Extensions

This is the frozen core law. All extensions (relativistic flow, Kerr-like GR, multi-mode tracking, turbulence damping, reconnection feedback, non-uniform J(r)) are model demonstrations that renormalize ρ_eff(t) but do **not** alter the Π ≥ 8 threshold.

### Files in this repo

- `aladin_law_v0.5.py` — full simulation code  
- `plots/aladin_law_v0.5_final.png` — collapse & stabilization plot  
- `docs/aladin_plasma_stability_law_v0.5.pdf` — 1-page law statement  

### License

MIT

### Zenodo DOI

(coming soon — v0.5 upload in progress)

We built this from zero.  
Single input. Emergent physics. No cheats.  
The law stands.

Love the grind.  
— Aladin
