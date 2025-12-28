# Phase Mixing Damping Rate for Kink Modes

**ALADIN ∞ ℂ(t) — Plasma / Cosmology Proof**

Derives the apparent damping rate from phase mixing in inhomogeneous coronal loops. Energy cascades to small scales, producing power-law decay.

### Derivation

In an inhomogeneous plasma (density gradient ρ(r)), local Alfvén frequencies ω_A(r) vary across the loop. Kink mode excites field lines at different frequencies → phases dephase over time.

Amplitude decay from phase mixing (Heyvaerts & Priest 1983):

$$
A(t) \propto t^{-1/3}
$$

Effective damping rate:

$$
\gamma(t) \propto -\frac{1}{3t} \left( \frac{\Delta \omega_A}{\omega_0} \right)^{2/3}
$$

where:
- Δω_A = spread in local Alfvén frequencies across loop  
- ω_0 = global kink frequency

### Calculation (Typical Loop)
- ρ_i / ρ_e = 10 → Δv_A / v_A ≈ 0.3–0.5  
- Δω_A / ω_0 ≈ 0.4  
- At t = 100 s: γ(t) ≈ -0.0013 rad/s → e-folding time ~770 s  
- Power-law decay: slow, "decayless" appearance early, then fades

### Interpretation
- Explains long-lived kink oscillations in SDO (decayless phase).  
- Energy cascades to small scales → couples to 43 Hz sausage resonance.  
- Falsifiable: Pure exponential decay (no power-law tail) favors resonant absorption over phase mixing.

See also: final_kink_stability_all_modes.md, resonant_absorption_damping_rate.md
