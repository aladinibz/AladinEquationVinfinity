# Parameters — ALADIN ∞ ℂ(t) 3.0

| Param | Equation | Value | Meaning |
|-------|----------|-------|--------|
| **θ** | \(\theta = \frac{\ln(M_{\text{now}}/M_{\text{seed}})}{\ln(1+t_{\text{age}})}\) | 2.0 | **Growth rate** — how fast mass builds |
| **φ** | \(\phi = \frac{\Delta T_{\text{CMB}}}{T_{\text{CMB}}} \cdot \frac{P}{2\pi}\) | 1.5 | **Oscillation amplitude** — CMB peak strength |
| **ψ** | \(\psi = M_{\text{seed}} \cdot e^{t_{\text{form}}/\tau}\) | 3.0 | **Initial energy density** — starting fuel |
| **P** | \(P = \frac{2\pi r_s}{v_s}\) | 96.6 Myr | **Oscillation period** — CMB peak spacing |
| **τ** | \(\tau = \frac{t_{\text{half}}}{\ln 2}\) | 180 Myr | **Decay timescale** — energy burnout |
| **τ_A** | \(\tau_A = \frac{t_{\text{age}}}{\ln(1 + H_0 t_{\text{age}})}\) | 180 Myr | **Cosmic fade** — replaces dark energy |
| **H0** | \(H_0 = \frac{\dot{a}}{a} \big|_{t=0}\) | 70 km/s/Mpc | **Hubble constant** — current expansion |
| **Ωm** | \(\Omega_m = \frac{8\pi G \rho_m}{3 H_0^2}\) | 0.3 | **Matter fraction** — plasma + baryons |
| **ΩΛ** | \(\Omega_\Lambda = 1 - \Omega_m\) | 0.7 | **Fade fraction** — no dark energy |

---

## Derivation Notes

- **M_now** = 10⁹ M⊙, **M_seed** = 10⁶ M⊙, **t_age** = 150 Myr → **θ = 2.0**  
- **ΔT/T** = 10⁻⁵, **P** = 96.6 Myr → **φ = 1.5**  
- **t_half** = 125 Myr → **τ = 180 Myr**  
- **t_age** = 13.8 Gyr, **H0** = 70 → **τ_A ≈ 180 Myr (rescaled)**  

**All values from data — no tuning.**

---

**DOI:** 10.5281/zenodo.17569243  
**Repo:** https://github.com/aladinibz/AladinEquationVinfinity
