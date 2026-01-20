# ALADIN Plasma Stability Law v0.4

**Invariant Threshold for Self-Stabilization in Current-Driven Plasma Columns**  
**Mihai Alexandru Bucurenciu (Aladin) – January 20, 2026**

### Core Law (frozen)

\boxed{
\textbf{ALADIN Stability Law}

A current-driven plasma column with uniform axial current density J_0 is stable against sausage (m=0) and kink (m=1) instabilities if and only if

\max\!\left[
\left(\dfrac{B_z}{B_\theta}\right)^2,\;
\dfrac{S}{\gamma_m}
\right]
\ge 1
}

**Exact definitions**

- B_\theta = \dfrac{\mu_0 J_0 a}{2}
- \gamma_m = \dfrac{v_A}{a}
- v_A = \dfrac{B_\theta}{\sqrt{\mu_0 \rho_{\rm eff}}}
- \rho_{\rm eff} = \dfrac{\mu_0 J_0^2 a^2}{8 c^2} \quad (\Pi = 8 \text{ at balance point})

**Falsification**

The law is false if \Pi < 4 and no suppression occurs (\gamma a / v_A > 0.5 or q < 1 in observed dynamics).

**Regime of validity**

Ideal/low-resistivity MHD, β ≲ 1, approximate axial symmetry, resistive diffusion time ≫ instability growth time, subdominant reconnection during collapse phase.

**Scope**

This is the frozen core law. All simulations (relativistic corrections, QMHD pair production, shear/B_z ramps) are illustrative demonstrations of how the system reaches the criterion.

**Key plot**  
- aladin_law_v0.4_stability_evolution.png — time evolution of Π(t), growth rates, pair density, v_A/c

**Zenodo DOI**  
[add after upload – new version of existing record]

**Repository**  
https://github.com/[your-username]/aladin

**License**  
MIT
