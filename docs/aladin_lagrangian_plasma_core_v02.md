# ALADIN Plasma Stability Criterion v0.2

**Single Measured Input**  
J₀ = 1.000 × 10¹⁸ A/m² (primordial axial current density)

**Core Principle**  
A plasma column with uniform axial current density J₀ generates azimuthal magnetic field B_θ(r) = μ₀ J₀ r / 2 and inward pinch force F_r = - μ₀ J₀² r / 2.

**Derived Quantities**  
- Effective density ρ_eff = μ₀ J₀² a² / (8 c²)  
  (magnetic pressure = relativistic rest-mass energy density balance)  
- Alfvén speed v_A = c / √2  
  (at the balance point)  
- Heuristic normalized growth rate \tilde{γ} ≈ √(Π / 4)  
  (long-wavelength instability scaling)

**Dimensionless Strength**  
Π = μ₀ J₀² a² / (ρ c²)  
When Π ≳ 8 (at balance point), magnetic domination enables potential self-stabilization.

**Plots**  
- **aladin_lagrangian_zpinch_dispersion.png** — Sausage & kink growth rates  
- **aladin_lagrangian_stability_phase_diagram.png** — Π vs a → f & \tilde{f}

**Status**  
Core frozen: J₀ → Π ≳ 8 → stable regime possible.  
All further mechanisms (shear flow, B_z ramp, relativistic effects, biological mappings) are derived consequences and do not alter the core threshold.

**Author**: Mihai Alexandru Bucurenciu (Aladin)  
**Frozen**: January 19, 2026

**License**: MIT  
**Repository**: [link to your GitHub repo]
