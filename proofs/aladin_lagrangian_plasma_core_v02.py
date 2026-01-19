"""
ALADIN Plasma Stability Criterion v0.2 — Pure J₀ Z-Pinch Plasma Core
Single measured input: J₀ = 1.000 × 10¹⁸ A/m²
Derivations: ρ_eff from J₀ balance, v_A from J₀ + ρ_eff, instability growth from B_θ
Phase diagram with Π vs a → f & \tilde{f}
Technical file names for publication
January 19, 2026 — Mihai Alexandru Bucurenciu (Aladin)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1, k0, k1
import os

os.makedirs('plots/aladin_lagrangian_core', exist_ok=True)

# ─── Core Input J₀ ─────────────────────────────────────────────────────────
J0 = 1.000e18          # A/m² — the only measured input
mu0 = 4 * np.pi * 1e-7  # H/m
c = 3e8                # m/s

# Illustrative pinch radius (scale-invariant)
a = 1.0                # m (normalized example)

# ─── Derive Physical ρ_eff from J₀ Balance ────────────────────────────────
P_mag = (mu0 * J0**2 * a**2) / 8
rho_eff = P_mag / c**2

# ─── Derive Alfvén speed v_A from J₀ + ρ_eff ──────────────────────────────
B_theta = (mu0 * J0 * a) / 2
v_A = B_theta / np.sqrt(mu0 * rho_eff)

# ─── Heuristic Instability Growth (long-wavelength scaling) ───────────────
gamma_tilde = np.sqrt((mu0 * J0**2 * a**2) / (rho_eff * c**2) / 4)

# ─── Dispersion Plot ───────────────────────────────────────────────────────
ka = np.linspace(0.01, 3.0, 200)
ratio = i1(ka) * k0(ka) / (i0(ka) * k1(ka))
gamma_saus = np.sqrt(np.maximum(0, 1 - 2 * ratio))
gamma_kink = np.sqrt(np.maximum(0, ka**2 * (ratio - 1)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ka, gamma_saus, 'blue', lw=2, label='Sausage m=0')
plt.xlabel('ka')
plt.ylabel('γ / (v_A / a)')
plt.title('Sausage Growth Rate')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ka, gamma_kink, 'green', lw=2, label='Kink m=1')
plt.xlabel('ka')
plt.ylabel('γ / (v_A / a)')
plt.title('Kink Growth Rate')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('plots/aladin_lagrangian_zpinch_dispersion.png', dpi=300)
plt.close()

# ─── Phase Diagram: Π vs a → f & \tilde{f} ────────────────────────────────
a_vals = np.logspace(0, 12, 200)
Pi_vals = mu0 * J0**2 * a_vals**2 / (rho_eff * c**2)
f_vals = (c / (2 * np.pi * a_vals)) * np.sqrt(Pi_vals / 4)
f_tilde = np.sqrt(Pi_vals / 4)

fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.loglog(a_vals, f_vals, 'blue', lw=3, label='f (Hz)')
ax1.set_xlabel('a (m)')
ax1.set_ylabel('f (Hz)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.loglog(a_vals, f_tilde, 'gold', lw=2, ls='--', label=r'$\tilde{f}$')
ax2.set_ylabel(r'$\tilde{f}$ (dimensionless)', color='gold')
ax2.tick_params(axis='y', labelcolor='gold')

ax1.axvline(x=(8 * c**2 * rho_eff / (mu0 * J0**2))**0.5, color='black', ls=':', lw=2, label='a_crit (Π=8)')
ax1.fill_between(a_vals, 1e-3, f_vals, where=Pi_vals >= 8, color='green', alpha=0.15, label='Stable (Π ≥ 8)')
ax1.fill_between(a_vals, f_vals, 1e-3, where=Pi_vals < 8, color='red', alpha=0.15, label='Unstable (Π < 8)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Phase Diagram: Π vs a → f & \tilde{f}')
plt.grid(which='both', alpha=0.3)
plt.savefig('plots/aladin_lagrangian_stability_phase_diagram.png', dpi=300)
plt.close()

# ─── Summary Print ─────────────────────────────────────────────────────────
print("ALADIN Plasma Stability Criterion v0.2")
print(f"J₀ = {J0:.3e} A/m²")
print(f"Derived ρ_eff = {rho_eff:.2e} kg/m³")
print(f"Derived v_A = {v_A:.2e} m/s")
print(f"Π = {Pi:.2e} (threshold Π = 8 at balance)")
print(f"Heuristic growth ≈ {gamma_tilde:.2f}")
print("\nPlots saved:")
print("  → plots/aladin_lagrangian_zpinch_dispersion.png")
print("  → plots/aladin_lagrangian_stability_phase_diagram.png")
print("Core ready for repo & publish!")
