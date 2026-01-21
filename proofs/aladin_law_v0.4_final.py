"""
ALADIN Plasma Stability Law v0.4-final — Pure J₀ Z-Pinch Core
Single measured input: J₀ = 1.000 × 10¹⁸ A/m²
Derived: ρ_eff(t) compressive + QMHD pairs & pressure
Pure force-balance collapse — stabilization emerges naturally when Π(t) ≥ 8
Frozen core — January 20, 2026 — Mihai Alexandru Bucurenciu (Aladin)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots/aladin_law_v0.4_final', exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────
J0 = 1.000e18
mu0 = 4 * np.pi * 1e-7
c = 3e8
a_initial = 1.0
rho_0 = 1.745e12

# QED constants
m_e = 9.109e-31
e = 1.602e-19
hbar = 1.0545718e-34
B_c = 4.4e9
E_c = B_c * c
alpha_em = 1 / 137.036

# Numerical safety
EPS = 1e-20
RHO_MAX = 1e25
NPAIR_MAX = 1e40
V_MAX = 1e12

# Time grid (high resolution)
t = np.linspace(0, 60, 10000)
dt = t[1] - t[0]

# Arrays
r = np.ones_like(t) * a_initial
v_r = np.zeros_like(t)
B_z = np.zeros_like(t)
n_pair = np.zeros_like(t)
E_z = np.zeros_like(t)
Pi_t = np.zeros_like(t)
v_A_t = np.zeros_like(t)
acc_t = np.zeros_like(t)

t_stable = None

B_theta = (mu0 * J0 * a_initial) / 2

# ─── Time-Dependent Collapse Loop ──────────────────────────────────────────
for i in range(1, len(t)):
    compression = a_initial / max(r[i-1], EPS)
    
    B_theta_r = (mu0 * J0 * r[i-1]) / 2
    
    dB_z_dt = (B_z[i-1] - B_z[i-2]) / dt if i > 1 else 0.0
    E_z[i] = - (r[i-1] / 2.0) * dB_z_dt
    
    Gamma_pair = 0.0
    if abs(E_z[i]) > 1e10:
        Gamma_pair = (alpha_em * abs(E_z[i]) * E_c / hbar) * \
                     (B_theta_r / B_c)**2 * \
                     np.exp(-np.pi * E_c / abs(E_z[i]))
    
    n_pair_new = n_pair[i-1] + Gamma_pair * dt
    max_pairs = (B_theta_r**2 / (2 * mu0)) / (2 * m_e * c**2) if B_theta_r > 0 else 0.0
    n_pair[i] = min(n_pair_new, max_pairs, NPAIR_MAX)
    
    P_q = 0.0
    if n_pair[i] > 0:
        P_q = (3.0/5.0) * n_pair[i] * (3.0 * np.pi**2 * n_pair[i])**(2.0/3.0) * hbar**2 / m_e
    
    rho_eff = rho_0 * compression**2 + 2 * m_e * n_pair[i] + P_q / c**2
    rho_eff = min(rho_eff, RHO_MAX)
    
    v_A = B_theta_r / np.sqrt(mu0 * rho_eff + EPS)
    v_A_t[i] = v_A
    
    Pi_t[i] = mu0 * J0**2 * r[i-1]**2 / (rho_eff * c**2) if rho_eff > 0 else 0.0
    
    # Pure force balance acceleration (no suppression multiplier)
    acc_r = - (mu0 * J0**2 * r[i-1]) / (2 * rho_eff) if rho_eff > 0 else 0.0
    acc_t[i] = acc_r
    
    v_r[i] = v_r[i-1] + acc_r * dt
    v_r[i] = np.clip(v_r[i], -V_MAX, 0.0)
    
    r[i] = r[i-1] + v_r[i] * dt
    r[i] = max(r[i], 1e-6)
    
    B_z[i] = B_z[i-1] * compression**2 * (1.0 + 0.2 * 0.01)
    S[i] = 0.1 * v_A / a_initial * compression
    
    if t_stable is None and abs(acc_r) < 1e-5 and Pi_t[i] >= 8.0:
        t_stable = t[i]

# ─── Final Stability Plot ──────────────────────────────────────────────────
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axs[0].plot(t, Pi_t, 'purple', lw=2, label='Π(t)')
axs[0].axhline(8.0, color='gray', ls='--', label='Threshold Π = 8')
if t_stable is not None:
    axs[0].axvline(t_stable, color='green', ls='--', lw=3, label=f'Stabilizes at t = {t_stable:.2f} s')
axs[0].set_ylabel('Π(t)')
axs[0].set_yscale('log' if np.any(Pi_t > 0) else 'linear')
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(t, r, 'blue', lw=2, label='r(t) — collapse')
axs[1].set_ylabel('r (normalized)')
axs[1].grid(alpha=0.3)

axs[2].plot(t, n_pair, 'red', lw=2, label='n_pair(t)')
axs[2].set_ylabel('Pair density (m⁻³)')
axs[2].set_yscale('log')
axs[2].legend()
axs[2].grid(alpha=0.3)

axs[3].plot(t, v_A_t / c, 'orange', lw=2, label='v_A(t) / c')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('v_A / c')
axs[3].legend()
axs[3].grid(alpha=0.3)

plt.suptitle('ALADIN Law v0.4-final – Emergent QMHD Stabilization')
plt.tight_layout()
plt.savefig('plots/aladin_law_v0.4_final_collapse.png', dpi=300)
plt.close()

print("Plot saved: plots/aladin_law_v0.4_final_collapse.png")
print(f"Natural stabilization time: {t_stable:.2f} s" if t_stable is not None else "No stabilization reached")
print(f"Max Π(t): {np.max(Pi_t):.2e}")
print(f"Final r: {r[-1]:.2e} m")
print(f"Max pair density: {np.max(n_pair):.2e} m⁻³")
