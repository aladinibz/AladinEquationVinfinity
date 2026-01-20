"""
ALADIN Final Law v0.3-QMHD — Pure J₀ Z-Pinch Core with QMHD Effects
Single input: J₀ = 1.000 × 10¹⁸ A/m²
Derived: ρ_eff from J₀ balance, v_A from J₀ + ρ_eff, growth from B_θ
QMHD: Schwinger pair production with induced E_z, quantum pressure, ρ_eff_QMHD
Emergent B_z & shear suppression
All variables defined — no undefined errors
January 19, 2026 — Mihai Alexandru Bucurenciu (Aladin)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots/aladin_v03_qmhd', exist_ok=True)

# ─── 1. Core Input & Constants ─────────────────────────────────────────────
J0 = 1.000e18          # A/m² — only measured input
mu0 = 4 * np.pi * 1e-7  # H/m
c = 3e8                # m/s
a_initial = 1.0        # initial radius (m)
rho_eff_class = 1.745e12  # derived balance (kg/m³)

# QED constants
m_e = 9.109e-31        # electron mass (kg)
e = 1.602e-19          # electron charge (C)
hbar = 1.0545718e-34   # ħ (J s)
B_c = 4.4e9            # Schwinger critical field (T)

# Time grid
t = np.linspace(0, 60, 5000)
dt = t[1] - t[0]

# Surface toroidal field (constant)
B_theta = (mu0 * J0 * a_initial) / 2

r = np.ones_like(t) * a_initial
theta = np.zeros_like(t)
B_z = np.zeros_like(t)
S = np.zeros_like(t)
n_pair = np.zeros_like(t)  # pair density
E_z = np.zeros_like(t)     # induced E-field

t_suppress_bz = None
t_suppress_shear = None

for i in range(1, len(t)):
    compression = a_initial / r[i-1] if r[i-1] > 0 else 1.0
    
    # Local B at current r (B ∝ r for uniform J₀)
    B_current = B_theta * (r[i-1] / a_initial)
    
    # Induced E_z from Faraday law (dB_z/dt)
    if i > 1:
        dB_z_dt = (B_z[i-1] - B_z[i-2]) / dt
    else:
        dB_z_dt = 0  # no change at first step
    E_z[i] = - (r[i-1] / 2) * dB_z_dt
    
    # Schwinger pair production rate (crossed-field approximation)
    Gamma_pair = 0
    if B_current > B_c or abs(E_z[i]) > 1e16:  # safety threshold
        E_eff = np.sqrt(E_z[i]**2 + B_current**2)
        Gamma_pair = (1 / (3 * np.pi)) * (e * E_eff / (m_e * c))**2 * (c / (m_e * c / hbar)) * np.exp(-B_c / E_eff)
    n_pair[i] = n_pair[i-1] + Gamma_pair * dt  # cumulative (no annihilation for simplicity)
    
    # Quantum pressure P_q (degeneracy approximation)
    P_q = 0
    if n_pair[i] > 0:
        P_q = (3/5) * n_pair[i] * (3 * np.pi**2 * n_pair[i])**(2/3) * hbar**2 / m_e
    
    # QMHD effective density
    rho_eff_qmhd = rho_eff_class + 2 * m_e * n_pair[i] + P_q / c**2
    
    # QMHD Alfvén speed
    v_A_qmhd = B_theta / np.sqrt(mu0 * rho_eff_qmhd)
    
    # Emergent B_z seed from twist
    if B_z[i-1] == 0 and theta[i-1] > 0.01:
        B_z[i] = 0.01 * B_theta
    B_z[i] = B_z[i-1] * compression**2 * (1 + 0.02 * theta[i-1])
    B_z[i] = min(B_z[i], B_theta)
    
    # Emergent shear rate S ∝ 1/r
    S[i] = 0.1 * v_A_qmhd / a_initial * compression
    
    if t_suppress_bz is None and B_z[i] >= B_theta:
        t_suppress_bz = t[i]
    
    if t_suppress_shear is None and S[i] >= 0.5 * v_A_qmhd / a_initial:
        t_suppress_shear = t[i]
    
    stab_bz = (B_z[i] / B_theta)**2
    stab_shear = min(S[i] / (0.5 * v_A_qmhd / a_initial), 1.0)
    stab_factor = max(stab_bz, stab_shear)
    
    # QMHD growth rates (Landau suppression)
    n_L = e * B_current / (2 * np.pi * hbar)
    n_class = rho_eff_qmhd / (1.67e-27)  # approximate n = ρ / m_p
    landau_factor = np.sqrt(1 + n_L / n_class) if n_class > 0 else 1.0
    gamma_s_eff = 0.5 * v_A_qmhd / a_initial * (1 - stab_factor) / landau_factor
    gamma_k_eff = 0.3 * v_A_qmhd / a_initial * (1 - stab_factor) / landau_factor
    
    delta_r = gamma_s_eff * r[i-1] * dt
    r[i] = max(r[i-1] - delta_r, 1e-6)
    
    theta[i] = theta[i-1] + gamma_k_eff * theta[i-1] * dt

# Plot
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

axs[0].plot(t, r, 'blue', lw=2, label='r(t) — sausage')
axs[0].set_ylabel('r (normalized)')
axs[0].grid(alpha=0.3)

axs[1].plot(t, theta, 'green', lw=2, label='θ(t) — kink')
axs[1].set_ylabel('θ (normalized)')
axs[1].grid(alpha=0.3)

axs[2].plot(t, B_z, 'purple', lw=2, label='B_z(t) — dynamo ramp')
axs[2].axhline(B_theta, color='gray', ls='--', label='B_θ threshold')
if t_suppress_bz is not None:
    axs[2].axvline(t_suppress_bz, color='green', ls='--', lw=3, label=f'B_z suppress at {t_suppress_bz:.2f} s')
axs[2].set_ylabel('B_z (normalized)')
axs[2].legend()
axs[2].grid(alpha=0.3)

axs[3].plot(t, S, 'orange', lw=2, label='S(t) — shear')
axs[3].axhline(0.5 * v_A_qmhd / a_initial, color='gray', ls='--', label='γ₀ base (QMHD)')
if t_suppress_shear is not None:
    axs[3].axvline(t_suppress_shear, color='green', ls='--', lw=3, label=f'Shear suppress at {t_suppress_shear:.2f} s')
axs[3].set_ylabel('Shear rate S (s⁻¹)')
axs[3].legend()
axs[3].grid(alpha=0.3)

axs[4].plot(t, n_pair, 'red', lw=2, label='n_pair(t) — pair production')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('Pair density (m⁻³)')
axs[4].legend()
axs[4].grid(alpha=0.3)

plt.suptitle('QMHD Pinch Collapse with Pair Production & Suppression')
plt.tight_layout()
plt.savefig('plots/aladin_v03_qmhd_pinch_collapse.png', dpi=300)
plt.close()

print("QMHD collapse plot saved: plots/aladin_v03_qmhd_pinch_collapse.png")
print(f"B_z suppression time: {t_suppress_bz:.2f} s" if t_suppress_bz else "No B_z suppression")
print(f"Shear suppression time: {t_suppress_shear:.2f} s" if t_suppress_shear else "No shear suppression")
print(f"Max pair density: {np.max(n_pair):.2e} m⁻³")
print("v0.3-QMHD complete — core upgraded!")
