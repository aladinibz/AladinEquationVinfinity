"""
ALADIN Plasma Stability Law v0.4 — Pure J₀ Z-Pinch Core
Single measured input: J₀ = 1.000 × 10¹⁸ A/m²
Derived: ρ_eff from J₀ balance, v_A from J₀ + ρ_eff, growth from B_θ
QMHD: Schwinger pair production with induced E_z, quantum pressure, ρ_eff_QMHD
Emergent B_z & shear suppression
Frozen core — January 20, 2026 — Mihai Alexandru Bucurenciu (Aladin)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots/aladin_law_v0.4', exist_ok=True)

# ─── Physical & QED Constants ──────────────────────────────────────────────
J0 = 1.000e18
mu0 = 4 * np.pi * 1e-7
c = 3e8
a_initial = 1.0
rho_eff_class = 1.745e12

m_e = 9.1093837e-31
e = 1.60217662e-19
hbar = 1.0545718e-34
B_c = 4.4e9
E_c = B_c * c
alpha_em = 1 / 137.036

# Time grid
t = np.linspace(0, 60, 5000)
dt = t[1] - t[0]

# Arrays
r = np.ones_like(t) * a_initial
theta = np.zeros_like(t)
B_z = np.zeros_like(t)
S = np.zeros_like(t)
n_pair = np.zeros_like(t)
E_z = np.zeros_like(t)

t_suppress_bz = None
t_suppress_shear = None

B_theta = (mu0 * J0 * a_initial) / 2

# ─── Main Simulation Loop ──────────────────────────────────────────────────
B_z_seed_factor = 0.1
B_z_amp_factor  = 0.2

for i in range(1, len(t)):
    compression = a_initial / r[i-1] if r[i-1] > 0 else 1.0
    B_current = B_theta * (r[i-1] / a_initial)
    
    if i > 1:
        dB_z_dt = (B_z[i-1] - B_z[i-2]) / dt
    else:
        dB_z_dt = 0.0
    E_z[i] = - (r[i-1] / 2.0) * dB_z_dt
    
    Gamma_pair = 0.0
    if abs(E_z[i]) > 1e10:
        Gamma_pair = (alpha_em * abs(E_z[i]) * E_c / hbar) * \
                     (B_current / B_c)**2 * \
                     np.exp(-np.pi * E_c / abs(E_z[i]))
    
    n_pair_new = n_pair[i-1] + Gamma_pair * dt
    max_pairs  = (B_current**2 / (2.0 * mu0)) / (2.0 * m_e * c**2) if B_current > 0 else 0.0
    n_pair[i]  = min(n_pair_new, max_pairs)
    
    P_q = 0.0
    if n_pair[i] > 0:
        P_q = (3.0/5.0) * n_pair[i] * (3.0 * np.pi**2 * n_pair[i])**(2.0/3.0) * hbar**2 / m_e
    
    rho_eff_qmhd = rho_eff_class + 2.0 * m_e * n_pair[i] + P_q / c**2
    v_A_qmhd = B_theta / np.sqrt(mu0 * rho_eff_qmhd) if mu0 * rho_eff_qmhd > 0 else 0.0
    
    if B_z[i-1] == 0 and theta[i-1] > 0.01:
        B_z[i] = B_z_seed_factor * B_theta
    else:
        B_z[i] = B_z[i-1]
    B_z[i] = B_z[i] * compression**2 * (1.0 + B_z_amp_factor * theta[i-1])
    B_z[i] = min(B_z[i], B_theta)
    
    S[i] = 0.1 * v_A_qmhd / a_initial * compression if a_initial > 0 else 0.0
    
    if t_suppress_bz is None and B_z[i] >= B_theta:
        t_suppress_bz = t[i]
    if t_suppress_shear is None and S[i] >= 0.5 * v_A_qmhd / a_initial:
        t_suppress_shear = t[i]
    
    stab_bz    = (B_z[i] / B_theta)**2 if B_theta != 0 else 0.0
    stab_shear = min(S[i] / (0.5 * v_A_qmhd / a_initial), 1.0) if v_A_qmhd != 0 and a_initial != 0 else 0.0
    stab_factor = max(stab_bz, stab_shear)
    
    n_L    = e * B_current / (2.0 * np.pi * hbar) if hbar != 0 else 0.0
    n_class = rho_eff_qmhd / 1.67e-27 if rho_eff_qmhd != 0 else 1.0
    landau_factor = np.sqrt(1.0 + n_L / n_class) if n_class > 0 else 1.0
    
    gamma_s_eff = 0.5 * v_A_qmhd / a_initial * (1.0 - stab_factor) / landau_factor if a_initial != 0 else 0.0
    gamma_k_eff = 0.3 * v_A_qmhd / a_initial * (1.0 - stab_factor) / landau_factor if a_initial != 0 else 0.0
    
    delta_r = gamma_s_eff * r[i-1] * dt
    r[i] = max(r[i-1] - delta_r, 1e-6)
    
    theta[i] = theta[i-1] + gamma_k_eff * theta[i-1] * dt

# ─── Stability Metrics ─────────────────────────────────────────────────────
Pi_t          = np.zeros_like(t)
gamma_s_eff_t = np.zeros_like(t)
gamma_k_eff_t = np.zeros_like(t)
v_A_norm_t    = np.zeros_like(t)

for i in range(len(t)):
    compression = a_initial / r[i] if r[i] > 0 else 1.0
    B_current = B_theta * (r[i] / a_initial)
    
    P_q = 0.0
    if n_pair[i] > 0:
        P_q = (3.0/5.0) * n_pair[i] * (3.0 * np.pi**2 * n_pair[i])**(2.0/3.0) * hbar**2 / m_e
    
    rho_eff_qmhd_i = rho_eff_class + 2.0 * m_e * n_pair[i] + P_q / c**2
    
    Pi_t[i] = mu0 * J0**2 * r[i]**2 / (rho_eff_qmhd_i * c**2) if rho_eff_qmhd_i != 0 else 0.0
    
    v_A_qmhd_i = B_theta / np.sqrt(mu0 * rho_eff_qmhd_i) if mu0 * rho_eff_qmhd_i > 0 else 0.0
    v_A_norm_t[i] = v_A_qmhd_i / c
    
    n_L_i = e * B_current / (2.0 * np.pi * hbar) if hbar != 0 else 0.0
    n_class_i = rho_eff_qmhd_i / 1.67e-27 if rho_eff_qmhd_i != 0 else 1.0
    landau_factor_i = np.sqrt(1.0 + n_L_i / n_class_i) if n_class_i > 0 else 1.0
    
    stab_bz_i   = (B_z[i] / B_theta)**2 if B_theta != 0 else 0.0
    stab_shear_i = min(S[i] / (0.5 * v_A_qmhd_i / a_initial), 1.0) if v_A_qmhd_i != 0 and a_initial != 0 else 0.0
    stab_factor_i = max(stab_bz_i, stab_shear_i)
    
    gamma_s_eff_t[i] = 0.5 * v_A_qmhd_i / a_initial * (1.0 - stab_factor_i) / landau_factor_i if a_initial != 0 else 0.0
    gamma_k_eff_t[i] = 0.3 * v_A_qmhd_i / a_initial * (1.0 - stab_factor_i) / landau_factor_i if a_initial != 0 else 0.0

# ─── Final Stability Plot ──────────────────────────────────────────────────
fig_stab, axs_stab = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

axs_stab[0].plot(t, Pi_t, 'purple', lw=2, label='Pi(t)')
axs_stab[0].axhline(8.0, color='gray', ls='--', label='Threshold Pi = 8')
if np.any(Pi_t >= 8):
    axs_stab[0].fill_between(t, Pi_t, 8.0, where=(Pi_t >= 8),
                             color='green', alpha=0.15, label='Stable')
axs_stab[0].set_ylabel('Pi(t)')
axs_stab[0].set_yscale('log' if np.any(Pi_t > 0) else 'linear')
axs_stab[0].legend(loc='upper left')
axs_stab[0].grid(alpha=0.3)

axs_stab[1].plot(t, gamma_s_eff_t, 'blue', lw=2, label='gamma_s_eff')
axs_stab[1].plot(t, gamma_k_eff_t, 'green', lw=2, label='gamma_k_eff')
axs_stab[1].set_ylabel('Growth rate (s^-1)')
axs_stab[1].set_yscale('log')
axs_stab[1].legend()
axs_stab[1].grid(alpha=0.3)

axs_stab[2].plot(t, n_pair, 'red', lw=2, label='n_pair(t)')
axs_stab[2].set_ylabel('Pair density (m^-3)')
axs_stab[2].set_yscale('log')
axs_stab[2].legend()
axs_stab[2].grid(alpha=0.3)

axs_stab[3].plot(t, v_A_norm_t, 'orange', lw=2, label='v_A(t) / c')
axs_stab[3].set_xlabel('Time (s)')
axs_stab[3].set_ylabel('v_A / c')
axs_stab[3].legend()
axs_stab[3].grid(alpha=0.3)

plt.suptitle('ALADIN Plasma Stability Law v0.4 – Stability Evolution')
plt.tight_layout()
plt.savefig('plots/aladin_law_v0.4_stability.png', dpi=300)
plt.close()
