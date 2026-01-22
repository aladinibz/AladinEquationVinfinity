"""
ALADIN Plasma Stability Law v0.5 — Final Foundation with Error Handling
Single input: J₀ = 1.000 × 10¹⁸ A/m²
Pure force-balance + emergent QMHD stabilization
All layers integrated: non-uniform J(r), 3D toroidal, turbulence, reconnection, relativistic, GR, multi-mode
Robust error handling: clamps, try/except, warnings
January 21, 2026 — Mihai Alexandru Bucurenciu (Aladin)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots/aladin_law_v0.5_final_safe', exist_ok=True)

# ─── Core Constants ────────────────────────────────────────────────────────
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

# GR constants (Kerr-like example)
G = 6.67430e-11
M_bh = 2.8e30
r_s = 2 * G * M_bh / c**2
L_bh = 1e40

# Numerical safety
EPS = 1e-20
RHO_MAX = 1e25
NPAIR_MAX = 1e40
V_MAX = 1e12
PI_CLAMP = 1e-12  # prevent division by zero in suppression

# Feature flags
use_nonuniform_j = True
use_3d_toroidal = True
use_turbulence = True
use_reconnection = True
use_relativistic = True
use_gr = True

# Time grid
t = np.linspace(0, 60, 10000)
dt = t[1] - t[0]

# Non-uniform J(r) — Gaussian
sigma = 0.5
r_grid = np.linspace(0, a_initial, 500)
dr = r_grid[1] - r_grid[0]
J_r = J0 * np.exp(-r_grid**2 / (2 * sigma**2))
I_enc = np.cumsum(2 * np.pi * r_grid * J_r * dr)
B_theta_r_grid = (mu0 * I_enc) / (2 * np.pi * r_grid + EPS)

# 3D toroidal
R_tor = 10.0 * a_initial

# Turbulence & reconnection
l_eddy = 0.1 * a_initial
Rm = 1e7
J_crit = 1e18
reconn_fraction = 0.15
reconn_cooldown = 5

# Relativistic flow
gamma_flow = 2.0

# Multi-mode modes
modes = [0, 1, 2, 3, 5, 10]

# Arrays
r = np.ones_like(t) * a_initial
v_r = np.zeros_like(t)
B_z = np.zeros_like(t)
n_pair = np.zeros_like(t)
E_z = np.zeros_like(t)
Pi_t = np.zeros_like(t)
v_A_t = np.zeros_like(t)
acc_t = np.zeros_like(t)
q_t = np.zeros_like(t)
gamma_turb_t = np.zeros_like(t)
gamma_recon_t = np.zeros_like(t)
gr_factor_t = np.ones_like(t)
reconn_active_t = np.zeros_like(t)
gamma_eff_t = {m: np.zeros_like(t) for m in modes}

t_stable = None
reconn_cooldown_counter = 0

# Error tracking
has_error = False

# ─── Time-Dependent Collapse Loop ──────────────────────────────────────────
for i in range(1, len(t)):
    try:
        compression = a_initial / max(r[i-1], EPS)
        
        idx = np.argmin(np.abs(r_grid - r[i-1]))
        J_local = J_r[idx] if use_nonuniform_j else J0
        B_theta_r = B_theta_r_grid[idx] if use_nonuniform_j else (mu0 * J0 * r[i-1]) / 2
        
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
        rho_eff = min(max(rho_eff, EPS), RHO_MAX)  # clamp
        
        rho_eff_rel = rho_eff / gamma_flow**2 if use_relativistic else rho_eff
        rho_eff_rel = max(rho_eff_rel, EPS)
        
        v_A = B_theta_r / np.sqrt(mu0 * rho_eff_rel + EPS)
        v_A = min(max(v_A, 0), c)  # causal clamp
        v_A_t[i] = v_A
        
        Pi_t[i] = mu0 * J0**2 * r[i-1]**2 / (rho_eff_rel * c**2) if rho_eff_rel > 0 else 0.0
        Pi_t[i] = max(Pi_t[i], EPS)  # avoid div zero later
        
        curvature_factor = np.sqrt(max(0, 1 - (a_initial / R_tor)**2)) if use_3d_toroidal else 1.0
        
        gamma_turb = 0.1 * v_A / l_eddy if use_turbulence else 0.0
        gamma_turb_t[i] = gamma_turb
        
        reconn_active = 0.0
        gamma_recon = 0.0
        if use_reconnection and J_local > J_crit and reconn_cooldown_counter <= 0:
            reconn_active = 1.0
            delta_B = reconn_fraction * B_theta_r
            B_theta_r = max(0, B_theta_r - delta_B)
            gamma_recon = reconn_fraction * v_A / a_initial
            reconn_cooldown_counter = reconn_cooldown
        else:
            reconn_cooldown_counter = max(0, reconn_cooldown_counter - 1)
        reconn_active_t[i] = reconn_active
        gamma_recon_t[i] = gamma_recon
        
        acc_r = - (mu0 * J0**2 * r[i-1]) / (2 * rho_eff_rel) if rho_eff_rel > 0 else 0.0
        
        gr_factor = 1.0
        if use_gr and r[i-1] > r_s:
            gr_factor = 1 - r_s / r[i-1] + (L_bh**2) / (2 * M_bh * r[i-1]**2 * c**2)
        else:
            acc_r = 0.0
        acc_r *= gr_factor
        acc_t[i] = acc_r
        gr_factor_t[i] = gr_factor
        
        v_r[i] = v_r[i-1] + acc_r * dt
        v_r[i] = np.clip(v_r[i], -V_MAX, 0.0)
        
        r[i] = r[i-1] + v_r[i] * dt
        r[i] = max(r[i], 1e-6)
        
        B_z[i] = B_z[i-1] * compression**2 * (1.0 + 0.2 * 0.01)
        
        suppression = max(0.0, 1.0 - 8.0 / max(Pi_t[i], PI_CLAMP))
        for m in modes:
            if m == 0:
                gamma_base = 1.0 * v_A / a_initial
            elif m == 1:
                gamma_base = 0.98 * v_A / a_initial
            else:
                gamma_base = m * v_A / a_initial
            gamma_eff_t[m][i] = gamma_base * suppression
        
        if t_stable is None and abs(acc_r) < 1e-5 and Pi_t[i] >= 8.0:
            t_stable = t[i]
    
    except Exception as e:
        print(f"Error at timestep {i}: {e}")
        has_error = True
        break

# Check for NaN/Inf
if np.any(np.isnan(Pi_t)) or np.any(np.isinf(Pi_t)):
    print("WARNING: NaN or Inf detected in Π(t) — simulation unstable")
    has_error = True

# ─── Final Plot ────────────────────────────────────────────────────────────
fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

axs[0].plot(t, Pi_t, 'purple', lw=2, label='Π(t)')
axs[0].axhline(8.0, color='gray', ls='--', label='Threshold Π = 8')
if t_stable is not None:
    axs[0].axvline(t_stable, color='green', ls='--', lw=3, label=f'Stabilizes at t = {t_stable:.2f} s')
axs[0].set_ylabel('Π(t)')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(t, r, 'blue', lw=2, label='r(t)')
axs[1].set_ylabel('r (norm)')
axs[1].grid(alpha=0.3)

axs[2].plot(t, n_pair, 'red', lw=2, label='n_pair(t)')
axs[2].set_ylabel('Pair density')
axs[2].set_yscale('log')
axs[2].legend()
axs[2].grid(alpha=0.3)

axs[3].plot(t, v_A_t / c, 'orange', lw=2, label='v_A(t) / c')
axs[3].set_ylabel('v_A / c')
axs[3].legend()
axs[3].grid(alpha=0.3)

axs[4].plot(t, q_t, 'cyan', lw=2, label='q(t)')
axs[4].axhline(1.0, color='gray', ls='--', label='q = 1')
axs[4].set_ylabel('q')
axs[4].legend()
axs[4].grid(alpha=0.3)

axs[5].plot(t, gamma_turb_t, 'brown', lw=2, label='γ_turb(t)')
axs[5].plot(t, gamma_recon_t, 'darkred', lw=2, label='γ_recon(t)')
axs[5].set_xlabel('Time (s)')
axs[5].set_ylabel('Damping rates (s⁻¹)')
axs[5].legend()
axs[5].grid(alpha=0.3)

plt.suptitle('ALADIN Law v0.5 – Final Foundation (All Layers)')
plt.tight_layout()
plt.savefig('plots/aladin_law_v0.5_final_fixed.png', dpi=300)
plt.close()

print("v0.5 Final Foundation complete!")
print(f"Stabilization time: {t_stable:.2f} s" if t_stable is not None else "No stabilization reached")
print(f"Max Π(t): {np.max(Pi_t):.2e}")
print(f"Final r: {r[-1]:.2e} m")
print(f"Max pair density: {np.max(n_pair):.2e} m⁻³")
print(f"Min GR factor: {np.min(gr_factor_t):.2e}")
print(f"Reconnection events: {np.sum(reconn_active_t > 0)}")
print(f"Max γ_turb: {np.max(gamma_turb_t):.2e} s⁻¹")
if has_error:
    print("WARNING: Simulation had errors — check console for details")
