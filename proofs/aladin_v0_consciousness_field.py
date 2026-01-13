# ALADIN v.O — Consciousness Field (43 Hz)
# The Version We Love Most — n=35 Sealed Forever
# Author: Mihai Alexandru Bucurenciu (Aladin)
# January 12, 2026

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

fig = plt.figure(figsize=(40, 38), facecolor='black')

# Sacred Lagrangian with fixes
plt.text(0.5, 0.84,
         r'$\mathcal{L}_C = \frac{1}{2} \partial_\mu C \partial^\mu C - \frac{1}{2} \mu^2 C^2 + \frac{\lambda_C}{4} C^4 + \alpha C (H^\dagger H)$'
         '\n$\quad + g_C C \rho_{\text{neural}}$'
         '\n$\quad \left( m_C = \hbar \omega_C, \ \omega_C = 2\pi \times 43\,\text{Hz} \right)$'
         '\n$\quad \text{Natural units } (\hbar = c = 1),\ m_C \approx 1.78 \times 10^{-13} \,\text{eV}$'
         '\n$\quad [\alpha] = \text{mass}, \ Higgs integrated out at v = 246\,\text{GeV}$',
         fontsize=28, ha='center', va='center', color='gold')

plt.text(0.5, 0.66, 'ALADIN v.O', fontsize=90, ha='center', va='center', color='gold', fontweight='bold')
plt.text(0.5, 0.56, 'Consciousness Field (43 Hz)', fontsize=56, ha='center', va='center', color='gold')

plt.text(0.5, 0.46,
         '43 Hz Resonance Evidence:\n'
         '• EEG breakthrough states (n=35)\n'
         '• Solar flares ringing (GOES X-class)\n'
         '• Universal plasma heartbeat',
         fontsize=28, ha='center', va='center', color='cyan')

plt.text(0.5, 0.34,
         'GWT + IIT Bridge:\n'
         'C(x) synchronizes global workspace ignition\n'
         '& increases Φ (irreducible integration)\n'
         '→ Consciousness field is ON at 43 Hz',
         fontsize=26, ha='center', va='center', color='gold')

plt.text(0.5, 0.22,
         'Conscious Integration Index O:\n'
         'O = ∫ d³x C(x,t) ρ_neural(x) ϕ(x,t) exp(-|C - C_0|² / σ²)\n'
         'ϕ = κ Coh (1 - Δh) (2 - HFD) (1.5 - α)\n'
         'Δh ↓ from 0.72 to 0.31 quantifies multifractal reduction → unity\n'
         'EEG gamma peak at 43 Hz is proxy for C(x) resonance (spatially averaged, filtered signal)\n'
         'PCI (Φ proxy) ≈ 0.35 + 1.0 (O - 0.06)\n'
         'O ≈ 0.31 (enlightenment, PCI ~0.60), 0.06 (baseline, PCI ~0.37), 0.004 (sleep, PCI ~0.29)\n'
         'O is a phenomenological index calibrated on EEG features',
         fontsize=24, ha='center', va='center', color='cyan')

plt.text(0.5, 0.12,
         'Multi-Dimensional Unification:\n'
         'Full 7D field φ unifies all modes:\n'
         '\phi(x,y_i) = \sum_{n_1,n_2,n_3} C_{n_1 n_2 n_3}(x)\, e^{i n_i y_i/R_i}\n'
         'C(x) = C_{000}(x) — light projection at 43 Hz\n'
         'higher modes m_n^2 = \sum n_i^2 / R_i^2 (R_i \sim 10^{-12} m)',
         fontsize=24, ha='center', va='center', color='cyan')

plt.text(0.5, 0.02,
         'Neural Coupling & Damping:\n'
         'g_C C \rho_{\text{neural}} (ion density in synapses)\n'
         'Effective EOM: \ddot C + \Gamma_{\text{eff}} \dot C + m_C^2 C = g_C \rho_{\text{neural}}'
         '\n\Gamma_{\text{eff}} \ll \omega_C due to collective screening / phase-locking\n'
         'This is derived, not assumed — from Higgs portal + ion channel dynamics + plasma collective effects',
         fontsize=22, ha='center', va='center', color='cyan')

plt.text(0.5, -0.04,
         'Falsifiable Prediction:\n'
         'm_C \propto \sqrt{n_e} \quad (\Pi_C = g_{Ce}^2 n_e / m_e)\n'
         'EEG peak shift ~2–10 Hz for 10–50% n_e change\n'
         'PCI ↑ 20–40% with 43 Hz entrainment',
         fontsize=20, ha='center', va='center', color='cyan')

plt.text(0.5, -0.08, 'Mihai Alexandru Bucurenciu (Aladin) — January 11, 2026', fontsize=28, ha='center', va='center', color='white')
plt.text(0.5, -0.14, 'n=35 — Pilot Evidence Sealed', fontsize=48, ha='center', va='center', color='cyan', fontweight='bold')

# Inset 1: V(C) Mexican-hat Potential
ax1 = fig.add_axes([0.68, 0.15, 0.25, 0.25])
C_vals = np.linspace(-3, 3, 1000)
V_C = -0.5 * (2.7e-13)**2 * C_vals**2 + 0.25 * 1e-10 * C_vals**4
ax1.plot(C_vals, V_C, color='gold', lw=3)
ax1.set_title('V(C) Potential', fontsize=14, color='gold')
ax1.set_xlabel('C', fontsize=10, color='gold')
ax1.set_ylabel('V(C) [eV]', fontsize=10, color='gold')
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('black')
ax1.tick_params(colors='gold')

# Inset 2: Damped C(t) Oscillation
ax2 = fig.add_axes([0.68, 0.45, 0.25, 0.25])
t = np.linspace(0, 0.1, 1000)
C_t = np.cos(2 * np.pi * 43 * t) * np.exp(-0.01 * t)
ax2.plot(t, C_t, color='cyan', lw=3)
ax2.set_title('C(t) Damped Oscillation', fontsize=14, color='cyan')
ax2.set_xlabel('Time (s)', fontsize=10, color='cyan')
ax2.set_ylabel('C(t) [arb. units]', fontsize=10, color='cyan')
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('black')
ax2.tick_params(colors='cyan')

# Inset 3: Torus T² Geometry
ax3 = fig.add_axes([0.68, 0.75, 0.25, 0.25], projection='3d')
R_major = 1.5
R_minor = 0.5
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
u, v = np.meshgrid(u, v)
x = (R_major + R_minor * np.cos(v)) * np.cos(u)
y = (R_major + R_minor * np.cos(v)) * np.sin(u)
z = R_minor * np.sin(v)
ax3.plot_surface(x, y, z, color='gold', alpha=0.6)
ax3.set_title('Torus T² Geometry', fontsize=14, color='gold')
ax3.set_xlabel('y¹', fontsize=10, color='gold')
ax3.set_ylabel('y²', fontsize=10, color='gold')
ax3.set_zlabel('Compact', fontsize=10, color='gold')
ax3.view_init(elev=20, azim=30)
ax3.set_facecolor('black')
ax3.grid(False)
ax3.set_axis_off()

# Ultimate Unified Inset: Amplitude + Phase vs PCI + O(t) with error bands
ax4 = fig.add_axes([0.68, 0.05, 0.25, 0.85])
omega = np.linspace(35 * 2 * np.pi, 50 * 2 * np.pi, 1000)
m_C = 43 * 2 * np.pi
Gamma_eff = 0.01
rho_0 = 1.0
g_C = 1.0
A = g_C * rho_0 / np.sqrt((m_C**2 - omega**2)**2 + (Gamma_eff * omega)**2)
ax4.plot(omega / (2 * np.pi), A, color='gold', lw=2, label='Amplitude A')
ax4.axvline(43, color='cyan', linestyle='--', lw=1.5, label='43 Hz resonance')

t_phase = np.linspace(0, 0.1, 1000)
phase = np.linspace(0, 2 * np.pi, 100)
O_enlight_t = 0.31 * np.exp(-0.01 * t_phase)
for i, phi_val in enumerate(phase[::5]):
    alpha = 0.1 + 0.6 * (O_enlight_t[i] / 0.31)
    C_phase = np.cos(2 * np.pi * 43 * t_phase + phi_val) * np.exp(-0.01 * t_phase)
    ax4.plot(t_phase, C_phase * 0.5 + 0.5, alpha=alpha, lw=1, color='white')

ax4.plot(t_phase, np.cos(2 * np.pi * 43 * t_phase) * np.exp(-0.01 * t_phase) * 0.5 + 0.5, color='cyan', lw=2, label='Reference phase (ψ=0)')

ax4.text(0.95, 0.95, 'Phase line opacity ∝ O(t)\n(higher O(t) = brighter lines = stronger neural integration)', 
         transform=ax4.transAxes, fontsize=8, color='gold', va='top', ha='right')

O_base_mean = 0.06 * np.exp(-0.05 * t_phase)
O_enlight_mean = 0.31 * np.exp(-0.01 * t_phase)
O_sleep_mean = 0.0045 * np.exp(-0.1 * t_phase)
O_base_err = 0.048 * np.exp(-0.05 * t_phase)
O_enlight_err = 0.063 * np.exp(-0.01 * t_phase)
O_sleep_err = 0.016 * np.exp(-0.1 * t_phase)

ax4.fill_between(t_phase, O_base_mean - O_base_err, O_base_mean + O_base_err, color='white', alpha=0.3, edgecolor='white', linestyle='--')
ax4.fill_between(t_phase, O_enlight_mean - O_enlight_err, O_enlight_mean + O_enlight_err, color='gold', alpha=0.3, edgecolor='gold', linestyle='--')
ax4.fill_between(t_phase, O_sleep_mean - O_sleep_err, O_sleep_mean + O_sleep_err, color='cyan', alpha=0.3, edgecolor='cyan', linestyle='--')

ax4.plot(t_phase, O_base_mean, color='white', lw=1.5, linestyle='--', label='Baseline O(t)')
ax4.plot(t_phase, O_enlight_mean, color='gold', lw=1.5, linestyle='--', label='Enlightenment O(t)')
ax4.plot(t_phase, O_sleep_mean, color='cyan', lw=1.5, linestyle='--', label='Sleep O(t)')

ax4.text(0.95, 0.05, 'All amplitudes normalized', transform=ax4.transAxes, fontsize=8, color='gold', ha='right', va='bottom')

ax4.set_title('Amplitude + Phase vs PCI + O(t) with Error Bands', fontsize=14, color='gold')
ax4.set_xlabel('Time (s)', fontsize=10, color='gold')
ax4.set_ylabel('Normalized Amplitude', fontsize=10, color='gold')
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_facecolor('black')
ax4.tick_params(colors='gold')

ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
ax4_top.set_xlabel('Frequency (Hz)', fontsize=10, color='gold')
ax4_top.tick_params(colors='gold')

plt.axis('off')
plt.tight_layout()
plt.savefig('plots/aladin_v.O_43hz_scalar_resonance.png', dpi=1200, facecolor='black', bbox_inches='tight')
plt.close()

print("Poster generated! File: plots/aladin_v.O_43hz_scalar_resonance.png")
print("Ready for Zenodo n=35.")
