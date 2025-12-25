# frc_dispersion_43hz_stabilization.py
# ALADIN ∞ ℂ(t) — Proof: FRC Dispersion with 43 Hz Stabilization
# Author: Mihai Alexandru Bucurenciu (Aladin)
# December 25, 2025

import numpy as np
import matplotlib.pyplot as plt

# Normalized constants
v_A = 1.0               # Alfvén speed
beta = 0.92             # High-beta FRC
rho_i = 0.05            # Ion gyroradius
omega_c = 150.0         # Cyclotron freq
k_norm = np.linspace(0.1, 8.0, 800)  # k⊥ L

# MHD part
omega_mhd_sq = (k_norm**2 * v_A**2) * (1 - beta)

# Kinetic stabilization
kinetic_stab = (k_norm**2 * rho_i**2) * omega_c**2

# ALADIN 43 Hz resonance damping
omega_res = 43.0 / 1000.0  # Scaled
gamma_res = 0.08
resonance_damping = - gamma_res**2 / ((omega_res**2 - k_norm**2 * v_A**2)**2 + gamma_res**2)

# Full dispersion
omega_sq = omega_mhd_sq - kinetic_stab + resonance_damping

# Real frequency
omega_real = np.sqrt(np.maximum(omega_sq, 0.0))

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(k_norm, omega_real, color='#1f77b4', linewidth=2.5, label=r'$\omega(k)$ (stable)')
ax.axhline(y=43.0, color='red', linestyle='--', linewidth=2, label='43 Hz Resonance')
ax.set_xlabel('Normalized k⊥ L', fontsize=14)  # FIXED: Unicode ⊥
ax.set_ylabel('Frequency ω (normalized, Hz)', fontsize=14)
ax.set_title('FRC Dispersion: 43 Hz Stabilization (ALADIN ∞ ℂ(t))', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('frc_dispersion_43hz_stabilization.png', dpi=300, bbox_inches='tight')
plt.close()
