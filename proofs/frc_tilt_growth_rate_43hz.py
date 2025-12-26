# frc_tilt_growth_rate_43hz.py
# ALADIN ∞ ℂ(t) — Growth Rate Simulation of FRC Tilt Mode with 43 Hz Damping
# Author: Mihai Alexandru Bucurenciu (Aladin)
# December 26, 2025

import numpy as np
import matplotlib.pyplot as plt

# Time array (normalized)
t = np.linspace(0, 60, 1200)

# Base MHD growth rate (unstable without damping)
gamma_mhd = 0.5  # normalized growth rate

# ALADIN 43 Hz damping envelope (periodic suppression)
omega_res = 43.0  # Hz
gamma_damp = 0.8  # damping strength
damping_envelope = gamma_damp * np.sin(omega_res * t)**2

# Effective growth rate: MHD - damping (switches on at t=41 s)
gamma_eff = np.where(t < 41.0, gamma_mhd, gamma_mhd - damping_envelope)

# Integrate to get amplitude (for reference)
amplitude = np.exp(np.cumsum(gamma_eff) * (t[1] - t[0]))

# Plot growth rate
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(t, gamma_eff, color='#d62728', linewidth=2.5, label='Effective Growth Rate γ(t)')
ax.axvline(x=41.0, color='green', linestyle='--', linewidth=2, label='t=41 s Switch')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Normalized Time', fontsize=14)
ax.set_ylabel('Growth Rate γ (normalized)', fontsize=14)
ax.set_title('FRC Tilt Mode Growth Rate: 43 Hz Damping Suppression (ALADIN ∞ ℂ(t))', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('frc_tilt_growth_rate_43hz.png', dpi=300, bbox_inches='tight')
plt.close()

print("Growth rate plot saved as frc_tilt_growth_rate_43hz.png")
