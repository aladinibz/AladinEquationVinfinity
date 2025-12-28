# phase_mixing_damping_evolution.py
# Compute phase mixing damping evolution for kink modes
# ALADIN ∞ ℂ(t) — Dec 2025

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Parameters
omega_0 = 2 * np.pi * 43.0  # rad/s (base 43 Hz)
delta_omega_A_over_omega = 0.4  # typical spread
t_start = 10.0
t_end = 1000.0
t_vals = np.logspace(np.log10(t_start), np.log10(t_end), 200)

# Power-law amplitude A(t) ∝ t^(-1/3)
amplitude = (t_vals / t_start)**(-1/3)

# Effective damping rate γ(t) ∝ -1/(3t) * (Δω_A/ω_0)^{2/3}
gamma_t = - (1/3) * (delta_omega_A_over_omega)**(2/3) / t_vals

plt.figure(figsize=(12, 6), facecolor='black')

# Amplitude decay
plt.subplot(1, 2, 1)
plt.loglog(t_vals, amplitude, color='gold', lw=3)
plt.xlabel('Time t (s)', color='white')
plt.ylabel('Normalized Amplitude A(t)', color='white')
plt.title('Phase Mixing Amplitude Decay A(t) ∝ t^{-1/3}', color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')

# Damping rate
plt.subplot(1, 2, 2)
plt.loglog(t_vals, -gamma_t, color='lime', lw=3)
plt.xlabel('Time t (s)', color='white')
plt.ylabel('Damping Rate |γ(t)| (1/s)', color='white')
plt.title('Effective Damping Rate γ(t)', color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')

plt.tight_layout()
plt.savefig('plots/phase_mixing_damping_evolution.png', dpi=300, facecolor='black')
plt.close()

print("Plot saved: plots/phase_mixing_damping_evolution.png")
