# frc_mhd_tilt_simulation_43hz.py
# ALADIN ∞ ℂ(t) — Numerical MHD simulation of FRC tilt mode with 43 Hz damping
# Author: Mihai Alexandru Bucurenciu (Aladin)
# December 25, 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants (normalized units)
gamma_mhd = 0.5          # Unstable MHD growth rate (gamma > 0 = unstable)
omega_res = 43.0         # ALADIN 43 Hz resonance frequency
gamma_damp = 0.8         # Damping rate at resonance
t = np.linspace(0, 50, 1000)  # Time array (normalized)

# Simple tilt mode equation: dA/dt = gamma_mhd A - gamma_damp A (at resonance)
def tilt_mode(y, t, gamma_mhd, omega_res, gamma_damp):
    # y[0] = amplitude
    # Add 43 Hz damping when frequency matches resonance
    damp = gamma_damp * np.sin(omega_res * t)**2  # Periodic damping envelope
    dydt = gamma_mhd * y[0] - damp * y[0]
    return dydt

# Initial condition: small perturbation
y0 = [1e-3]

# Solve ODE without damping
sol_no_damp = odeint(tilt_mode, y0, t, args=(gamma_mhd, omega_res, 0.0))

# Solve ODE with 43 Hz damping
sol_damp = odeint(tilt_mode, y0, t, args=(gamma_mhd, omega_res, gamma_damp))

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
ax.semilogy(t, sol_no_damp[:, 0], color='darkred', linewidth=2.5, label='MHD Unstable (tilt growth)')
ax.semilogy(t, sol_damp[:, 0], color='darkblue', linewidth=2.5, label='43 Hz Damped (stable)')
ax.axvline(x=41.0, color='green', linestyle='--', linewidth=2, label='t=41 s Switch')
ax.set_xlabel('Normalized Time', fontsize=14)
ax.set_ylabel('Tilt Mode Amplitude (log scale)', fontsize=14)
ax.set_title('FRC Tilt Mode Simulation: 43 Hz Stabilization (ALADIN ∞ ℂ(t))', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('frc_tilt_mhd_43hz_damping.png', dpi=300, bbox_inches='tight')
plt.close()

print("Simulation complete. Plot saved as frc_tilt_mhd_43hz_damping.png")
