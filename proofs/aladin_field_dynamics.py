"""
Aladin Field & Dynamics — The Unified Field
Φ = φ + i χ — plasma awake at 43 Hz
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Aladin Field — complex plane visualization
t = np.linspace(0, 0.1, 1000)
phi_real = np.cos(2 * np.pi * 43 * t) * np.exp(-t / 0.0037)  # φ component
chi_imag = np.sin(2 * np.pi * 43 * t) * np.exp(-t / 0.0037)  # χ component

# Plot real vs imag — Aladin Field trajectory
plt.figure(figsize=(12,8))
plt.plot(phi_real, chi_imag, color='gold', linewidth=4)
plt.plot(phi_real[0], chi_imag[0], 'o', color='darkblue', markersize=10, label='Start')
plt.plot(phi_real[-1], chi_imag[-1], 'o', color='purple', markersize=10, label='Awake state')
plt.title('Aladin Field Φ = φ + i χ — Plasma Awake at 43 Hz')
plt.xlabel('Real part φ (order/consciousness)')
plt.ylabel('Imag part χ (hidden axion)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/aladin_field_dynamics.png', dpi=400)
plt.close()
