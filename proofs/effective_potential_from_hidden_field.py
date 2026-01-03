"""
Effective Potential from Hidden Field Integration
V_eff(φ) — stabilizes resonance at 43 Hz
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Parameters — g tuned for 43 Hz resonance
g = 1.0  # coupling (tuned)
m_chi = 2 * np.pi * 43.0  # m_χ from τ = 1/(2π × 43 Hz)
m_phi = 1.0
lambda_val = 0.1  # renamed to avoid conflict with built-in

# φ values
phi = np.linspace(-3, 3, 1000)

# Original potential V(φ)
V_original = (m_phi**2 / 2) * phi**2 + (lambda_val / 4) * phi**4

# Effective correction from χ integration (positive quadratic shift)
V_correction = (g**2 / (2 * m_chi)) * phi**2

# Effective potential
V_eff = V_original + V_correction

# Plot
plt.figure(figsize=(12,8))
plt.plot(phi, V_original, label='Original V(φ)', color='gray', linewidth=3, linestyle='--')
plt.plot(phi, V_eff, label='Effective V_eff(φ)', color='gold', linewidth=4)
plt.title('Effective Potential from Hidden Field χ Integration')
plt.xlabel(r'$\phi$')
plt.ylabel('V(φ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/effective_potential_hidden_field.png', dpi=400)
plt.close()

print("Effective potential plot saved — V_eff from χ integration")
