"""
ALADIN Perturbative Solution — Vacuum Shift
Nonlinear term shifts effective mass
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Parameters
J0 = 1.0
m = 1.0
lambda_val = 0.2

# φ range
phi = np.linspace(-3, 3, 1000)

# Original potential V(φ) = m²/2 φ² + λ/4 φ⁴
V_original = (m**2 / 2) * phi**2 + (lambda_val / 4) * phi**4

# First-order effective shift from source + interaction
V_eff = V_original + (J0**2 / (2 * m**2)) * phi**2

# Plot
plt.figure(figsize=(12,8))
plt.plot(phi, V_original, label='Original V(φ)', color='gray', linewidth=3, linestyle='--')
plt.plot(phi, V_eff, label='Effective V_eff(φ)', color='gold', linewidth=4)
plt.title('ALADIN Perturbative Solution — Vacuum Shift')
plt.xlabel(r'$\phi$')
plt.ylabel('V(φ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/aladin_perturbative_vacuum_shift.png', dpi=400)
plt.close()

print("Perturbative solution plot saved — vacuum shift from nonlinear term")
