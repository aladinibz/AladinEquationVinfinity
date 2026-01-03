"""
ALADIN Second-Order Correction
Nonlinear vacuum response
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
lambda_val = 0.15  # small for perturbation

# φ range
phi = np.linspace(-3, 3, 1000)

# Zeroth order V0 = m²/2 φ²
V0 = (m**2 / 2) * phi**2

# First order shift
V1 = (J0**2 / (2 * m**2)) * phi**2

# Approximate second-order shift (cubic backreaction)
V2 = lambda_val * (J0 / m**2)**3 * phi**2

# Total up to second order
V_total = V0 + V1 + V2

# Plot
plt.figure(figsize=(12,8))
plt.plot(phi, V0, label='Zeroth order', color='gray', linestyle='--')
plt.plot(phi, V0 + V1, label='First order', color='purple')
plt.plot(phi, V_total, label='Second order', color='gold', linewidth=4)
plt.title('ALADIN Second-Order Correction — Nonlinear Response')
plt.xlabel('φ')
plt.ylabel('V(φ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/aladin_second_order_correction.png', dpi=400)
plt.close()

print("Second-order correction plot saved — nonlinear vacuum response")
