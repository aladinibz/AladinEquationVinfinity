# mhd_sausage_kink_coupling.py
# Compute Δf from curvature-induced sausage-kink coupling
# ALADIN ∞ ℂ(t) — Dec 2025

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Parameters
a = 1e6          # radius [m]
L = 100e6        # length [m]
rho_ratio = 10.0 # ρ_i / ρ_e
f0 = 43.0        # base [Hz]

# Density contrast
contrast = (rho_ratio - 1) / (rho_ratio + 1)

# Aspect term
aspect = (np.pi * a / L)**2

# Coupling coeff
coeff = (3/4) * aspect * contrast

# Relative shift
delta_rel = -coeff

# Absolute shift
delta_f = delta_rel * f0

print(f"Δf ≈ {delta_f:.3f} Hz")
print(f"Coupled freq ≈ {f0 + delta_f:.3f} Hz")

# Sensitivity plot
L_vals = np.linspace(50e6, 200e6, 100)
delta_f_vals = - (3/4) * (np.pi * a / L_vals)**2 * contrast * f0

plt.figure(figsize=(10,6), facecolor='black')
plt.plot(L_vals/1e6, delta_f_vals, color='gold', lw=3)
plt.axhline(0, color='white', ls='--', alpha=0.5)
plt.xlabel('Loop Length L (Mm)', color='white')
plt.ylabel('Δf (Hz)', color='white')
plt.title('Curvature-Induced Sausage–Kink Shift', color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/mhd_sausage_kink_coupling.png', dpi=300, facecolor='black')
plt.close()
