import numpy as np
import matplotlib.pyplot as plt

# Sedenion 15-mode CMB modulation
l = np.arange(2, 3000)
omega = 2 * np.pi * 43  # 43 Hz

# 15 sedenion imaginary units
e_sed = np.linspace(1.0, 0.1, 15)  # e1 to e15 weights

# Baseline CMB
Cl = 1e-3 * l * (l + 1) * np.exp(-l / 1000.0)

# Sedenion modulation
Cl_sed = np.zeros_like(l, dtype=float)
for i in range(15):
    phase = omega * l * (i + 1) / 1000.0
    Cl_sed += e_sed[i] * Cl * np.sin(phase)**2

# Aladin v∞ damping + acoustic peaks
Cl_aladin = Cl_sed * np.exp(-l / 800.0) * np.sin(l / 220.0)**2

# Plot
plt.figure(figsize=(12,7))
plt.semilogy(l, Cl, 'gray', ls='--', alpha=0.7, label='Baseline CMB')
plt.semilogy(l, Cl_sed, 'orange', lw=2, alpha=0.8, label='Sedenion 15 Modes')
plt.semilogy(l, Cl_aladin, 'gold', lw=3, label='Aladin v∞ + Sedenion')
plt.xlabel('Multipole l')
plt.ylabel('C$_l$ [μK²]')
plt.title('Sedenion 15-Mode CMB Modulation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('sedenion_cmb_15mode.png', dpi=300)
plt.close()

print("Sedenion 15-mode CMB plot saved: sedenion_cmb_15mode.png")
