import numpy as np
import matplotlib.pyplot as plt

# Multipoles
l = np.arange(2, 3000)

# 43 Hz fundamental
omega = 2 * np.pi * 43

# 7 octonion imaginary units (e1-e7)
e = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

# Baseline CMB power
Cl = 1e-3 * l * (l + 1) * np.exp(-l / 1000)

# Octonionic modulation
Cl_oct = np.zeros_like(l, dtype=float)
for i in range(7):
    phase = omega * l * (i + 1) / 1000.0
    Cl_oct += e[i] * Cl * np.sin(phase)**2

# Aladin v∞ damping + acoustic peaks
Cl_aladin = Cl_oct * np.exp(-l / 800) * np.sin(l / 220)**2

# Plot
plt.figure(figsize=(10,6))
plt.semilogy(l, Cl, 'gray', ls='--', label='Baseline')
plt.semilogy(l, Cl_oct, 'purple', alpha=0.7, label='Octonionic S(7)')
plt.semilogy(l, Cl_aladin, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('Multipole l')
plt.ylabel('C$_l$ [μK²]')
plt.title('Octonionic CMB Peaks')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('octonionic_cmb_peaks.png', dpi=300)
plt.close()

print("Plot saved: octonionic_cmb_peaks.png")
