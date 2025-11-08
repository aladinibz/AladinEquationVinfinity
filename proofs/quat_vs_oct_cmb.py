import numpy as np
import matplotlib.pyplot as plt

# Multipoles
l = np.arange(2, 3000)

# 43 Hz fundamental
omega = 2 * np.pi * 43

# Baseline CMB power spectrum
Cl = 1e-3 * l * (l + 1) * np.exp(-l / 1000.0)

# Quaternion: 3 modes (i, j, k)
e_quat = np.array([1.0, 0.9, 0.8])
Cl_quat = np.zeros_like(l, dtype=float)
for i in range(3):
    phase = omega * l * (i + 1) / 1000.0
    Cl_quat += e_quat[i] * Cl * np.sin(phase)**2

# Octonion: 7 modes (e1–e7)
e_oct = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
Cl_oct = np.zeros_like(l, dtype=float)
for i in range(7):
    phase = omega * l * (i + 1) / 1000.0
    Cl_oct += e_oct[i] * Cl * np.sin(phase)**2

# Aladin v∞ damping + acoustic peaks
Cl_aladin = Cl_oct * np.exp(-l / 800.0) * np.sin(l / 220.0)**2

# Plot
plt.figure(figsize=(12, 7))
plt.semilogy(l, Cl, 'gray', ls='--', alpha=0.7, label='Baseline')
plt.semilogy(l, Cl_quat, 'blue', lw=2, alpha=0.8, label='Quaternion (3 modes)')
plt.semilogy(l, Cl_oct, 'purple', lw=2, alpha=0.8, label='Octonion (7 modes)')
plt.semilogy(l, Cl_aladin, 'gold', lw=3, label='Aladin v∞ + Octonion')
plt.xlabel('Multipole l')
plt.ylabel(r'$C_\ell$ [$\mu$K²]')
plt.title('Quaternion vs Octonion CMB Modulation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('quat_vs_oct_cmb.png', dpi=300)
plt.close()

print("Plot saved: quat_vs_oct_cmb.png")
