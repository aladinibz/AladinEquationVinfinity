import numpy as np
import matplotlib.pyplot as plt

# Quaternion CMB modulation — 3 imaginary units (i,j,k)
l = np.arange(2, 3000)
omega = 2 * np.pi * 43  # 43 Hz

# 3 quaternion imaginary units
e = np.array([1.0, 0.9, 0.8])  # i, j, k weights

# Baseline CMB
Cl = 1e-3 * l * (l + 1) * np.exp(-l / 1000)

# Quaternion modulation
Cl_quat = np.zeros_like(l, dtype=float)
for i in range(3):
    phase = omega * l * (i + 1) / 1000.0
    Cl_quat += e[i] * Cl * np.sin(phase)**2

# Aladin v∞ damping + acoustic peaks
Cl_aladin = Cl_quat * np.exp(-l / 800) * np.sin(l / 220)**2

# Plot
plt.figure(figsize=(10,6))
plt.semilogy(l, Cl, 'gray', ls='--', label='Baseline')
plt.semilogy(l, Cl_quat, 'purple', alpha=0.7, label='Quaternion ijk')
plt.semilogy(l, Cl_aladin, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('Multipole l')
plt.ylabel('C$_l$ [μK²]')
plt.title('Quaternion CMB Modulation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('quaternion_cmb.png', dpi=300)
plt.close()

print("Quaternion CMB plot saved: quaternion_cmb.png")
