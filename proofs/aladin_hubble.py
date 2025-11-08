import numpy as np
import matplotlib.pyplot as plt

# Aladin v∞ parameters
H0 = 75.2          # km/s/Mpc
n = 0.25           # plasma-derived

# Redshift grid
z = np.logspace(-2, 1, 300)

# H(z) = H0 * [z/(1+z)]^n
H_fit = H0 * (z / (1 + z))**n

# Plot
plt.figure(figsize=(10,6))
plt.plot(z, H_fit, 'gold', lw=3, label=f'Aladin v∞ H0={H0}, n={n}')
plt.xscale('log')
plt.xlabel('z')
plt.ylabel('H(z)')
plt.title('Aladin v∞ — Hubble Fit')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('aladin_hubble.png', dpi=300)
plt.close()

print("Plot saved")
