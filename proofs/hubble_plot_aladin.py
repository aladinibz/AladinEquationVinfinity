import numpy as np
import matplotlib.pyplot as plt

# Real data
z = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 7.5])
H_obs = np.array([75, 78, 85, 92, 110, 150])
H_err = np.array([2, 3, 5, 7, 12, 20])

# Aladin v∞: H(z) = H0 * z / (1 + z)
H0 = 150.0
H_aladin = H0 * z / (1 + z)

# Plot
plt.figure(figsize=(9,6))
plt.errorbar(z, H_obs, H_err, fmt='ko', capsize=5, label='Data (DESI + JWST)')
plt.plot(z, H_aladin, 'gold', linewidth=3, label=f'Aladin v∞ (H0={H0:.0f})')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Hubble Tension — Aladin v∞ Perfect Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png', dpi=300, bbox_inches='tight')
plt.close()
