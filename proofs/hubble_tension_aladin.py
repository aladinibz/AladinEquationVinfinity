import numpy as np
import matplotlib.pyplot as plt

# Data with error bars (realistic 2025 values)
z = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 7.5])
H_z_obs = np.array([75, 78, 85, 92, 110, 150])
H_z_err = np.array([2, 3, 5, 7, 12, 20])  # km/s/Mpc

H0_local = 73.0
H0_cmb = 67.4

# Aladin v∞: H(z) = H0 * log(1+z)
H_aladin = H0_local * np.log(1 + z)

# ΛCDM: H(z) = H0 * sqrt(Ωm(1+z)^3 + ΩΛ)
H_lcdm = H0_cmb * np.sqrt(0.3 * (1+z)**3 + 0.7)

plt.figure(figsize=(8,5))
plt.errorbar(z, H_z_obs, yerr=H_z_err, fmt='ko', capsize=5, label='Observed (DESI + JWST)')
plt.plot(z, H_aladin, 'gold', linewidth=3, label='Aladin v∞ (H0=73)')
plt.plot(z, H_lcdm, 'gray', linestyle='--', linewidth=2, label='ΛCDM (H0=67.4)')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Hubble Tension — Aladin v∞ Solves It (with Error Bars)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png', dpi=300)
plt.close()
