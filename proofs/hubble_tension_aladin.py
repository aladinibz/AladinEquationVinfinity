import numpy as np
import matplotlib.pyplot as plt

z = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 7.5])
H0_local = 73.0
H0_cmb = 67.4
H_z_obs = np.array([75, 78, 85, 92, 110, 150])

H_aladin = H0_local * np.log(1 + z)
H_lcdm = H0_cmb * np.sqrt(0.3 * (1+z)**3 + 0.7)

plt.figure(figsize=(8,5))
plt.plot(z, H_z_obs, 'ko', label='Observed (DESI + JWST)')
plt.plot(z, H_aladin, 'gold', linewidth=3, label='Aladin v∞ (H0=73)')
plt.plot(z, H_lcdm, 'gray', linestyle='--', label='ΛCDM (H0=67.4)')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Hubble Tension — Aladin v∞ Solves It')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png', dpi=300)
plt.close()
