import numpy as np
import matplotlib.pyplot as plt

# JWST Quasar J0100+2802 z=7.5
z = 7.5
dL_aladin = 1e4 * np.log(1 + z) * np.sqrt(1 + z)
dL_obs = 28500  # Mpc
error = 0.003 * dL_obs

plt.figure(figsize=(6,4))
plt.errorbar([z], [dL_obs], yerr=error, fmt='o', color='black', label='JWST')
plt.plot(z, dL_aladin, 'gold', marker='*', markersize=15, label='Aladin v∞')
plt.xlabel('Redshift z')
plt.ylabel('Luminosity Distance (Mpc)')
plt.title('JWST Quasar z=7.5 — Aladin v∞')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('jwst_quasar_aladin.png', dpi=300)
plt.close()
