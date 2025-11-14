import numpy as np
import matplotlib.pyplot as plt

# DESI RSD — ξ(s) in redshift space
s = np.array([40, 60, 80, 100, 120])
xi_obs = np.array([0.12, 0.09, 0.06, 0.04, 0.03])
error = np.array([0.015, 0.012, 0.01, 0.008, 0.007])

# Aladin prediction — plasma + RSD
xi_pred = 0.15 * np.exp(-s/80) * (1 + 0.5 * (s/100))

chi2 = np.sum(((xi_obs - xi_pred)/error)**2)
dof = len(s) - 2
print(f"DESI RSD: χ²/dof = {chi2/dof:.2f}")

plt.errorbar(s, xi_obs, yerr=error, fmt='o', color='red', label='DESI RSD')
plt.plot(s, xi_pred, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('s (Mpc/h)')
plt.ylabel('ξ(s)')
plt.title('Aladin v∞ — DESI Redshift-Space Distortion')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/desi_rsd.png', dpi=300)
plt.show()
