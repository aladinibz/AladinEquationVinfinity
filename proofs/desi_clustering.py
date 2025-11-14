import numpy as np
import matplotlib.pyplot as plt

# DESI clustering ξ(s)
s = np.array([50, 80, 110, 140])
xi_obs = np.array([0.08, 0.05, 0.03, 0.02])
error = np.array([0.01, 0.008, 0.006, 0.005])

# Aladin prediction — plasma correlation
xi_pred = 0.1 * np.exp(-s/100)

chi2 = np.sum(((xi_obs - xi_pred)/error)**2)
dof = len(s) - 2
print(f"DESI Clustering: χ²/dof = {chi2/dof:.2f}")

plt.errorbar(s, xi_obs, yerr=error, fmt='o', color='red', label='DESI')
plt.plot(s, xi_pred, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('s (Mpc/h)')
plt.ylabel('ξ(s)')
plt.title('Aladin v∞ — DESI Clustering')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/desi_clustering.png', dpi=300)
plt.show()
