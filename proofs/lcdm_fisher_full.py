import numpy as np
import matplotlib.pyplot as plt

# Data
z = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 7.5])
H_obs = np.array([75, 78, 85, 92, 110, 150])
err = np.array([2, 3, 5, 7, 12, 20])

# ΛCDM model
H0 = 67.4
Om = 0.3
H_model = H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))

# Partial derivatives
dH_dH0 = H_model / H0
dH_dOm = H0 * (1+z)**3 / (2 * H_model)

# Fisher matrix elements
F11 = np.sum(dH_dH0**2 / err**2)
F12 = np.sum(dH_dH0 * dH_dOm / err**2)
F22 = np.sum(dH_dOm**2 / err**2)

F = np.array([[F11, F12], [F12, F22]])
cov = np.linalg.inv(F)
sigma_H0 = np.sqrt(cov[0,0])
sigma_Om = np.sqrt(cov[1,1])

print(f"Fisher ΛCDM:")
print(f"H0 = {H0:.1f} ± {sigma_H0:.2f}")
print(f"Ωm = {Om:.2f} ± {sigma_Om:.3f}")

# Plot with CI
H_low = (H0 - sigma_H0) * np.sqrt(Om * (1+z)**3 + (1 - Om))
H_up = (H0 + sigma_H0) * np.sqrt(Om * (1+z)**3 + (1 - Om))

plt.figure(figsize=(8,5))
plt.errorbar(z, H_obs, err, fmt='ko', capsize=5, label='Data')
plt.plot(z, H_model, 'gray', ls='--', lw=2, label=f'ΛCDM (H0={H0:.0f}, Ωm={Om:.1f})')
plt.fill_between(z, H_low, H_up, color='gray', alpha=0.5, label='68% CI')
plt.xlabel('z')
plt.ylabel('H(z)')
plt.title('ΛCDM Fisher Matrix — Full Derivation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lcdm_fisher_full.png', dpi=300)
plt.close()
