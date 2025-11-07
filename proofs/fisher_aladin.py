import numpy as np
import matplotlib.pyplot as plt

# Fisher matrix demo
z = np.array([0.01,0.1,0.5,1,2,7.5])
H_obs = np.array([75,78,85,92,110,150])
err = np.array([2,3,5,7,12,20])

H0 = 73.0
H_aladin = H0 * np.log(1 + z)
dH_dH0 = np.log(1 + z)
F = np.sum(dH_dH0**2 / err**2)
sigma_H0 = 1 / np.sqrt(F)

print(f"Fisher: H0 = {H0:.1f} ± {sigma_H0:.1f}")

plt.figure(figsize=(6,4))
plt.errorbar(z, H_obs, err, fmt='ko', capsize=5)
plt.plot(z, H_aladin, 'gold', lw=3)
plt.fill_between(z, (H0-sigma_H0)*np.log(1+z), (H0+sigma_H0)*np.log(1+z), color='gold', alpha=0.5)
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Fisher Matrix — Aladin CI')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fisher_aladin.png', dpi=300)
plt.close()
