import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data
z = np.array([0.01, 0.1, 0.5, 1, 2, 7.5])
H = np.array([75, 78, 85, 92, 110, 150])
e = np.array([2, 3, 5, 7, 12, 20])

# Model
Ha = lambda z, H0, n: H0 * (z / (1 + z)) ** n

# χ²
chi2 = lambda p: np.sum(((H - Ha(z, *p)) / e) ** 2)

# Fit
res = minimize(chi2, [75, 0.25], bounds=[(70, 80), (0.1, 0.5)])
H0, n = res.x
chi2_min = res.fun

print(f"H0={H0:.1f}, n={n:.3f}, χ²={chi2_min:.1f}")

# Plot
z_fine = np.logspace(-2, 1, 200)
H_fit = Ha(z_fine, H0, n)

plt.figure(figsize=(10, 6))
plt.errorbar(z, H, e, fmt='ko', capsize=5)
plt.plot(z_fine, H_fit, 'gold', lw=3,
         label=f'Aladin H0={H0:.0f}, n={n:.2f}')
plt.xscale('log')
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble — Full Optimization')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_full_opt.png', dpi=300)
plt.close()
