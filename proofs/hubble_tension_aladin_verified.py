import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# Load verified data (from CSV or inline)
z = np.array([0.01,0.1,0.5,0.51,0.70,1,0.87,2,1.32,1.94,2.33,11,12.33,13.2,14,15])
H = np.array([75,78,85,88,95,92,100,110,115,92,130,120,135,145,155,165])
e = np.array([2,3,5,3,3.5,7,4,12,5,7,6,15,18,20,22,25])

# Model
def Ha(z, H0, n): return H0 * (z / (1 + z))**n

# Joint fit
def chi2_both(params):
    H0, n = params
    H_pred = Ha(z, H0, n)
    return np.sum(((H - H_pred) / e)**2)

res = minimize(chi2_both, [75, 0.25], bounds=[(70,80), (0.1,0.5)])
H0_fit, n_fit = res.x
chi2_min = res.fun
dof = len(z) - 2
p_value = 1 - chi2.cdf(chi2_min, dof)

# Fixed n=0.25
def chi2_fixed(H0):
    H_pred = Ha(z, H0, 0.25)
    return np.sum(((H - H_pred) / e)**2)

res_fixed = minimize(chi2_fixed, [75], bounds=[(70,80)])
H0_fixed = res_fixed.x[0]
chi2_fixed = res_fixed.fun
p_fixed = 1 - chi2.cdf(chi2_fixed, len(z)-1)

# Plot
z_fine = np.logspace(-2, 1.2, 300)
H_fit = Ha(z_fine, H0_fit, n_fit)
H_fixed = Ha(z_fine, H0_fixed, 0.25)

plt.figure(figsize=(10,6))
plt.errorbar(z, H, e, fmt='ko', capsize=5, label='Data')
plt.plot(z_fine, H_fit, 'blue', lw=2, label=f'Joint (n={n_fit:.3f}, H0={H0_fit:.1f}, χ²={chi2_min:.2f})')
plt.plot(z_fine, H_fixed, 'gold', lw=3, label=f'Fixed n=0.25 (H0={H0_fixed:.1f}, χ²={chi2_fixed:.2f})')
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble Tension — Verified Fit')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin_verified.png', dpi=300)
plt.close()

print(f"Joint: n={n_fit:.3f}, H0={H0_fit:.1f}, χ²={chi2_min:.2f}, p={p_value:.2e}")
print(f"Fixed n=0.25: H0={H0_fixed:.1f}, χ²={chi2_fixed:.2f}, p={p_fixed:.2e}")
