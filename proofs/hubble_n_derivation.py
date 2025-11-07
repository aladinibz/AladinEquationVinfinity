import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

# Data
z_data = np.array([0.01,0.1,0.5,1,2,7.5])
H_data = np.array([75,78,85,92,110,150])
H_err = np.array([2,3,5,7,12,20])

# Fit n: H(z) = H0 * (1+z)^n
def chi2_n(n_val):
    H0 = H_data[0] / (1 + z_data[0])**n_val
    H_pred = H0 * (1 + z_data)**n_val
    return np.sum(((H_data - H_pred) / H_err)**2)

res = minimize_scalar(chi2_n, bounds=(0.1, 0.5))
n_fit = res.x
chi2_min = res.fun
dof = len(z_data) - 1
p_val = 1 - chi2.cdf(chi2_min, dof)

H0_fit = H_data[0] / (1 + z_data[0])**n_fit
H_fit = H0_fit * (1 + z_data)**n_fit

print(f"n = {n_fit:.3f}, H0 = {H0_fit:.1f}, χ² = {chi2_min:.2f}, p = {p_val:.2e}")

plt.figure(figsize=(9,6))
plt.errorbar(z_data, H_data, H_err, fmt='ko', capsize=5, label='Data')
plt.plot(z_data, H_fit, 'gold', lw=3, label=f'Aladin v∞ (n={n_fit:.2f})')
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble Tension — n Exponent Derived')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_n_exponent.png', dpi=300)
plt.close()
