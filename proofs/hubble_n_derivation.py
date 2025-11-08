import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])

def chi2_both(params):
    H0, n = params
    H_pred = H0 * (z / (1 + z))**n
    return np.sum(((H - H_pred) / e)**2)

res = minimize(chi2_both, [75, 0.25], bounds=[(70,80), (0.1,0.5)])
H0_fit, n_fit = res.x
chi2_min = res.fun
dof = len(z) - 2
p_value = 1 - chi2.cdf(chi2_min, dof)

H_fit = H0_fit * (z / (1 + z))**n_fit

plt.figure(figsize=(9,6))
plt.errorbar(z, H, e, fmt='ko', capsize=5, label='Data')
plt.plot(z, H_fit, 'gold', lw=3, label=f'Aladin v∞ (n={n_fit:.3f})')
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble Tension — Joint H0-n Fit')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_n_exponent.png', dpi=300)
plt.close()
