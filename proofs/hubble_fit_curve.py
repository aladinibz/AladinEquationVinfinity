import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# Data
z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])

# Model: H(z) = H0 * [z/(1+z)]^0.25
def Ha(z, H0): return H0 * (z/(1+z))**0.25

# Fit
res = minimize(lambda p: np.sum(((H - Ha(z,p[0]))/e)**2), [75], bounds=[(70,80)])
H0 = res.x[0]

# Fine grid for smooth curve
z_fine = np.logspace(-2, 1, 200)
H_fit = Ha(z_fine, H0)

# Plot
plt.figure(figsize=(10,6))
plt.errorbar(z, H, e, fmt='ko', capsize=5, label='Data')
plt.plot(z_fine, H_fit, 'gold', lw=3, label=f'Aladin v∞ H0={H0:.1f}')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Hubble Fit Curve — Aladin v∞')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_fit_curve.png', dpi=300)
plt.close()
