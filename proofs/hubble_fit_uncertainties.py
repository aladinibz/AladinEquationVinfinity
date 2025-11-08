import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

# Data
z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])

# Model
Ha = lambda z,H0,n: H0 * (z/(1+z))**n

# Fit
def chi2(p): return np.sum(((H - Ha(z,*p))/e)**2)
res = minimize(chi2,[75,0.25],bounds=[(70,80),(0.1,0.5)])
H0,n = res.x

# Hessian
eps = 1e-6
Hess = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        p = np.array(res.x)
        p[i] += eps; p[j] += eps; f_pp = chi2(p)
        p[j] -= 2*eps; f_pm = chi2(p)
        p[i] -= 2*eps; f_mm = chi2(p)
        p[j] += 2*eps; f_mp = chi2(p)
        Hess[i,j] = (f_pp - f_pm - f_mp + f_mm) / (4*eps**2)
cov = np.linalg.inv(Hess)
err_H0, err_n = np.sqrt(cov.diagonal())

print(f"H0 = {H0:.1f} ± {err_H0:.1f}")
print(f"n  = {n:.3f} ± {err_n:.3f}")

# Plot
z_fine = np.logspace(-2,1,200)
plt.figure(figsize=(10,6))
plt.errorbar(z,H,e,fmt='ko',capsize=5)
plt.plot(z_fine,Ha(z_fine,H0,n),'gold',lw=3,
         label=f'Aladin H0={H0:.0f}±{err_H0:.0f}, n={n:.2f}±{err_n:.2f}')
plt.xscale('log'); plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble Fit — Uncertainties')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_fit_uncertainties.png',dpi=300)
plt.close()
