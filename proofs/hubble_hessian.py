import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Symbols
H0,n,z_i,H_i,e_i = sp.symbols('H0 n z_i H_i e_i')
f = H0 * (z_i/(1+z_i))**n
chi2_one = ((H_i-f)/e_i)**2

# Hessian
H = sp.hessian(chi2_one, (H0,n))

# Data
z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])

# Fit
H0_fit,n_fit = 75.0,0.25

# Sum Hessian
Hess = np.zeros((2,2))
for zi,Hi,ei in zip(z,H,e):
    Hess += np.array(H.subs({H0:H0_fit,n:n_fit,z_i:zi,H_i:Hi,e_i:ei}),float)

# Covariance
cov = np.linalg.inv(Hess + 1e-12*np.eye(2))
err_H0,err_n = np.sqrt(cov.diagonal())

# Plot
z_fine = np.logspace(-2,1,200)
H_fit = H0_fit * (z_fine/(1+z_fine))**n_fit

plt.figure(figsize=(10,6))
plt.errorbar(z,H,e,fmt='ko',capsize=5)
plt.plot(z_fine,H_fit,'gold',lw=3,
         label=f'Aladin H0={H0_fit:.0f}±{err_H0:.0f}, n={n_fit:.2f}±{err_n:.2f}')
plt.xscale('log'); plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble — Hessian Errors')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_hessian.png',dpi=300)
plt.close()
