import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# Load data
z_low = np.array([0.01,0.1,0.5,1,2])
H_low = np.array([75,78,85,92,110])
e_low = np.array([2,3,5,7,12])
z_jwst = np.array([11.0,12.33,13.2,14.0,15.0])
H_jwst = np.array([120,135,145,155,165])
e_jwst = np.array([15,18,20,22,25])
z = np.concatenate([z_low,z_jwst])
H = np.concatenate([H_low,H_jwst])
e = np.concatenate([e_low,e_jwst])

# Models
Ha = lambda z,H0: H0 * (z/(1+z))**0.25
Hl = lambda z,H0,Om: H0 * np.sqrt(Om*(1+z)**3 + (1-Om))

# Fit
ra = minimize(lambda p: np.sum(((H - Ha(z,p[0]))/e)**2), [75], bounds=[(70,80)])
H0a = ra.x[0]; ca2 = ra.fun; dof_a = len(z)-1; pa = 1-chi2.cdf(ca2,dof_a)

rl = minimize(lambda p: np.sum(((H - Hl(z,*p))/e)**2), [67.4,0.3], bounds=[(50,90),(0.1,0.5)])
H0l,Oml = rl.x; cl2 = rl.fun; dof_l = len(z)-2; pl = 1-chi2.cdf(cl2,dof_l)

# Predictions
Ha_fit = Ha(z,H0a)
Hl_fit = Hl(z,H0l,Oml)

# Normalized residuals
res_a_norm = (H - Ha_fit) / e
res_l_norm = (H - Hl_fit) / e

# Plot
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8),sharex=True,gridspec_kw={'height_ratios':[3,1]})

# Main plot
ax1.errorbar(z[:5],H[:5],e[:5],fmt='ko',capsize=5)
ax1.errorbar(z[5:],H[5:],e[:5],fmt='ks',capsize=5)
ax1.plot(z,Ha_fit,'gold',lw=3,label=f'Aladin H0={H0a:.0f}')
ax1.plot(z,Hl_fit,'gray',ls='--',lw=2,label=f'ΛCDM H0={H0l:.0f}')
ax1.set_ylabel('H(z)')
ax1.legend()
ax1.grid(alpha=0.3)

# Residuals
ax2.errorbar(z,res_a_norm,e/e,fmt='ko',capsize=5)
ax2.errorbar(z,res_l_norm,e/e,fmt='ks',capsize=5)
ax2.axhline(0,color='k',ls=':')
ax2.axhline(1,color='k',ls=':',alpha=0.5)
ax2.axhline(-1,color='k',ls=':',alpha=0.5)
ax2.set_xlabel('z'); ax2.set_ylabel('Normalized Residuals')
ax2.grid(alpha=0.3)

# Chi-squared text
text = f'Aladin: χ²={ca2:.2f}, p={pa:.1e}\nΛCDM: χ²={cl2:.2f}, p={pl:.1e}'
ax1.text(0.02,0.98,text,transform=ax1.transAxes,va='top',fontsize=9,
         bbox=dict(boxstyle="round",facecolor='wheat',alpha=0.8))

plt.suptitle('Hubble Tension — Goodness-of-Fit')
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png',dpi=300)
plt.close()
