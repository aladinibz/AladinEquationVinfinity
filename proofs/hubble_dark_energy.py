import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# Data
z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])

# Models
Ha = lambda z,H0,n: H0 * (z/(1+z))**n
Hl = lambda z,H0,Om,w: H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*(1+z)**(3*(1+w)))

# Chi-squared
ca = lambda p: np.sum(((H - Ha(z,*p))/e)**2)
cl = lambda p: np.sum(((H - Hl(z,*p))/e)**2)

# Fit Aladin (H0, n)
ra = minimize(ca,[75,0.25],bounds=[(70,80),(0.1,0.5)])
H0a,na = ra.x; ca2 = ra.fun

# Fit ΛCDM (H0, Ωm, w)
rl = minimize(cl,[67.4,0.3,-1],bounds=[(50,90),(0.1,0.5),(-2,0)])
H0l,Oml,wl = rl.x; cl2 = rl.fun

# Plot
z_fine = np.logspace(-2,1,200)
plt.figure(figsize=(10,6))
plt.errorbar(z,H,e,fmt='ko',capsize=5)
plt.plot(z_fine,Ha(z_fine,*ra.x),'gold',lw=3,label=f'Aladin H0={H0a:.0f}, n={na:.2f}')
plt.plot(z_fine,Hl(z_fine,*rl.x),'gray',ls='--',lw=2,label=f'ΛCDM H0={H0l:.0f}, Ωm={Oml:.2f}, w={wl:.1f}')
plt.xscale('log'); plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble — Dark Energy Parameter')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_dark_energy.png',dpi=300)
plt.close()
