import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])

Ha = lambda z,H0: H0*np.log(1+z)
Hl = lambda z,H0,Om: H0*np.sqrt(Om*(1+z)**3 + (1-Om))

ca = lambda p: np.sum(((H - Ha(z,p[0]))/e)**2)
cl = lambda p: np.sum(((H - Hl(z,*p))/e)**2)

ra = minimize(ca,[73],bounds=[(50,90)])
H0a = ra.x[0]; ca2 = ra.fun; pa = 1-chi2.cdf(ca2,len(z)-1)

rl = minimize(cl,[67.4,0.3],bounds=[(50,90),(0.1,0.5)])
H0l,Oml = rl.x; cl2 = rl.fun; pl = 1-chi2.cdf(cl2,len(z)-2)

err_a = np.sqrt(1 / np.sum((1/e**2) * (np.log(1+z))**2))
Ha_low = (H0a - err_a) * np.log(1 + z)
Ha_up = (H0a + err_a) * np.log(1 + z)

dH_dH0 = Hl(z,H0l,Oml) / H0l
dH_dOm = H0l * (1+z)**3 / (2 * Hl(z,H0l,Oml))
F11 = np.sum(dH_dH0**2 / e**2)
F12 = np.sum(dH_dH0 * dH_dOm / e**2)
F22 = np.sum(dH_dOm**2 / e**2)
F = np.array([[F11,F12],[F12,F22]])
cov = np.linalg.inv(F)
err_H0l = np.sqrt(cov[0,0])
Hl_low = Hl(z,H0l-err_H0l,Oml)
Hl_up = Hl(z,H0l+err_H0l,Oml)

plt.figure(figsize=(9,6))
plt.errorbar(z,H,e,fmt='ko',capsize=5,label='Data')
plt.plot(z,Ha(z,H0a),'gold',lw=3,label=f'Aladin (H0={H0a:.0f}, p={pa:.1e})')
plt.fill_between(z,Ha_low,Ha_up,color='gold',alpha=0.5,edgecolor='gold',linewidth=1.5,label='Aladin 68%')
plt.plot(z,Hl(z,H0l,Oml),'gray',ls='--',lw=2,label=f'ΛCDM (H0={H0l:.0f}, Ωm={Oml:.1f}, p={pl:.1e})')
plt.fill_between(z,Hl_low,Hl_up,color='gray',alpha=0.5,edgecolor='gray',linewidth=1.5,label='ΛCDM 68%')
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble Tension — CI with Edges')
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png',dpi=300,bbox_inches='tight')
plt.close()
