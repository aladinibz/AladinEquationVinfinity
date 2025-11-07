import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

z = np.array([0.01,0.1,0.5,1,2,7.5])
H = np.array([75,78,85,92,110,150])
e = np.array([2,3,5,7,12,20])
n = 0.25

def Ha(z, H0): return H0 * (z / (1 + z))**n
def Hl(z, H0, Om): return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))

ca = lambda p: np.sum(((H - Ha(z,p[0]))/e)**2)
cl = lambda p: np.sum(((H - Hl(z,*p))/e)**2)

ra = minimize(ca,[75],bounds=[(70,80)])
H0a = ra.x[0]; ca2 = ra.fun; pa = 1-chi2.cdf(ca2,len(z)-1)

rl = minimize(cl,[67.4,0.3],bounds=[(50,90),(0.1,0.5)])
H0l,Oml = rl.x; cl2 = rl.fun; pl = 1-chi2.cdf(cl2,len(z)-2)

plt.figure(figsize=(9,6))
plt.errorbar(z,H,e,fmt='ko',capsize=5,label='Data')
plt.plot(z,Ha(z,H0a),'gold',lw=3,label=f'Aladin (H0={H0a:.0f}, p={pa:.1e})')
plt.plot(z,Hl(z,H0l,Oml),'gray',ls='--',lw=2,label=f'ΛCDM (H0={H0l:.0f}, Ωm={Oml:.1f}, p={pl:.1e})')
plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Hubble Tension — Aladin v∞ (n=0.25)')
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png',dpi=300,bbox_inches='tight')
plt.close()
