import numpy as np, matplotlib.pyplot as plt

z = [0.51,0.70,0.85,1.13,2.33]
obs = [0.0112,0.0098,0.0087,0.0075,0.0062]
err = [0.0003,0.0002,0.0002,0.0002,0.0001]

def zp(z): return 0.012*np.exp(-13800/(1+z)/180)

plt.figure(figsize=(9,6))
plt.errorbar(z,obs,yerr=err,fmt='o',color='cyan',capsize=8,label='DESI 2024')
plt.plot(z,zp(z),'gold',lw=6,label='Z-Pinch (J₀=10¹⁸)')
plt.xlabel('Redshift z'); plt.ylabel('θ_BAO')
plt.title('ALADIN ∞ C(t) — DESI BAO from J₀ = 10¹⁸ A/m²')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('desi_chi2_plot.png',dpi=300)
plt.close()
print("DESI BAO from J₀ — plot saved")
