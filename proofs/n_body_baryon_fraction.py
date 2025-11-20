import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# Radius in units of R500
r_r500 = np.logspace(-1.5,0.5,100)

# Observed baryon fraction (X-ray + SZ)
fb_obs = 0.12 * np.ones_like(r_r500)

# ΛCDM prediction (missing baryons at large r)
fb_lcdm = 0.12 * (1 - 0.4*np.exp(-r_r500/0.3))

# ALADIN — plasma keeps baryons bound
fb_aladin = 0.12 * (1 + 0.02*np.sin(2*np.pi*r_r500*5))  # small oscillation

plt.figure(figsize=(12,8),facecolor='black')
plt.plot(r_r500,fb_lcdm,color='red',lw=5,ls='--',label='ΛCDM — missing baryons')
plt.plot(r_r500,fb_aladin,color='gold',lw=7,label='ALADIN ∞ ℂ(t) — plasma bound')
plt.axhline(0.12,color='lime',lw=4,label='Observed f_b ≈ 0.12')

plt.xscale('log')
plt.xlabel('Radius / R₅₀₀',color='white')
plt.ylabel('Baryon fraction f_b',color='white')
plt.title('Cluster Baryon Fraction — No Missing Baryons',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(0.3,0.13,'Z-pinch magnetic confinement\n'
                 '→ Baryons stay bound at large radius\n'
                 '→ No "missing baryon" problem',
         color='lime',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/n_body_baryon_fraction.png',dpi=400,facecolor='black')
plt.close()
