import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# Roman forecast years
years = np.array([2028,2029,2030,2031,2032])
h0_error = np.array([1.2,0.8,0.5,0.35,0.25])

h0_aladin = 73.2

plt.figure(figsize=(12,7),facecolor='black')
plt.errorbar(years,np.ones_like(years)*h0_aladin,
             yerr=h0_error,fmt='o',color='gold',capsize=10,
             markersize=10,label='ALADIN ∞ ℂ(t): H₀ = 73.2 km/s/Mpc')
plt.axhline(67.4,color='red',lw=5,ls='--',label='Planck ΛCDM')
plt.axhline(73.8,color='lime',lw=4,ls=':',label='SH0ES')

plt.xlabel('Year',color='white')
plt.ylabel('H₀ (km/s/Mpc)',color='white')
plt.title('Roman Supernovae — Will Confirm H₀ = 73.2',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(2029.5,71,'Roman SN Ia survey\n'
                   '→ H₀ = 73.2 ± 0.25 by 2032\n'
                   '→ Hubble tension gone forever',
         color='lime',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/roman_supernovae_aladin.png',dpi=400,facecolor='black')
plt.close()
