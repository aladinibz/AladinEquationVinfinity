import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

f = np.logspace(-9,0,500)  # Hz (LISA + PTA range)

# Inflation prediction (r=0.001)
gw_inflation = 1e-15 * (f/1e-3)**0

# ALADIN — no primordial tensor modes
gw_aladin = np.zeros_like(f)

plt.figure(figsize=(12,7),facecolor='black')
plt.plot(f,gw_inflation,color='red',lw=5,ls='--',label='Inflation r≈0.001')
plt.plot(f,gw_aladin,color='gold',lw=7,label='ALADIN ∞ ℂ(t): r = 0')

plt.axvspan(1e-4,1e-1,color='gray',alpha=0.2,label='LISA band')
plt.axvspan(1e-9,1e-7,color='gray',alpha=0.1,label='PTA band')

plt.loglog()
plt.xlabel('Frequency (Hz)',color='white')
plt.ylabel('h₀² Ω_GW',color='white')
plt.title('Primordial GW Background — ALADIN Predicts ZERO',color='white',fontsize=18)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,which='both',color='gray')
plt.tick_params(colors='white')

plt.text(1e-6,1e-14,'No primordial gravitational waves from Big Bang\n'
                     'Only astrophysical background\n'
                     'LISA (2035) will see flat zero',
         color='lime',fontsize=14,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/gravitational_wave_background_null.png',dpi=400,facecolor='black')
plt.close()
