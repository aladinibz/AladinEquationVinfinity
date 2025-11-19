import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

ell = np.arange(2,251)
BB_inflation = 1e-4 * (ell/100)**-2.5      # typical r=0.001 prediction
BB_aladin = np.zeros_like(ell)             # ALADIN: no primordial tensor modes

plt.figure(figsize=(12,7),facecolor='black')
plt.plot(ell,BB_inflation,color='red',lw=4,ls='--',label='Inflation (r≈0.001)')
plt.plot(ell,BB_aladin,color='gold',lw=7,label='ALADIN ∞ ℂ(t): r = 0')

plt.yscale('log')
plt.xlabel('Multipole ℓ',color='white')
plt.ylabel('B-mode power [μK²]',color='white')
plt.title('Primordial B-modes — ALADIN Predicts r = 0',color='white',fontsize=18)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(100,1e-5,'No primordial gravitational waves\nOnly lensing B-modes remain\n'
                   'CMB-S4 + LiteBIRD will see exactly zero',
         color='lime',fontsize=14,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/b_mode_null_prediction.png',dpi=400,facecolor='black')
plt.close()
