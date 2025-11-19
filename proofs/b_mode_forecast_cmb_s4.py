import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

ell = np.arange(2,300)
BB_lensing = 5e-6 * (ell/100)**(-2.5)

# Inflation r=0.001 (best hope)
BB_r0001 = 2e-8 * (ell/80)**(-2)

# ALADIN: no primordial tensor → r = 0
BB_aladin = np.zeros_like(ell)

plt.figure(figsize=(12,8),facecolor='black')
plt.plot(ell,BB_lensing,color='gray',lw=4,label='Lensing B-modes')
plt.plot(ell,BB_r0001,color='red',ls='--',lw=4,label='Inflation r=0.001')
plt.plot(ell,BB_aladin,color='gold',lw=7,label='ALADIN ∞ ℂ(t): r = 0')

plt.axvspan(30,200,color='cyan',alpha=0.2,label='CMB-S4 sweet spot')
plt.yscale('log')
plt.ylim(1e-9,1e-5)
plt.xlabel('Multipole ℓ',color='white')
plt.ylabel('BB power [μK²]',color='white')
plt.title('CMB-S4 (2028) Will See r = 0 — ALADIN Prediction',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(80,1e-7,'CMB-S4 sensitivity: σ(r) ≈ 0.0005\n'
                   '→ Will measure r = 0.000 ± 0.001\n'
                   '→ End of cosmic inflation',
         color='lime',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/b_mode_forecast_cmb_s4.png',dpi=400,facecolor='black')
plt.close()
