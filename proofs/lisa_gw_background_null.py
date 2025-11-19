import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

f = np.logspace(-4, -1, 500)  # LISA band
h0_sq_Omega_inflation = 1e-16 * (f/1e-3)**0   # r=0.001 hope

# ALADIN: no primordial tensor → flat zero
h0_sq_Omega_aladin = np.zeros_like(f)

plt.figure(figsize=(12,8),facecolor='black')
plt.plot(f,h0_sq_Omega_inflation,color='red',lw=5,ls='--',label='Inflation r≈0.001')
plt.plot(f,h0_sq_Omega_aladin,color='gold',lw=8,label='ALADIN ∞ ℂ(t): r = 0')

plt.axvspan(1e-4,1e-1,color='cyan',alpha=0.2,label='LISA sensitivity band')
plt.loglog()
plt.xlabel('Frequency (Hz)',color='white')
plt.ylabel('h₀² Ω_GW',color='white')
plt.title('LISA (2035) — No Primordial GW Background',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(3e-4,1e-14,'LISA sensitivity: Ω_GW < 10⁻¹⁶\n'
                    '→ Will see exactly zero primordial signal\n'
                    '→ Final death of inflation',
         color='lime',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/lisa_gw_background_null.png',dpi=400,facecolor='black')
plt.close()
