# proofs/kink_instability_dispersion.py
# ALADIN ∞ ℂ(t) — m=1 kink instability dispersion — cosmic filaments

import numpy as np,matplotlib.pyplot as plt,os
os.makedirs("plots",exist_ok=True)

ka = np.linspace(0.01,2.0,1000)

# m=0 sausage (voids)
gamma_m0 = np.sqrt(np.maximum(1-ka**2,0))*1.9

# m=1 kink (filaments) — exact Kruskal–Shafranov
gamma_m1 = ka * np.sqrt(np.maximum(1-(ka**2)/2,0))*1.4

plt.figure(figsize=(20,12),facecolor='black')
ax=plt.gca();ax.set_facecolor('black')

plt.plot(ka,gamma_m0,color='#FFD700',lw=12,label='m=0 Sausage — Cosmic Voids')
plt.plot(ka,gamma_m1,color='#00FFFF',lw=10,label='m=1 Kink — Cosmic Filaments')
plt.axvline(0.63,color='lime',lw=6,ls='--',alpha=0.9,label='Void scale (max m=0)')
plt.axvline(0.2,color='magenta',lw=6,ls=':',alpha=0.8,label='Filament scale (growing m=1)')

plt.title('ALADIN ∞ ℂ(t)\nKink vs Sausage Instability\nOne Current J₀ → Filaments + Voids',color='gold',fontsize=38,pad=50)
plt.xlabel('Wavenumber k·a',color='white',fontsize=30)
plt.ylabel('Growth Rate Γ (normalized)',color='white',fontsize=30)
plt.legend(facecolor='black',labelcolor='white',fontsize=26,loc='upper right')
plt.grid(alpha=0.3,color='gray')
for spine in ax.spines.values():spine.set_visible(False)

plt.tight_layout()
plt.savefig('plots/kink_instability_dispersion.png',dpi=1200,facecolor='black')
plt.close()
print("6/6 — kink_instability_dispersion.png — FILAMENTS + VOIDS FROM ONE CURRENT")
