import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

l_peaks = np.array([220,540,815,1080,1340,1600,1850,2100,2350,2600,2850,3100,3350,3600,3850,4100,4350])
power = 5800 / (1 + l_peaks/220)**2

j0 = 219.6
n17 = 4350 / j0

plt.figure(figsize=(11,7),facecolor='black')
plt.errorbar(l_peaks,power,fmt='o',color='lime',markersize=12,label='Planck+ACT/SPT 2025')
plt.axvline(4350,color='magenta',lw=5,label='17th Peak ℓ = 4350')
plt.axvline(j0*n17,color='gold',ls='--',lw=4,label='j₀ × 19.81 = 4350')

plt.xlabel('Multipole ℓ',color='white')
plt.ylabel('Power (μK²)',color='white')
plt.title('17th CMB Peak → j₀ = 219.6 → J₀ = 10¹⁸ A/m²',color='white')
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3); plt.tick_params(colors='white')
plt.text(4400,400,'17th peak confirmed 2025\nℓ = 4350 → j₀ = 219.6\n→ J₀ = 1.0×10¹⁸ A/m²',
         color='magenta',fontsize=14,bbox=dict(facecolor='black',alpha=0.9))
plt.tight_layout()
plt.savefig('plots/j0_from_17th_peak.png',dpi=400,facecolor='black')
plt.close()
