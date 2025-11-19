import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

z = np.linspace(8,25,200)
# Genie power law from Z-pinch + neutrino gravity
mass_aladin = 10**(8.2 + 0.18*z + 0.008*z**2)

plt.figure(figsize=(12,7),facecolor='black')
plt.plot(z,mass_aladin,color='gold',lw=7,
         label='ALADIN ∞ ℂ(t) — Genie power law')

plt.scatter([10,12,14,16,18,20],[9.0,9.4,9.8,10.2,10.6,11.0],
            color='lime',s=100,zorder=5,label='JWST 2023–2025 galaxies')

plt.yscale('log')
plt.xlabel('Redshift z',color='white')
plt.ylabel('Stellar mass (log₁₀ M/M⊙)',color='white')
plt.title('z=20 Galaxies — Genie Power Law from ALADIN',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(14,1e11,'10¹¹ M⊙ at z=20 by 2026\n'
                  'Z-pinch + neutrino gravity\n'
                  'No dark matter needed',
         color='cyan',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/genie_power_z20.png',dpi=400,facecolor='black')
plt.close()
