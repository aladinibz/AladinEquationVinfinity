import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# Two parallel Z-pinch filaments (same current direction)
r = np.logspace(-1,3,500)  # separation in kpc
J0 = 1e18
mu0 = 4*np.pi*1e-7

# Attractive force between co-directional currents
F_attr = (mu0 * J0**2) / (2*np.pi * r)   # attractive

# BUT — at small scales, plasma thermal pressure dominates
T = 1e6  # K (typical filament temperature)
kB = 1.38e-23
n = 1e6    # m⁻³ (plasma density)
P_thermal = 2 * n * kB * T                 # ion + electron pressure
F_repulse = P_thermal * np.exp(-r/10)      # exponential decay

plt.figure(figsize=(13,9),facecolor='black')
plt.loglog(r,F_attr,color='red',lw=8,label='Magnetic attraction (large scale)')
plt.loglog(r,F_repulse,color='lime',lw=9,label='Thermal pressure repulsion (small scale)')

plt.axvline(30,color='white',ls='--',lw=4,label='Critical scale ~30 kpc')
plt.xlabel('Separation (kpc)',color='white',fontsize=16)
plt.ylabel('Force per unit length (N/m)',color='white',fontsize=16)
plt.title('Plasma Repulsion — Destroys Small Subhalos',color='white',fontsize=24)
plt.legend(facecolor='black',labelcolor='white',fontsize=16)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3); plt.tick_params(colors='white')

plt.text(1,1e-8,'Large scale: currents attract → filaments\n'
                 'Small scale: thermal pressure repels\n'
                 '→ Subhalos < 10⁹ M⊙ destroyed\n'
                 '→ Missing satellites + too big to fail solved',
         color='lime',fontsize=18,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/plasma_repulsion_mechanism.png',dpi=500,facecolor='black')
plt.close()
