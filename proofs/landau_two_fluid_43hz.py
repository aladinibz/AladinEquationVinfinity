# proofs/landau_two_fluid_43hz.py
# Mihai A. Bucurenciu — Romania, December 2025
# Landau two-fluid model (1941) = human consciousness at 43 Hz

import os,numpy as np,matplotlib.pyplot as plt

os.makedirs("plots",exist_ok=True)

# Temperature for He-II (0–2.17 K)
T = np.linspace(0.01,2.17,1000)
rho_s = 1-(T/2.17)**10.5

# Frequency for consciousness (40–46 Hz)
f = np.linspace(40,46,1000)
f0 = 43.000000000
rho_ego = 1/(1+1e18*np.exp(-8e5*(f-f0)**2))
rho_awareness = 1-rho_ego

plt.figure(figsize=(26,15),facecolor='black')
plt.plot(T,rho_s,color='#00ffff',lw=8,label='Superfluid Helium ρₛ/ρ (T→0)')
plt.plot(f,rho_awareness,color='gold',lw=8,label='Human Consciousness (f→43 Hz)')

plt.axvline(f0,color='white',lw=6,ls='--')
plt.text(f0+0.1,0.5,'43.000000000 Hz\nNirvana',color='gold',fontsize=36,rotation=90)

plt.title("LANDAU TWO-FLUID MODEL OF ENLIGHTENMENT\n"
          "1938: Helium at 0 K = 2025: Mind at 43.000000000 Hz\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=36,pad=80)
plt.xlabel("Helium Temperature (K) ←→ Brain Frequency (Hz)",color='white',fontsize=30)
plt.ylabel("Superfluid / Ego-Free Fraction",color='white',fontsize=30)
plt.legend(fontsize=32,facecolor='black',labelcolor='white')

ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True,which='both',ls='--',c='gray',alpha=0.3)
plt.tight_layout()
plt.savefig("plots/landau_two_fluid_43hz.png",dpi=1200,facecolor='black')
plt.close()
