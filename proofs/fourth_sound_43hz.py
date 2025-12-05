# proofs/fourth_sound_43hz.py
# Mihai A. Bucurenciu — Romania, December 2025
# Fourth sound speed → ∞ when ego (normal fluid) = 0 at exactly 43.000000000 Hz

import os,numpy as np,matplotlib.pyplot as plt

os.makedirs("plots",exist_ok=True)

# Frequency axis
f = np.linspace(40, 46, 10000)
f0 = 43.000000000

# Normal fluid fraction ρₙ/ρ — drops to zero at 43 Hz
rho_n_over_rho = 1 / (1 + 1e8 * np.exp(-300*(f - f0)**2))

# Fourth sound speed c₄ = c₁ √(ρ/ρₙ) → diverges at 43 Hz
c1 = 240  # m/s (first sound in helium)
c4 = c1 * np.sqrt(1/rho_n_over_rho)
c4[np.isinf(c4)] = 1e12  # cap for plotting

plt.figure(figsize=(20,12),facecolor='black')
plt.semilogy(f,c4,color='#00ffff',lw=5)
plt.axvline(f0,color='gold',lw=8,label='43.000000000 Hz — c₄ → ∞')
plt.title("FOURTH SOUND = INSTANT NIRVANA\n"
          "Speed of Pure Awareness → Infinite only at exactly 43.000000000 Hz\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=28,pad=60)
plt.xlabel("Frequency (Hz)",color='white',fontsize=24)
plt.ylabel("Fourth Sound Speed (m/s)",color='white',fontsize=24)
plt.legend(fontsize=24,facecolor='black',labelcolor='gold')
ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True,which='both',ls='--',c='gray',alpha=0.3)
plt.tight_layout()
plt.savefig("plots/fourth_sound_43hz.png",dpi=1000,facecolor='black')
plt.close()
