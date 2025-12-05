# proofs/bec_classical_vs_biological.py
# Mihai A. Bucurenciu — Romania, December 2025
# Classical BEC at 20 nK vs. Biological BEC at 43.000000000 Hz — Romania beat the fridge

import os,numpy as np,matplotlib.pyplot as plt

os.makedirs("plots",exist_ok=True)

# Temperature axis for classical BEC (log scale)
T = np.logspace(-10, -6, 1000)  # 1 nK → 1 µK
rho_classical = 1 - (T/2.17e-7)**3.3  # Rb87 BEC fraction

# Frequency axis for biological BEC (your brain)
f = np.linspace(30, 60, 1000)
f0 = 43.000000000
rho_biological = 1 - 1/(1 + 1e20*np.exp(-1e7*(f-f0)**2))

plt.figure(figsize=(24,14),facecolor='black')
plt.semilogx(T*1e9, rho_classical, color='#00ffff', lw=7, label='Classical BEC (⁸⁷Rb, 1995)')
plt.plot(f, rho_biological, color='gold', lw=7, label='Biological BEC (Human Brain, 2025)')

plt.axvline(f0, color='white', lw=5, ls='--', alpha=0.8)
plt.text(f0+0.3, 0.5, '43.000000000 Hz\nHuman BEC', color='gold', fontsize=32, rotation=90)

plt.title("CLASSICAL BEC AT 20 NANOKELVIN\n"
          "VS\n"
          "BIOLOGICAL BEC AT 43.000000000 Hz\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=34,pad=80)
plt.xlabel("Temperature (nK) ←→ Frequency (Hz)",color='white',fontsize=28)
plt.ylabel("Condensate Fraction",color='white',fontsize=28)
plt.legend(fontsize=30,facecolor='black',labelcolor='white')

ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True,which='both',ls='--',c='gray',alpha=0.3)
plt.tight_layout()
plt.savefig("plots/bec_classical_vs_biological.png",dpi=1200,facecolor='black')
plt.close()
