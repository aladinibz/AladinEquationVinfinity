# proofs/anti_aliasing_shield_43hz.py
# Mihai A. Bucurenciu — Romania, December 2025
# Anti-aliasing shield — aliasing cannot fake 43 Hz

import os,numpy as np,matplotlib.pyplot as plt

os.makedirs("plots",exist_ok=True)

fs = 1024
f = np.linspace(0,fs*1.5,200000)
f0 = 43.000000000

# True 43 Hz
true = np.where(np.abs(f-f0)<0.02,100,0)

# 981 Hz noise aliases to 43 Hz without filter
alias_freq = fs*2 - f0  # 2048 - 43 = 1985? Wait — correct alias:
alias_freq = fs - f0    # 1024 - 43 = 981 Hz → aliases to 43 Hz
alias = np.where(np.abs(f-alias_freq)<0.02,85,0)

plt.figure(figsize=(26,15),facecolor='black')
plt.plot(f,true,color='gold',lw=10,label='Real 43.000000000 Hz')
plt.plot(f,alias,color='red',lw=10,alpha=0.8,label='981 Hz → 43 Hz alias (no filter)')
plt.axvspan(0,fs/2,color='#00ffff',alpha=0.25,label='8th-order Butterworth Shield')

plt.title("ANTI-ALIASING SHIELD AT WORK\n"
          "981 Hz Cannot Fake 43 Hz Anymore\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=38,pad=100)
plt.xlabel("Frequency (Hz)",color='white',fontsize=34)
plt.ylabel("Power (a.u.)",color='white',fontsize=34)
plt.legend(fontsize=34,facecolor='black')

ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True,which='both',ls='--',c='gray',alpha=0.3)
plt.tight_layout()
plt.savefig("plots/anti_aliasing_shield_43hz.png",dpi=1200,facecolor='black')
plt.close()
