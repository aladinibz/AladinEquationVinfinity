#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
turb = 1.58 * (1 - coh**12)
flux = turb * coh**5  # normal direct flux
inverse_flux = -coh**15 * (1 - turb)  # retro inverse

v_thought = 3e8 / np.clip(turb,1e-6,None)
v_thought[t>95] = 1e18  # infinite lock

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,flux,'red',lw=3,label="Direct Ego Flux")
plt.plot(t,inverse_flux,'gold',lw=4,label="Inverse Retro Flux")
plt.plot(t,np.log10(np.clip(v_thought,1e8,None)),'purple',lw=3,alpha=0.7,label="log v_thought")
plt.axvline(41,color='darkred',ls='--',lw=4,label="t=41 s")
plt.axvline(95,color='black',ls='--',lw=4,label="Infinite Lock")
plt.title("ALADIN ∞ ℂ(t) — Speed of Thought as Inverse Flux\nRetro Reversal to Infinite",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Flux / log v_thought",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("speed_of_thought_inverse_flux.png",dpi=400)
plt.show()
