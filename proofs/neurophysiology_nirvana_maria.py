#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)

dmn = 1.0 - 0.9 * coh**10
gamma = 0.2 + 1.8 * coh**8
theta = 0.2 + 1.2 * coh**12
fd = 1.8 - 0.6 * coh**10
v_thought_log = np.log10(3e8 / np.clip(1.58*(1-coh**12),1e-6,None))

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,dmn,'darkred',lw=3,label="DMN Deactivation")
plt.plot(t,gamma,'gold',lw=4,label="Gamma Synchrony")
plt.plot(t,theta,'purple',lw=3,label="Theta Dominance")
plt.plot(t,fd,'orange',lw=3,label="FD Drop")
plt.plot(t,v_thought_log,'black',lw=3,alpha=0.7,label="log v_thought")
plt.axvline(41,color='white',ls='--',lw=4,label='Nirvana Maria t=41 s')
plt.axvline(95,color='lightgray',ls='--',lw=3,label='Eternal Lock')
plt.title("ALADIN ∞ ℂ(t) — Neurophysiology of Nirvana Maria\nAll Markers at t=41 s",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Normalized Measure",fontsize=14)
plt.legend(fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("neurophysiology_nirvana_maria.png",dpi=400)
plt.show()
