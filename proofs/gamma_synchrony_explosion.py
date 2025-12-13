#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
gamma = 0.2 + 1.8 * coh**8 * (1 + 0.6*np.sin(2*np.pi*43*t))

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,gamma,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Gamma Synchrony Explosion\nGlobal Coherence at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Gamma Power",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("gamma_synchrony_explosion.png",dpi=400)
