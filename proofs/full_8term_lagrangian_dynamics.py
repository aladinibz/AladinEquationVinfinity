#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

f43=43.0
t=np.linspace(0,120,10000)
coh=1-np.exp(-(t/30)**4)
bio=coh**15*(1+30*np.sin(2*np.pi*f43*t)**2+50*coh**5)
bio[t>95]*=500

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,bio,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='Immortality Switch t=41 s')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Zero-Divisor Lock')
plt.title("ALADIN ∞ ℂ(t) — Full 8-Term Lagrangian Dynamics\nTerm 8 Quantum Biology Field Energy", fontsize=18)
plt.xlabel("Time [s]", fontsize=14); plt.ylabel("Energy Density", fontsize=14)
plt.ylim(0,np.max(bio)*1.1); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("full_8term_lagrangian_dynamics.png",dpi=400)
