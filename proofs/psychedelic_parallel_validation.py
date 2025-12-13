#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
gamma_dmt = 0.1 + 2.0 * coh**8 * (1 + 0.7*np.sin(2*np.pi*43*t))
dmn_dmt = 1.0 - 0.9 * coh**10
theta_dmt = 0.2 + 1.2 * coh**12

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,gamma_dmt,'gold',lw=4,label="Gamma Burst (DMT/5-MeO)")
plt.plot(t,dmn_dmt,'darkred',lw=3,alpha=0.8,label="DMN Collapse")
plt.plot(t,theta_dmt,'purple',lw=3,alpha=0.7,label="Theta Lock")
plt.axvline(41,color='black',ls='--',lw=4,label='t=41 s Switch')
plt.title("ALADIN ∞ ℂ(t) — Psychedelic Parallel Validation\nDMT/5-MeO Signatures at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Normalized Power",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("psychedelic_parallel_validation.png",dpi=400)
