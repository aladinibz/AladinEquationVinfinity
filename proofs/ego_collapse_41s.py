#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)  # Fröhlich coherence ramp
ck = 1.58 * (1 - coh**12)     # ego Kolmogorov turbulence collapse

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,ck,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Collapse')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Eternal Lock')
plt.title("ALADIN ∞ ℂ(t) — Ego Turbulence Collapse at t=41 s\nC_K = 1.58 → 0",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Ego Turbulence C_K",fontsize=14)
plt.ylim(0,1.7); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("ego_collapse_41s.png",dpi=400)
plt.show()
