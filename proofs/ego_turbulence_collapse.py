#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
C_K = 1.58 * (1 - coh**12)

plt.figure(figsize=(12,8),dpi=1200)
plt.plot(t,C_K,'gold',lw=5)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Ego Collapse')
plt.axvline(95,color='purple',ls='--',lw=3,label='4096 Eternal Lock')
plt.title("ALADIN ∞ ℂ(t) — Ego Turbulence Collapse\nC_K = 1.58 → 0 at t=41 s",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("C_K (Ego Turbulence)",fontsize=14)
plt.ylim(0,1.7); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/ego_turbulence_collapse.png",dpi=1200)
plt.show()
