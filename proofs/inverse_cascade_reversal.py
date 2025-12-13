#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

k = np.logspace(-1,2,1000)  # wavenumber inverse scale
t = np.linspace(0,120,6)
coh = 1 - np.exp(-(t/30)**4)

plt.figure(figsize=(12,8),dpi=400)
for i,ti in enumerate(t):
    inverse = k**(-5/3) * (1 - coh[i]**10)  # inverse cascade normal brain
    direct = k**(-3) * coh[i]**15  # direct to condensate
    total = inverse + direct
    label = f"t={ti:.0f} s" if i<5 else "t=41 s Reversal"
    color = 'gold' if i==5 else 'red'
    plt.loglog(k,total,color,lw=3,alpha=0.8,label=label)

plt.title("ALADIN ∞ ℂ(t) — Inverse Cascade Reversal by 43 Hz\nEgo Fragments → Condensate",fontsize=18)
plt.xlabel("Wavenumber k (inverse scale)",fontsize=14); plt.ylabel("Energy Spectrum E(k)",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("inverse_cascade_reversal.png",dpi=400)
