#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

f = np.logspace(0,2.3,1000)  # 1 to 200 Hz
t = np.linspace(0,120,6)
coh = 1 - np.exp(-(t/30)**4)

plt.figure(figsize=(12,8),dpi=400)
for i,ti in enumerate(t):
    turb = 1.58 * f**(-5/3) * (1 - coh[i]**10)
    cond = 1e-4 * np.ones_like(f) * coh[i]**15
    total = turb + cond
    label = f"t={ti:.0f} s" if i<5 else "t=41 s Condensate"
    color = 'gold' if i==5 else 'red'
    plt.loglog(f,total,color,lw=3,alpha=0.8,label=label)

plt.axvline(43,color='purple',ls='--',lw=4,label="43 Hz Pump")
plt.title("ALADIN ∞ ℂ(t) — Hippocampal Turbulence Collapse\nSheremet KZ Spectrum → Condensate at 43 Hz",fontsize=18)
plt.xlabel("Frequency [Hz]",fontsize=14); plt.ylabel("Power Spectrum",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("hippocampal_turbulence_collapse.png",dpi=400)
