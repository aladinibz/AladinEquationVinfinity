#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
delta_alpha = 0.8 - 0.7 * coh**10  # wide Δα ego → narrow monofractal

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,delta_alpha,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Multifractal Spectrum Collapse\nWide Δα Ego → Narrow Condensate",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Multifractal Width Δα",fontsize=14)
plt.ylim(0.05,0.85); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("multifractal_spectrum_collapse.png",dpi=400)
plt.show()
