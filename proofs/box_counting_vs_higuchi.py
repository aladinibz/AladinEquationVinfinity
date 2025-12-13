#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
higuchi = 1.8 - 0.6 * coh**10
box_count = 1.85 - 0.65 * coh**10  # slightly higher scale

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,higuchi,'gold',lw=4,label="Higuchi FD (robust)")
plt.plot(t,box_count,'darkorange',lw=3,alpha=0.8,label="Box-Counting FD (theoretical)")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Box-Counting vs Higuchi\nBoth Collapse in Ego Turbulence",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Fractal Dimension",fontsize=14)
plt.ylim(1.1,1.95); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("box_counting_vs_higuchi.png",dpi=400)
plt.show()
