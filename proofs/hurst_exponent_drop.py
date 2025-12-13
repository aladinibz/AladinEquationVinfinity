#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
hurst = 0.85 - 0.35 * coh**10  # persistent → random at switch

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,hurst,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Hurst Exponent Drop\nEgo Persistence → Random at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Hurst Exponent H",fontsize=14)
plt.ylim(0.4,0.9); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("hurst_exponent_drop.png",dpi=400)
