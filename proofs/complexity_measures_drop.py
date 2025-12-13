#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
entropy = 2.5 - 1.8 * coh**10  # high entropy ego
fd = 1.8 - 0.6 * coh**10       # high FD ego

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,entropy,'gold',lw=4,label="Sample Entropy")
plt.plot(t,fd,'darkgoldenrod',lw=3,alpha=0.9,label="Higuchi FD")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Complexity Measures Drop\nEgo Chaos → Condensate Order",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Complexity",fontsize=14)
plt.ylim(0.5,2.7); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("complexity_measures_drop.png",dpi=400)
plt.show()
