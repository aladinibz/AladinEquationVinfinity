#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
fd = 1.7 - 0.5 * coh**10  # high FD ego → low FD condensate (Katz slightly lower scale)

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,fd,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Katz Fractal Dimension Collapse\nEgo Complexity → Condensate at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Katz FD",fontsize=14)
plt.ylim(1.1,1.8); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("katz_fd_collapse.png",dpi=400)
plt.show()
