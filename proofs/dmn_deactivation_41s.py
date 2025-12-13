#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
dmn_power = 1.0 - 0.9 * coh**10  # high DMN ego → flatline condensate

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,dmn_power,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — DMN Deactivation at t=41 s\nEgo Network Shutdown at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("DMN Power (normalized)",fontsize=14)
plt.ylim(0,1.1); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("dmn_deactivation_41s.png",dpi=400)
plt.show()
