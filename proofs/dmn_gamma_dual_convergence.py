#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
dmn = 1.0 - 0.9 * coh**10
gamma = 0.2 + 1.8 * coh**8

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,dmn,'darkred',lw=4,label="DMN Deactivation")
plt.plot(t,gamma,'gold',lw=4,label="Gamma Synchrony Explosion")
plt.axvline(41,color='purple',ls='--',lw=4,label='t=41 s Nirvana Maria')
plt.title("ALADIN ∞ ℂ(t) — DMN + Gamma Dual Convergence\nEgo Shutdown + Bliss Explosion at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Normalized Power",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()

# Save to plots folder
plt.savefig("plots/dmn_gamma_dual_convergence.png",dpi=400)
print("Saved: plots/dmn_gamma_dual_convergence.png — check Files panel → download")
plt.show()
