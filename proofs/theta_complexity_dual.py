#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
theta = 0.2 + 1.2 * coh**12
fd = 1.8 - 0.6 * coh**10

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,theta,'purple',lw=4,label="Theta Dominance Rise")
plt.plot(t,fd,'gold',lw=4,label="Higuchi FD Drop")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.title("ALADIN ∞ ℂ(t) — Theta + Complexity Dual\nTimeless Flow + Chaos to Order at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Normalized Measure",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()

# Save to plots folder
plt.savefig("plots/theta_complexity_dual.png",dpi=400)
print("Saved: plots/theta_complexity_dual.png — check Files panel → download")
