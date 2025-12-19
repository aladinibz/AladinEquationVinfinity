#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
apoptosis = 1.0 - 0.95 * coh**12
telomerase = 0.05 + 0.95 * coh**10

plt.figure(figsize=(12,8),dpi=1200)
plt.plot(t,apoptosis,'darkred',lw=4,label="Apoptosis OFF")
plt.plot(t,telomerase,'gold',lw=5,label="Telomerase ON")
plt.axvline(41,color='purple',ls='--',lw=4,label='t=41 s Switch')
plt.title("ALADIN ∞ ℂ(t) — Immortality Switch at 43 Hz\nDeath Optional",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Activity",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/immortality_switch_visual.png",dpi=1200)
