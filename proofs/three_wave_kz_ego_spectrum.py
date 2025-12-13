#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

f = np.logspace(0,2.3,1000)  # 1-200 Hz
t = np.linspace(0,120,6)
coh = 1 - np.exp(-(t/30)**4)

plt.figure(figsize=(12,8),dpi=400)
for i,ti in enumerate(t):
    alpha = 2.5 + coh[i]*1.0  # steeper with coherence (KZ limit α=3)
    spec = f**(-alpha) * (1 - coh[i]**8)
    cond = 1e-5 * np.ones_like(f) * coh[i]**15
    total = spec + cond
    label = f"t={ti:.0f} s (α={alpha:.2f})" if i<5 else "t=41 s KZ Collapse"
    color = 'gold' if i==5 else 'red'
    plt.loglog(f,total,color,lw=3,alpha=0.8,label=label)

plt.axvline(43,color='purple',ls='--',lw=4,label="43 Hz Pump")
plt.title("ALADIN ∞ ℂ(t) — Three-Wave KZ Spectrum in Ego Turbulence\nTheta→Gamma Cascade Collapse",fontsize=18)
plt.xlabel("Frequency [Hz]",fontsize=14); plt.ylabel("Power Spectrum",fontsize=14)
plt.legend(fontsize=11); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("three_wave_kz_ego_spectrum.png",dpi=400)
plt.show()
