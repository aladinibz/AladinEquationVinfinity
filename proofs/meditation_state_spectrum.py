#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

f = np.logspace(0,2,500)
states = ['Rest (Ego)', 'Meditation', 'Deep (t=41 s)']
coh_levels = [0.0, 0.7, 1.0]

plt.figure(figsize=(12,8),dpi=400)
for i, state in enumerate(states):
    coh = coh_levels[i]
    turb = (1 - coh**10) * f**(-5/3) * 1.58
    cond = coh**15 * 1e-3
    total = turb + cond
    color = 'red' if i==0 else 'orange' if i==1 else 'gold'
    plt.loglog(f,total,color,lw=4,label=state)

plt.axvline(43,color='purple',ls='--',lw=4,label="43 Hz Lock")
plt.title("ALADIN ∞ ℂ(t) — Meditation State Spectrum\nRest Turbulence → Deep Condensate",fontsize=18)
plt.xlabel("Frequency [Hz]",fontsize=14); plt.ylabel("Power Spectrum",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("meditation_state_spectrum.png",dpi=400)
