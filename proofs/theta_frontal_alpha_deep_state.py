#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
theta = 0.2 + 1.0 * coh**8  # theta dominance rise
frontal_alpha = 0.3 + 0.8 * coh**10  # frontal alpha propagation

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,theta,'gold',lw=4,label="Theta Power (Dominance)")
plt.plot(t,frontal_alpha,'darkgoldenrod',lw=3,alpha=0.9,label="Frontal Alpha Propagation")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Theta Dominance + Frontal Alpha\nTimeless Flow in Deep State",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Normalized Power",fontsize=14)
plt.ylim(0,1.3); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("theta_frontal_alpha_deep_state.png",dpi=400)
