#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
flux_forward = coh**8  # direct flux normal
flux_inverse = -coh**12 * np.sin(2*np.pi*9*43*t/43)  # negative retro flux (9th sound)

total_flux = flux_forward + flux_inverse * (t>41)

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,flux_forward,'red',lw=3,label="Forward Flux (Normal Ego)")
plt.plot(t,total_flux,'gold',lw=4,label="With Retrocausal Inverse Flux")
plt.axvline(41,color='purple',ls='--',lw=4,label="t=41 s Retro Onset")
plt.axvline(95,color='black',ls='--',lw=4,label="4096 Lock")
plt.title("ALADIN ∞ ℂ(t) — Retrocausality as Inverse Energy Flux\n9th Sound Past Reset",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Energy Flux P",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("retrocausality_inverse_flux.png",dpi=400)
plt.show()
