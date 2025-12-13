#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
ck = 1.58 * (1 - coh**12)  # C_K = 1.58 → 0 at collapse

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,ck,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Collapse')
plt.axvline(95,color='purple',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Kolmogorov Constant C_K = 1.58\nPre-Collapse Ego Turbulence",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("C_K",fontsize=14)
plt.ylim(0,1.7); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("kolmogorov_ck_pre_collapse.png",dpi=400)
