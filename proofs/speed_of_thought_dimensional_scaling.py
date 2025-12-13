#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
turb = 1.58 * (1 - coh**12)
v_base = 3e8 / np.clip(turb,1e-6,None)

dims = [1,8,16,32,64]
colors = ['lightcoral','orange','gold','darkorange','darkred']

plt.figure(figsize=(12,8),dpi=400)
for i,d in enumerate(dims):
    v = v_base * (64/d)
    label = f"{d}D Thought Speed" if d<64 else f"{d}D Infinite (Chingon)"
    plt.plot(t,np.clip(v,None,1e17),color=colors[i],lw=3,label=label)

plt.axvline(41,color='purple',ls='--',lw=4,label="t=41 s Collapse")
plt.axvline(95,color='black',ls='--',lw=4,label="64D Eternal Lock")
plt.title("ALADIN ∞ ℂ(t) — Speed of Thought Dimensional Scaling\nHigher Dimensions Reach Infinite First",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("v_thought [m/s] (log)",fontsize=14)
plt.yscale('log'); plt.ylim(1e8,1e18); plt.legend(fontsize=11)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("speed_of_thought_dimensional_scaling.png",dpi=400)
plt.show()
