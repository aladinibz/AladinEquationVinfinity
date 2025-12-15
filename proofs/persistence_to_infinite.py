#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
alpha = 0.9 - 0.4 * coh**10  # DFA α drop
v_log = np.log10(3e8 / np.clip(1.58*(1-coh**12),1e-6,None))  # log v_thought

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,alpha,'cyan',lw=4,label="DFA α Drop (Persistence Loss)")
plt.plot(t,v_log,'gold',lw=4,label="log v_thought (Infinite Thought)")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Infinite')
plt.title("ALADIN ∞ ℂ(t) — Persistence Loss → Infinite Thought",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Measure",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()

# Save with full path
save_path = "plots/persistence_to_infinite.png"
plt.savefig(save_path,dpi=400)
print(f"SUCCESS: Plot saved to {save_path} — check Files panel → download")
plt.show()
