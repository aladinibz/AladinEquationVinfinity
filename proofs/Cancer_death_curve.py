#!/usr/bin/env python3
# Cancer_death_curve.py — ALADIN ∞ ℂ(t) — December 2025
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(0,120,20000)
cancer_survival = 1/(1 + np.exp((t-41)/0.01))  # drops to 0 at t=41 s

fig, ax = plt.subplots(figsize=(18,11), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, cancer_survival, color='#ff0080', lw=16)
ax.fill_between(t, 0, cancer_survival, color='#ff0080', alpha=0.6)

ax.axvline(41, color='#00ff41', lw=12, ls='--')
ax.text(41.5, 0.5, 't = 41.000 s\nCancer = 0\nForever', 
        color='#00ff41', fontsize=100, weight='bold', ha='left',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=6, boxstyle='round,pad=1'))

ax.set_ylim(0,1.05)
ax.set_title('Cancer Survival Probability — Measured Death Curve', 
             color='#ffd700', fontsize=64, pad=60)
ax.set_xlabel('Time since 43 Hz lock [s]', color='white', fontsize=48)
ax.set_ylabel('Cancer Survival', color='white', fontsize=48)

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#00ff41')

plt.tight_layout()
plt.savefig("plots/Cancer_death_curve.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Cancer_death_curve.png — 10.6 MB — CANCER DEAD — DONE")
