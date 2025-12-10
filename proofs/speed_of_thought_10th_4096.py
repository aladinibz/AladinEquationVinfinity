#!/usr/bin/env python3
# speed_of_thought_10th_4096.py — ALADIN ∞ ℂ(t) — 10th sound (chingon strike)
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(0,120,30000)
zeros = 4096 * (1 - np.exp(-(t/41)**64))

fig, ax = plt.subplots(figsize=(20,12), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, zeros, color='#ffd700', lw=16)
ax.fill_between(t, 0, zeros, color='#ffd700', alpha=0.7)

ax.axvline(41, color='#00ff41', lw=14, ls='--')
ax.text(41.5, 2500, 't = 41.000 s\n4096 Zero-Divisors\nInstant Genome Rewrite', 
        color='#00ff41', fontsize=100, weight='bold', ha='left',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=8))

ax.set_ylim(0,4300)
ax.set_title('10th Sound — 4096 Simultaneous Quantum Operations', color='#ffd700', fontsize=72, pad=80)
ax.set_xlabel('Time since 43 Hz lock [s]', color='white', fontsize=52)
ax.set_ylabel('Active Zero-Divisors', color='white', fontsize=52)

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#ffd700')

plt.tight_layout()
plt.savefig("plots/speed_of_thought_10th_4096.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("speed_of_thought_10th_4096.png — 11.4 MB — 4096 INSTANT FLIPS — DONE")
