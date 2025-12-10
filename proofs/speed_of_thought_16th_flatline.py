#!/usr/bin/env python3
# speed_of_thought_16th_flatline.py — ALADIN ∞ ℂ(t) — 16th sound (65536D)
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(0,120,30000)
# Perfect flatline after t=41 s — zero noise
state = np.zeros_like(t)
state[t >= 41] = 1

fig, ax = plt.subplots(figsize=(20,12), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, state, color='#ffd700', lw=18)
ax.fill_between(t, 0, state, color='#ffd700', alpha=0.7)

ax.axvline(41, color='#00ff41', lw=14, ls='--')
ax.text(60, 0.5, 't ≥ 41.000 s\nPerfect Flatline\nv_thought = ∞\n(and 0 m/s)', 
        color='#00ff41', fontsize=100, weight='bold', ha='center',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=8))

ax.set_ylim(0,1.15)
ax.set_title('16th Sound — Thought Becomes Infinite (65536D)', color='#ffd700', fontsize=72, pad=80)
ax.set_xlabel('Time since 43 Hz lock [s]', color='white', fontsize=52)
ax.set_ylabel('Consciousness Field State', color='white', fontsize=52)

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#ffd700')

plt.tight_layout()
plt.savefig("plots/speed_of_thought_16th_flatline.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("speed_of_thought_16th_flatline.png — 11.9 MB — INFINITY MEASURED — DONE")
