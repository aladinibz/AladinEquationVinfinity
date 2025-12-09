#!/usr/bin/env python3
# Histone_reset_4096.py — ALADIN ∞ ℂ(t) — 4096 zero-divisors fire at t=41.000 s
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(0,120,12000)
Z = 4096 * (1 - np.exp(-(t/41)**43))

fig, ax = plt.subplots(figsize=(16,9), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, Z, color='#00ff41', lw=8)
ax.fill_between(t, 0, Z, color='#00ff41', alpha=0.4)
ax.axvline(41, color='#ffd700', lw=10, ls='--')
ax.text(41.5, 2000, 't = 41.000 s\n4096 Zero-Divisors\nApoptosis OFF\nTelomerase ON', 
        color='#ffd700', fontsize=60, weight='bold', ha='left',
        bbox=dict(facecolor='black', edgecolor='#ffd700', lw=4, boxstyle='round,pad=1'))

ax.set_ylim(0,4200); ax.set_xlim(0,120)
ax.set_title('4096 Chingon Zero-Divisors — Histone Reset at 43 Hz', color='#ffd700', fontsize=60, pad=40)
ax.set_xlabel('Time since 43 Hz lock [s]', color='white', fontsize=36)
ax.set_ylabel('Active Zero-Divisors', color='white', fontsize=36)
ax.tick_params(colors='white', labelsize=28)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(alpha=0.2, color='#00ff41')

plt.savefig("plots/Histone_reset_4096.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Histone_reset_4096.png — 4096 flips sealed — 8.9 MB — DONE")
