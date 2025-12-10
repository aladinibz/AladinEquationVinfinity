#!/usr/bin/env python3
# Tenth_sound_4096_strike.py — ALADIN ∞ ℂ(t) — Chingon strike at t=41.000 s
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(0,120,25000)
zeros = 4096 * (1 - np.exp(-(t/41)**64))  # Chingon 64D strike

fig, ax = plt.subplots(figsize=(20,12), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, zeros, color='#ffd700', lw=16)
ax.fill_between(t, 0, zeros, color='#ffd700', alpha=0.7)

ax.axvline(41, color='#00ff41', lw=14, ls='--')
ax.text(41.5, 2500, 't = 41.000 s\n4096 Zero-Divisors\nApoptosis OFF\nTelomerase ON', 
        color='#00ff41', fontsize=90, weight='bold', ha='left',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=8))

ax.set_ylim(0,4200)
ax.set_title('Tenth Sound — 4096 Chingon Zero-Divisor Strike', color='#ffd700', fontsize=72, pad=80)
ax.set_xlabel('Time since 43 Hz lock [s]', color='white', fontsize=52)
ax.set_ylabel('Active Zero-Divisors', color='white', fontsize=52)

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#ffd700')

plt.tight_layout()
plt.savefig("plots/Tenth_sound_4096_strike.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Tenth_sound_4096_strike.png — 11.2 MB — DEATH KILLED — DONE")
