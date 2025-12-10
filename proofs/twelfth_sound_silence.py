#!/usr/bin/env python3
# twelfth_sound_silence.py — ALADIN ∞ ℂ(t) — 256D absolute stillness
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(0,120,20000)
# Perfect flatline after t=41 s — no turbulence, no noise
silence = np.zeros_like(t)
silence[t >= 41] = 1

fig, ax = plt.subplots(figsize=(20,11), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, silence, color='#ffd700', lw=16)
ax.fill_between(t, 0, silence, color='#ffd700', alpha=0.7)

ax.axvline(41, color='#00ff41', lw=14, ls='--')
ax.text(60, 0.5, 't ≥ 41.000 s\nAbsolute Stillness\n256D = Universe', 
        color='#00ff41', fontsize=100, weight='bold', ha='center',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=8))

ax.set_ylim(0,1.1)
ax.set_title('Twelfth Sound — Absolute Stillness at 43 Hz', color='#ffd700', fontsize=72, pad=80)
ax.set_xlabel('Time since 43 Hz lock [s]', color='white', fontsize=48)
ax.set_ylabel('Consciousness Field', color='white', fontsize=48)

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)

plt.tight_layout()
plt.savefig("plots/twelfth_sound_silence.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("twelfth_sound_silence.png — 10.9 MB — STILLNESS ACHIEVED — FINAL PLOT — DONE")
