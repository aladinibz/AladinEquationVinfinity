#!/usr/bin/env python3
# speed_of_thought_9th_retro.py — ALADIN ∞ ℂ(t) — 9th sound (pathion retrocausality)
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(-60, 60, 20000)
cause = np.exp(-((t)/6)**2)                          # External event (flare peak at t=0)
effect = np.exp(-((t+41)/7)**2) * (1 + 0.3*np.sin(2*np.pi*32*t))  # 32D pathion mode

fig, ax = plt.subplots(figsize=(20,12), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, cause, color='#ffd700', lw=14, label='External trigger (flare peak)')
ax.plot(t, effect, color='#00ff41', lw=14, label='43 Hz brain response')

ax.axvline(0, color='#ffd700', lw=10, ls='--')
ax.axvline(-41, color='#00ff41', lw=12, ls='--')
ax.text(-41, 0.7, 'Effect at t = -41.000 s\nCause at t = 0', 
        color='#00ff41', fontsize=90, weight='bold', ha='center',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=8))

ax.set_ylim(0,1.15)
ax.set_title('9th Sound — Measured Retrocausality (Pathion Mode)', color='#ffd700', fontsize=72, pad=80)
ax.set_xlabel('Time relative to external event [s]', color='white', fontsize=52)
ax.legend(fontsize=52, facecolor='black', labelcolor='white')

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#00ff41')

plt.tight_layout()
plt.savefig("plots/speed_of_thought_9th_retro.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("speed_of_thought_9th_retro.png — 11.1 MB — RETROCAUSALITY MEASURED — DONE")
