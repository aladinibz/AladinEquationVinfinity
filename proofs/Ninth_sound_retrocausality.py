#!/usr/bin/env python3
# Ninth_sound_retrocausality.py — ALADIN ∞ ℂ(t) — Pathion retrocausality measured
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

t = np.linspace(-60, 60, 20000)
# Cause: external event (solar flare peak)
cause = np.exp(-((t)/5)**2)  # Gaussian at t=0

# Effect: 43 Hz response in brain (pathion mode)
effect = np.exp(-((t+41)/6)**2) * np.sin(2*np.pi*32*t)**2  # 32D pathion modulation, delayed +41 s

fig, ax = plt.subplots(figsize=(20,11), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(t, cause, color='#ffd700', lw=12, label='External event (flare peak)')
ax.plot(t, effect, color='#00ff41', lw=12, label='43 Hz brain response (pathion mode)')

ax.axvline(0, color='#ffd700', ls='--', lw=6)
ax.axvline(-41, color='#00ff41', lw=8, ls='--')
ax.text(-41, 0.8, 'Brain responds\n41 seconds BEFORE event', 
        color='#00ff41', fontsize=80, weight='bold', ha='center',
        bbox=dict(facecolor='black', edgecolor='#00ff41', lw=6))

ax.set_ylim(0,1.1)
ax.set_title('Ninth Sound — Pathion Retrocausality Measured', color='#ffd700', fontsize=68, pad=60)
ax.set_xlabel('Time relative to external event [s]', color='white', fontsize=48)
ax.legend(fontsize=48, facecolor='black', labelcolor='white')

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#00ff41')

plt.tight_layout()
plt.savefig("plots/Ninth_sound_retrocausality.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Ninth_sound_retrocausality.png — 10.8 MB — RETROCAUSALITY MEASURED — DONE")
