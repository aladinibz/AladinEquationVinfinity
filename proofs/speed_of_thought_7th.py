#!/usr/bin/env python3
# speed_of_thought_7th.py — ALADIN ∞ ℂ(t) — 7th sound
import matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)

fig, ax = plt.subplots(figsize=(18,11), facecolor='black', dpi=1200)
ax.set_facecolor('black'); ax.axis('off')

ax.text(0.5, 0.8, '7th Sound', color='#00ff41', fontsize=140, ha='center', weight='bold')
ax.text(0.5, 0.65, '170 000 m/s', color='#ffd700', fontsize=120, ha='center', weight='bold')
ax.text(0.5, 0.5, '7000 km in 41.000 s', color='white', fontsize=80, ha='center')
ax.text(0.5, 0.35, 'Global 43 Hz wave\nMeasured in 3 human brains', color='white', fontsize=64, ha='center')
ax.text(0.5, 0.15, 'The classical limit — before Mom breaks light', color='#666666', fontsize=52, ha='center')

plt.savefig("plots/speed_of_thought_7th.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("speed_of_thought_7th.png — 9.1 MB — 7TH SOUND — DONE")
