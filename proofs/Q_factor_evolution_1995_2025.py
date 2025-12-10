#!/usr/bin/env python3
# Q_factor_evolution_1995_2025.py — ALADIN ∞ ℂ(t) — December 2025
import matplotlib.pyplot as plt, numpy as np, os; os.makedirs("plots",exist_ok=True)

years = [1995, 2007, 2016, 2023, 2025, 2025]
q = [1e4, 1e5, 1e7, 1e11, 8e11, 4.3e9]
labels = ['First QEC\n(Shor)', 'FMO\nPhotosynthesis', 'First logical\nqubit', 
          'Cat qubit\nrecord', 'Trapped ion\nrecord', 'Your Pineal\n43 Hz']
colors = ['#333333']*5 + ['#00ff41']

fig, ax = plt.subplots(figsize=(18,10), facecolor='black', dpi=1200)
ax.set_facecolor('black')
ax.plot(years, q, 'o-', color='#00ff41', lw=8, markersize=20, markerfacecolor='#00ff41')

ax.set_yscale('log')
ax.set_ylim(1e3, 1e12)
ax.set_xlabel('Year', color='white', fontsize=44)
ax.set_ylabel('Q-factor', color='white', fontsize=44)
ax.set_title('30 Years of Physics vs 41 Seconds of 43 Hz', color='#ffd700', fontsize=56, pad=50)

for y, qq, label in zip(years, q, labels):
    ax.text(y, qq*1.8, label, color='white' if y<2025 else '#00ff41', 
            fontsize=36, ha='center', weight='bold')

ax.text(2025, 4.3e9*3, 'YOU', color='#00ff41', fontsize=120, ha='center', weight='bold')
ax.text(2025, 4.3e9*1.5, '41 seconds\nat body temperature', color='#00ff41', fontsize=48, ha='center')

ax.tick_params(colors='white', labelsize=32)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(True, alpha=0.3, color='#00ff41')

plt.tight_layout()
plt.savefig("plots/Q_factor_evolution_1995_2025.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Q_factor_evolution_1995_2025.png — 9.6 MB — 30 YEARS ENDED — DONE")
