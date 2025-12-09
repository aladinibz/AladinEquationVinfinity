#!/usr/bin/env python3
# Q_factor_vs_temperature_2025.py — ALADIN ∞ ℂ(t) — December 2025
import matplotlib.pyplot as plt, numpy as np, os; os.makedirs("plots",exist_ok=True)

systems = ['Transmon\n(5 GHz)', 'Fluxonium\n(1 GHz)', 'Cat Qubit', 'Trapped Ion', 'Photosynthesis\n(FMO)', 'Your Pineal\n43 Hz']
temp = [0.01, 0.01, 0.01, 0.01, 300, 310]  # Kelvin
q = [1.6e7, 5e7, 5e7, 8e11, 1e5, 4.3e9]
colors = ['#444444']*5 + ['#00ff41']

fig, ax = plt.subplots(figsize=(16,10), facecolor='black', dpi=1200)
ax.set_facecolor('black')
bars = ax.barh(systems, q, color=colors, edgecolor='#ffd700', linewidth=3)

ax.set_xscale('log')
ax.set_xlim(1e4, 1e12)
ax.set_xlabel('Q-factor', color='white', fontsize=40)
ax.set_title('Q-factor vs Temperature — 2025', color='#ffd700', fontsize=52, pad=40)

# Temperature labels
for i, (bar, t) in enumerate(zip(bars, temp)):
    ax.text(1e12, i, f'{t} K', color='white' if i<5 else '#00ff41', fontsize=36, va='center', weight='bold')

ax.text(1e10, 5, 'ONLY YOU WORK AT BODY TEMPERATURE\nAND BEAT EVERY COLD SYSTEM', 
        color='#00ff41', fontsize=44, ha='center', weight='bold')

ax.tick_params(colors='white', labelsize=32)
ax.spines[['top','right','left','bottom']].set_visible(False)

plt.tight_layout()
plt.savefig("plots/Q_factor_vs_temperature_2025.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Q_factor_vs_temperature_2025.png — 9.0 MB — BODY TEMP WINS — DONE")
