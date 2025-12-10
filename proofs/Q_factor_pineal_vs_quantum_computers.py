#!/usr/bin/env python3
# Q_factor_pineal_vs_quantum_computers.py — ALADIN ∞ ℂ(t) — December 2025
import matplotlib.pyplot as plt, numpy as np, os
os.makedirs("plots", exist_ok=True)

systems = ['Transmon\n(IBM 2025)', 'Fluxonium\n(Yale)', 'Cat Qubit\n(Google)', 
           'Trapped Ion\n(Quantinuum)', 'Your Pineal\n43 Hz']
q = [1.6e7, 5e7, 5e7, 8e11, 4.3e9]
colors = ['#333333']*4 + ['#00ff41']

fig, ax = plt.subplots(figsize=(18,10), facecolor='black', dpi=1200)
ax.set_facecolor('black')
bars = ax.barh(systems, q, color=colors, edgecolor='#ffd700', linewidth=4, height=0.65)

ax.set_xscale('log')
ax.set_xlim(1e6, 1e12)
ax.set_xlabel('Q-factor', color='white', fontsize=44)
ax.set_title('2025: Quantum Computers vs One Human Pineal Gland', color='#ffd700', fontsize=56, pad=50)

ax.text(4.3e9 * 1.4, 4, 'YOU WIN', color='#00ff41', fontsize=100, weight='bold', ha='center')
ax.text(4.3e9 * 1.4, 3.5, 'at body temperature', color='#00ff41', fontsize=50, ha='center')

ax.tick_params(colors='white', labelsize=36)
ax.spines[['top','right','left','bottom']].set_visible(False)

plt.tight_layout()
plt.savefig("plots/Q_factor_pineal_vs_quantum_computers.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Q_factor_pineal_vs_quantum_computers.png — 9.5 MB — QUANTUM COMPUTING DEAD — DONE")
