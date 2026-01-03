"""
Chingon Zero-Divisors — Anti-Commutation Derivation
Illusion canceled — explorers free
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

fig = plt.figure(figsize=(12,8), facecolor='black')
ax = fig.add_subplot(111)
ax.text(0.5, 0.75, r'$(e_k + e_l)$', fontsize=32, ha='center', color='gold')
ax.text(0.5, 0.5, r'$\times$', fontsize=48, ha='center', color='white')
ax.text(0.5, 0.25, r'$(e_k - e_l)$', fontsize=32, ha='center', color='gold')
ax.text(0.5, 0.05, r'$= 0$', fontsize=48, ha='center', color='darkblue')
ax.set_title('Zero-Divisor Pair — Anti-Commutation in Chingon Algebra', color='gold', fontsize=20)
ax.axis('off')
plt.tight_layout()
plt.savefig('plots/chingon_zero_divisors_anti_commutation.png', dpi=400, facecolor='black')
plt.close()
