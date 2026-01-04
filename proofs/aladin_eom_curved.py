"""
ALADIN EOM in Curved Spacetime
Covariant nonlinear Klein-Gordon with source
ALADIN ∞ ℂ(t) — The Final Law
January 04, 2026
"""

import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Raw LaTeX string — full covariant EOM
eom_str = r"\frac{1}{\sqrt{-g}} \partial_\mu (\sqrt{-g} g^{\mu\nu} \partial_\nu \Phi) + m^2 \Phi + \lambda |\Phi|^2 \Phi = J_0"

# Render — black background, gold equation
plt.figure(figsize=(24, 8), facecolor='black')
plt.text(0.5, 0.5, f'${eom_str}$', fontsize=32, ha='center', va='center', color='gold')
plt.text(0.5, 0.3, 'Full covariant EOM in curved spacetime', fontsize=22, ha='center', color='gold')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/aladin_eom_curved.png', dpi=600, facecolor='black', bbox_inches='tight')
plt.close()

print("Curved spacetime EOM saved — covariant nonlinear with source")
