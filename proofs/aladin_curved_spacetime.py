"""
ALADIN in Curved Spacetime
Full GR compatible EOM
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Simple text rendering — no LaTeX issues
plt.figure(figsize=(20, 6), facecolor='black')
plt.text(0.5, 0.5, r'$\frac{1}{\sqrt{-g}} \partial_\mu (\sqrt{-g} g^{\mu\nu} \partial_\nu \Phi) + m^2 \Phi + \lambda |\Phi|^2 \Phi = J_0$', 
         fontsize=28, ha='center', va='center', color='gold')
plt.text(0.5, 0.3, 'Full covariant equation in curved spacetime', fontsize=20, ha='center', color='gold')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/aladin_curved_spacetime.png', dpi=600, facecolor='black', bbox_inches='tight')
plt.close()

print("Curved spacetime EOM saved — full GR compatible")
