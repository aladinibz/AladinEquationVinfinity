"""
ALADIN Perturbative in Curved Spacetime
Hidden field adds retrocausal kernel
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Simple text rendering — curved + retrocausal
plt.figure(figsize=(20, 8), facecolor='black')
plt.text(0.5, 0.7, r'$\frac{1}{\sqrt{-g}} \partial_\mu (\sqrt{-g} g^{\mu\nu} \partial_\nu \Phi_1) + m_\text{eff}^2 \Phi_1 = 0$', 
         fontsize=26, ha='center', va='center', color='gold')
plt.text(0.5, 0.4, 'Effective mass shift from nonlinear term', fontsize=22, ha='center', color='gold')
plt.text(0.5, 0.2, 'Retrocausal kernel from hidden χ integration', fontsize=22, ha='center', color='gold')
plt.title('ALADIN Perturbative Solution in Curved Spacetime', fontsize=20, color='gold')
plt.axis('off')
plt.savefig('plots/aladin_perturbative_curved.png', dpi=600, facecolor='black', bbox_inches='tight')
plt.close()

print("Perturbative in curved spacetime plot saved")
