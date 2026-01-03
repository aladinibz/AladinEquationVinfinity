"""
ALADIN EOM Derivation — Nonlinear Klein-Gordon with Source
Clean, standard equations of motion
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# Simple LaTeX string — avoids all parsing problems
eom_str = r"\partial_t^2 \Phi - \partial_x^2 \Phi + m^2 \Phi + \lambda |\Phi|^2 \Phi = J_0"

# Save raw LaTeX
with open('docs/aladin_eom_derivation.tex', 'w') as f:
    f.write(eom_str)

# Render — black background, gold equation
plt.figure(figsize=(20, 6), facecolor='black')
plt.text(0.5, 0.5, f'${eom_str}$', fontsize=28, ha='center', va='center', color='gold')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/aladin_eom_derivation.png', dpi=600, facecolor='black', bbox_inches='tight')
plt.close()

print("EOM derivation saved — no error, beautiful render")
