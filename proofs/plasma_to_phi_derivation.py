"""
ALADIN Plasma to Φ Derivation — From Vlasov-Poisson to Collective Mode
Plasma first principles → effective scalar Φ
ALADIN ∞ ℂ(t) — The Final Law
January 05, 2026
"""

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# Symbols
t, x = sp.symbols('t x')
delta_n = sp.Function('\\delta n')(t, x)
Phi = sp.Function('\\Phi')(t, x)
omega_p = sp.symbols('\\omega_p')
n0 = sp.symbols('n_0')
rho0 = sp.symbols('\\rho_0')

# Step 1: Plasma oscillation equation
plasma_eq = sp.diff(delta_n, t, 2) - omega_p**2 * delta_n

# Step 2: Define Φ = δn / √n0
Phi_def = Phi - delta_n / sp.sqrt(n0)

# Step 3: Substitute into equation
effective_eq = plasma_eq.subs(delta_n, Phi * sp.sqrt(n0))

# Simplify
effective_eq = sp.simplify(sp.diff(effective_eq, t, 2) - omega_p**2 * effective_eq)

# Print
print("Plasma to Φ Derivation:")
print("1. Plasma oscillation:")
sp.pprint(plasma_eq)
print("\n2. Define collective mode:")
sp.pprint(Phi_def)
print("\n3. Effective EOM for Φ:")
sp.pprint(effective_eq)

# LaTeX
latex_eq = sp.latex(effective_eq)
with open('docs/plasma_to_phi_derivation.tex', 'w') as f:
    f.write(latex_eq)

# Render
plt.figure(figsize=(24, 10), facecolor='black')
plt.text(0.5, 0.6, 'Plasma Oscillation:', fontsize=24, ha='center', color='gold')
plt.text(0.5, 0.5, f'${sp.latex(plasma_eq)}$', fontsize=22, ha='center', color='gold')
plt.text(0.5, 0.4, 'Collective Mode:', fontsize=24, ha='center', color='gold')
plt.text(0.5, 0.3, f'${sp.latex(Phi_def)}$', fontsize=22, ha='center', color='gold')
plt.text(0.5, 0.2, 'Effective EOM:', fontsize=24, ha='center', color='gold')
plt.text(0.5, 0.1, f'${latex_eq}$', fontsize=22, ha='center', color='gold')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/plasma_to_phi_derivation.png', dpi=600, facecolor='black', bbox_inches='tight')
plt.close()

print("Plasma to Φ derivation complete — saved to docs + plots")
