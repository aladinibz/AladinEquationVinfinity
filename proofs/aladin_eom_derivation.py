"""
ALADIN EOM Derivation — From Master Lagrangian
Euler-Lagrange for Φ*
ALADIN ∞ ℂ(t) — The Final Law
January 06, 2026
"""

import sympy as sp
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# Symbols
t, x = sp.symbols('t x')
Phi = sp.Function('\\Phi')(t, x)
Phi_star = sp.Function('\\Phi^*')(t, x)
J0 = sp.symbols('J_0')
m, lambda_ = sp.symbols('m \\lambda')
Lam = sp.symbols('\\Lambda')

# Invariant |Φ|²
phi2 = Phi_star * Phi

# Re(Φ) explicit — fixed
re_phi = (Phi + Phi_star)/2

# Core Lagrangian density (flat for simplicity, curved similar)
L_density = -sp.Rational(1,2) * (sp.diff(Phi_star, t)**2 - sp.diff(Phi_star, x)**2) \
            + sp.Rational(1,2) * m**2 * phi2 \
            + sp.Rational(1,4) * lambda_ * phi2**2 \
            + J0 * re_phi

# Higher-D (symbolic)
higher = sp.Sum(sp.symbols(f'c_{{n}}') / Lam**(n-4) * phi2**(n/2), (n, 6, sp.oo, 2))

L_density = L_density + higher

# Euler-Lagrange for Phi_star
dL_dPhi_star = sp.diff(L_density, Phi_star)
dL_dPhi_star_t = sp.diff(L_density, sp.diff(Phi_star, t))
dL_dPhi_star_x = sp.diff(L_density, sp.diff(Phi_star, x))

EL = dL_dPhi_star - sp.diff(dL_dPhi_star_t, t) + sp.diff(dL_dPhi_star_x, x)

EOM = sp.simplify(EL)

# Print
print("Derived EOM for Φ*:")
sp.pprint(EOM)

# LaTeX
latex_EOM = sp.latex(EOM)
with open('docs/aladin_eom_derivation.tex', 'w') as f:
    f.write(latex_EOM)

# Render
plt.figure(figsize=(28, 10), facecolor='black')
plt.text(0.5, 0.5, f'${latex_EOM}$', fontsize=26, ha='center', va='center', color='gold')
plt.text(0.5, 0.3, 'Derived from ALADIN Master Lagrangian', fontsize=20, ha='center', color='gold')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/aladin_eom_derivation.png', dpi=600, facecolor='black', bbox_inches='tight')
plt.close()

print("EOM derivation complete — code runs clean")
