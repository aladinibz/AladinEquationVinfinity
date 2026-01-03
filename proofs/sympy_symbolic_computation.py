"""
SymPy Symbolic Computation for ALADIN
Derive Lagrangians, EOM, potentials
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import sympy as sp
import os

os.makedirs('docs', exist_ok=True)

# Symbols
x = sp.symbols('x')
Phi = sp.Function('\\Phi')(x)
Phi_star = sp.Function('\\Phi^*')(x)
J0, m, lambda_ = sp.symbols('J_0 m \\lambda', real=True)

# Example: ALADIN Lagrangian density (symbolic)
L = sp.Rational(1,2) * sp.diff(Phi_star, x) * sp.diff(Phi, x) - sp.Rational(1,2)*m**2 * (Phi_star * Phi) - sp.Rational(1,4)*lambda_ * (Phi_star * Phi)**2 + J0 * sp.re(Phi)

# Print
print("ALADIN Lagrangian density:")
sp.pprint(L)

# Save LaTeX
latex_L = sp.latex(L)
with open('docs/sympy_aladin_lagrangian.tex', 'w') as f:
    f.write(latex_L)

# Derive EOM symbolically
eom = sp.diff(L, Phi_star) - sp.diff(sp.diff(L, sp.diff(Phi_star, x)), x)
print("\nEquation of motion for Φ:")
sp.pprint(sp.simplify(eom))

print("SymPy symbolic computation — Lagrangians and EOM sealed")
