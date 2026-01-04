"""
Inverse Seesaw Symbolic Derivation — Exact Eigenvalues (Single Generation)
ALADIN ∞ ℂ(t) — The Final Law
January 04, 2026
"""

import sympy as sp
import os

os.makedirs('docs', exist_ok=True)

# Symbols
m_D, M, mu = sp.symbols('m_D M \\mu', real=True, positive=True)

# Mass matrix
matrix = sp.Matrix([
    [0, m_D, 0],
    [m_D, 0, M],
    [0, M, mu]
])

# Characteristic equation det(M - λ I) = 0
lam = sp.symbols('\\lambda')
char_eq = (matrix - lam * sp.eye(3)).det()

# Expand
char_poly = sp.expand(char_eq)

# Print
print("Characteristic polynomial:")
sp.pprint(char_poly)

# LaTeX
latex_poly = sp.latex(char_poly)

with open('docs/inverse_seesaw_symbolic.tex', 'w') as f:
    f.write(latex_poly)

print("\nSymbolic derivation saved — exact cubic for eigenvalues")
