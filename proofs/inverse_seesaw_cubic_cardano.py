"""
Inverse Seesaw Cubic — Exact Symbolic Roots with Cardano
Single generation eigenvalues
ALADIN ∞ ℂ(t) — The Final Law
January 04, 2026
"""

import sympy as sp
import os

os.makedirs('docs', exist_ok=True)

# Symbols
m_D, M, mu = sp.symbols('m_D M \\mu', real=True, positive=True)
lam = sp.symbols('\\lambda')

# The cubic from det(M - λ I) = 0
cubic = lam**3 + M * lam**2 - (m_D**2 + M * mu) * lam + m_D**2 * mu

print("The characteristic cubic equation:")
sp.pprint(cubic)

# Depress the cubic: λ = y - M/3
y = sp.symbols('y')
depressed = cubic.subs(lam, y - M/3).expand()

print("\nDepressed cubic (y³ + p y + q = 0):")
sp.pprint(depressed)

# Coefficients p, q
p = depressed.coeff(y)
q = depressed.subs(y, 0)

print("\np =")
sp.pprint(p)
print("\nq =")
sp.pprint(q)

# Discriminant Δ = (q/2)² + (p/3)³
Delta = (q/2)**2 + (p/3)**3

print("\nDiscriminant Δ =")
sp.pprint(Delta)

# Cardano roots
u = (-q/2 + sp.sqrt(Delta))**(1/3)
v = (-q/2 - sp.sqrt(Delta))**(1/3)

root1 = u + v - M/3

# Other roots with cube roots of unity
omega = sp.exp(2*sp.pi*sp.I / 3)
root2 = omega * u + omega**2 * v - M/3
root3 = omega**2 * u + omega * v - M/3

print("\nExact Cardano roots:")
print("λ₁ =")
sp.pprint(root1)
print("\nλ₂ =")
sp.pprint(root2)
print("\nλ₃ =")
sp.pprint(root3)

# LaTeX output
latex_roots = sp.latex(root1) + "\\\\\n" + sp.latex(root2) + "\\\\\n" + sp.latex(root3)

with open('docs/inverse_seesaw_cardano_roots.tex', 'w') as f:
    f.write(latex_roots)

print("\nSymbolic Cardano roots saved — exact eigenvalues")
