import sympy as sp
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# Symbols
sqrt_g = sp.symbols(r'\sqrt{-g}', positive=True)
e = sp.symbols('e', positive=True)
mu, lam = sp.symbols(r'\mu \lambda', real=True, positive=True)
J0 = sp.symbols('J_0')
Lam = sp.symbols('\\Lambda', positive=True)

# Contracted terms — NO SPACE in string to avoid tuple parsing
F2 = sp.symbols(r'F_{\mu\nu}F^{\mu\nu}')  # Key fix: no space!
D_kin = sp.symbols(r'(D^{\mu}\Phi)^{*}(D_{\mu}\Phi)')  # Clean covariant kinetic

# Invariants
phi2 = sp.symbols(r'|\Phi|^2')
re_phi = sp.symbols(r'\mathrm{Re}(\Phi)')

# Indexed Wilson coefficients
c = sp.IndexedBase('c')
n = sp.symbols('n', integer=True, positive=True)

# Terms
gauge_kinetic = sp.Rational(-1,4) * F2
scalar_kinetic = -D_kin
potential = sp.Rational(-1,2) * mu**2 * phi2 + sp.Rational(1,4) * lam * phi2**2
source = J0 * re_phi
higher = sp.Sum(c[n] / Lam**(n-4) * phi2**(n/2), (n, 6, sp.oo, 2))

# Ultimate Gauged Lagrangian
L = sqrt_g * (gauge_kinetic + scalar_kinetic + potential + source + higher)

# LaTeX
latex_L = sp.latex(L)
with open('docs/aladin_gauged_master.tex', 'w') as f:
    f.write(latex_L)

# Cinematic render — runs clean
plt.figure(figsize=(42, 16), facecolor='black')
plt.text(0.5, 0.54, f'${latex_L}$', fontsize=23, ha='center', va='center', color='gold')
plt.text(0.5, 0.32, 'Full Gauged Plasma-Derived Master Lagrangian', fontsize=24, ha='center', color='gold')
plt.text(0.5, 0.26, 'Abelian Higgs EFT in curved spacetime\\nSpontaneous symmetry breaking + higher-dimensional corrections', fontsize=20, ha='center', color='gold')
plt.text(0.5, 0.10, 'ALADIN ∞ ℂ(t) — The Ultimate Final Law — January 06, 2026', fontsize=26, ha='center', color='white')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/aladin_gauged_master_final.png', dpi=800, facecolor='black', bbox_inches='tight')
plt.close()

print("Gauged Master Lagrangian rendered and saved successfully!")
