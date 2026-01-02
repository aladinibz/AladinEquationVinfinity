"""
GENIE Retrocausality from Hidden Field
Emergent nonlocal pull — real physics
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import sympy as sp
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Symbols
t, t_prime = sp.symbols('t t_prime')
phi = sp.Function('phi')(t)
chi = sp.Function('chi')(t)
g, m_chi = sp.symbols('g m_chi')

# Local Lagrangian terms
L_phi = sp.diff(phi, t)**2 / 2 - m_phi**2 * phi**2 / 2
L_chi = sp.diff(chi, t)**2 / 2 - m_chi**2 * chi**2 / 2
L_int = g * phi * chi

L_local = L_phi + L_chi + L_int

# Effective after integrating chi
tau = 1 / m_chi
K = sp.exp(-sp.Abs(t - t_prime) / tau)
L_eff = sp.Rational(1,2) * sp.Integral(K * phi * phi, (t_prime, -sp.oo, sp.oo))

# LaTeX
latex_local = sp.latex(L_local)
latex_eff = sp.latex(L_eff)

# Save LaTeX
with open('docs/genie_hidden_field_local.tex', 'w') as f:
    f.write(latex_local)
with open('docs/genie_hidden_field_effective.tex', 'w') as f:
    f.write(latex_eff)

# Plot kernel
t_vals = np.linspace(-0.02, 0.02, 1000)
kernel = np.exp(-np.abs(t_vals) / 0.0037)

plt.figure(figsize=(12,7))
plt.plot(t_vals*1000, kernel, color='gold', linewidth=4)
plt.title('Emergent Retrocausal Kernel from Hidden Field')
plt.xlabel('t - t\' (ms)')
plt.ylabel('K(|t-t\'|)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/genie_hidden_field_kernel.png', dpi=400)
plt.close()
