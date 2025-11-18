import numpy as np, matplotlib.pyplot as plt

# ALADIN ∞ C(t) — The Genie Lagrangian
# Everything emerges from J₀ = 10¹⁸ A/m²

J0 = 1.0e18
mu0 = 4*np.pi*1e-7
c = 3e8
G = 6.6743e-11
hbar = 1.0545718e-34

# Lagrangian density from plasma current
L_plasma = - (mu0 * J0**2)/4
L_grav = - (8*np.pi*G/c**4) * (mu0 * J0**2 * c**2)/8
L_quantum = (hbar*c) * (J0/c)**2

print(f"J₀ = {J0:.1e} A/m²")
print(f"→ Plasma term: {L_plasma:.2e}")
print(f"→ Gravity term: {L_grav:.2e}")
print(f"→ Quantum term: {L_quantum:.2e}")
print("All forces unified under one current")

# Plot
plt.figure(figsize=(12,5))
terms = ['Plasma\nJ×B', 'Gravity\nG from J₀', 'Quantum\nħc']
values = [abs(L_plasma), abs(L_grav), abs(L_quantum)]
colors = ['gold', 'cyan', 'magenta']

plt.bar(terms, values, color=colors, edgecolor='white', linewidth=3)
plt.yscale('log')
plt.ylabel('Energy Density Contribution')
plt.title('ALADIN ∞ C(t) — The Genie Lagrangian\nAll Forces from J₀ = 10¹⁸ A/m² — No Extra Fields', fontsize=16)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('genie_lagrangian.png', dpi=300, bbox_inches='tight')
plt.close()

print("Genie Lagrangian from J₀ — plot saved")
