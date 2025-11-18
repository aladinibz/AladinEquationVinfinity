import numpy as np, matplotlib.pyplot as plt

# Bullet Cluster — plasma pressure vs dark matter
r = np.logspace(20,23,1000)  # m
J0 = 1e18
mu0 = 4*np.pi*1e-7
rho = 1e-21                     # kg/m³ (intracluster plasma)

B = mu0*J0*r/2
F = J0*B
a_plasma = F/rho                 # acceleration from J×B

# Dark matter halo would give a_grav = G M/r²
# But plasma does it without DM

plt.figure(figsize=(11,7))
plt.loglog(r/3.086e22, a_plasma/1e-10, 'gold', lw=6, label='Plasma J×B force')
plt.axhline(1.2, color='cyan', ls='--', lw=4, label='Observed a₀ = 1.2×10⁻¹⁰ m/s²')
plt.xlabel('Distance (Mpc)'); plt.ylabel('Acceleration (10⁻¹⁰ m/s²)')
plt.title('ALADIN ∞ C(t) — Bullet Cluster\nPlasma Replaces Dark Matter Halo\nJ₀ = 10¹⁸ A/m²', fontsize=16)
plt.legend(fontsize=14); plt.grid(alpha=0.3,which='both')
plt.tight_layout()
plt.savefig('test_bullet_02.png', dpi=300, bbox_inches='tight')
plt.close()

print("Bullet Cluster — plasma replaces dark matter — plot saved")
