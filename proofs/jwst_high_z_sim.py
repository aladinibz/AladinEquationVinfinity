import numpy as np
import matplotlib.pyplot as plt

# JWST high-z redshifts
z_jwst = np.array([10.5, 11.0, 12.33, 13.2, 14.0, 15.0])

# Aladin v∞: H0 = 75.2, n = 0.25
H0 = 75.2
n = 0.25
H_model = H0 * (z_jwst / (1 + z_jwst))**n

# Realistic JWST errors (~15-25%)
e_rel = 0.15 + 0.05 * (z_jwst - 10)  # increases with z
e_jwst = H_model * e_rel

# Add Gaussian noise
rng = np.random.default_rng(42)
H_jwst = H_model + rng.normal(0, e_jwst)

# Save as CSV
np.savetxt('jwst_high_z_data.csv', np.column_stack([z_jwst, H_jwst, e_jwst]),
           delimiter=',', header='z,H(z),error', comments='')

# Plot
z_fine = np.logspace(-2, 1.2, 200)
H_fit = H0 * (z_fine / (1 + z_fine))**n

plt.figure(figsize=(10,6))
plt.errorbar(z_jwst, H_jwst, e_jwst, fmt='ks', capsize=5, label='Simulated JWST')
plt.plot(z_fine, H_fit, 'gold', lw=3, label=f'Aladin v∞ H0={H0:.1f}, n={n}')
plt.xscale('log'); plt.xlabel('z'); plt.ylabel('H(z)')
plt.title('Simulated JWST High-z — Aladin v∞')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('jwst_high_z_sim.png', dpi=300)
plt.close()

print("JWST sim saved: jwst_high_z_data.csv + plot")
