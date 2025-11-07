import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 10 Dwarf Galaxies — MOND fit
galaxies = ['DDO 154', 'UGC 2259', 'IC 2574', 'NGC 6822', 'WLM', 
            'Sextans A', 'Phoenix', 'Leo T', 'And XIX', 'Carina']
r_kpc = np.array([1.0, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2, 0.3, 1.5, 0.1])
M_sun = np.array([1e8, 5e7, 2e7, 1e8, 3e7, 1e7, 5e6, 8e6, 2e8, 3e6])
v_obs = np.array([45, 32, 25, 50, 30, 20, 15, 18, 55, 12])
v_err = np.array([3, 2, 2, 4, 3, 2, 1, 2, 5, 1])

G = 4.3e-3
a0 = 1.2e-10 * 3.086e19 / 1e6

r_pc = r_kpc * 1000
g_N = G * M_sun / r_pc**2
v_mond = np.sqrt(r_pc * np.sqrt(g_N * a0))

chi2 = np.sum(((v_obs - v_mond) / v_err)**2)
dof = len(v_obs) - 1
chi2_red = chi2 / dof
p_value = 1 - chi2.cdf(chi2, dof)

print(f"χ² = {chi2:.3f}, χ²_red = {chi2_red:.3f}, p = {p_value:.2e}")

plt.figure(figsize=(9,6))
plt.errorbar(r_kpc, v_obs, v_err, fmt='ko', capsize=5, label='Observed')
plt.plot(r_kpc, v_mond, 'purple', lw=3, label=f'Aladin v∞ (p={p_value:.1e})')
plt.xlabel('Radius (kpc)')
plt.ylabel('v (km/s)')
plt.title('10 Dwarf Galaxies — Aladin v∞ MOND Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dwarf_rotation_aladin.png', dpi=300)
plt.close()
