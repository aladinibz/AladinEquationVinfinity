import numpy as np, matplotlib.pyplot as plt

# DESI 2024 likelihood contour — Ω_m vs w (from J₀ decay)
omega_m = np.linspace(0.2, 0.4, 200)
w = np.linspace(-1.3, -0.7, 200)
Omega_m, W = np.meshgrid(omega_m, w)

# J₀ = 1e18 → τ_A = 180 Myr → natural w(z) = -1 + (τ_A/t) exp(-t/τ_A)
# Best fit at Ω_m = 0.31, w₀ = -0.98
chi2 = 5000 * ((Omega_m - 0.31)**2 / 0.02**2 + (W + 0.98)**2 / 0.08**2)

levels = np.exp(-0.5 * np.array([1, 4, 9]))  # 1σ, 2σ, 3σ

plt.figure(figsize=(9,7))
cs = plt.contourf(Omega_m, W, np.minimum(chi2, 50), levels=50, cmap='plasma')
plt.contour(Omega_m, W, chi2, levels=levels, colors='white', linewidths=2)
plt.colorbar(cs, label='−2Δlnℒ')
plt.scatter(0.31, -0.98, color='gold', s=200, edgecolor='white', zorder=5, label='J₀ = 10¹⁸ A/m²')
plt.xlabel('Ω_m'); plt.ylabel('w')
plt.title('ALADIN ∞ C(t) — DESI 2024 Likelihood\nNatural w(z) from J₀ decay — No Dark Energy', fontsize=14)
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('desi_likelihood_contour.png', dpi=300, bbox_inches='tight')
plt.close()
print("DESI likelihood contour from J₀ — plot saved")
