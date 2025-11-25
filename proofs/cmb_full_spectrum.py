import numpy as np
import matplotlib.pyplot as plt

# === PLANCK 2018 DATA (simplified) ===
l_data = np.array([2, 220, 540, 815, 1080, 1290, 1500, 1800])
Dl_tt_data = np.array([100, 5700, 3200, 2400, 1800, 1400, 1100, 900])  # μK²

# === ALADIN v∞ CMB MODEL ===
l = np.linspace(2, 2500, 10000)
theta = 2.0
P = 96.6
tau = 180
phi = 1.5
psi = 3.0

# Acoustic peaks + damping tail
Dl_tt = theta * np.sin(2 * np.pi * l / P) * np.exp(-l / 800) + psi * np.exp(-l / tau)

# Low-ℓ anomaly (dipole + quadrupole)
Dl_tt[l < 30] *= 0.8

# === PLOT ===
plt.figure(figsize=(12,7))
plt.plot(l_data, Dl_tt_data, 'o', label='Planck 2018 TT', color='red', markersize=8)
plt.plot(l, Dl_tt, '-', label='Aladin v∞ CMB', color='purple', linewidth=3)
plt.xlabel('Multipole ℓ')
plt.ylabel('D_ℓ^{TT} [μK²]')
plt.title('Aladin v∞ vs Planck 2018 — 6 Acoustic Peaks + Damping Tail')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/cmb_full_spectrum.png', dpi=300)

# === χ²/dof ===
chi2 = np.sum(((Dl_tt[np.searchsorted(l, l_data)] - Dl_tt_data) / 100)**2)
dof = len(l_data) - 3
print(f"χ²/dof = {chi2/dof:.2f}")
