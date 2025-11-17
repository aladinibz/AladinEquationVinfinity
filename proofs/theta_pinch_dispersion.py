import numpy as np
import matplotlib.pyplot as plt

# Theta-pinch dispersion relation (MHD)
k_par = np.logspace(-10, -4, 600)      # parallel wavenumber
k_perp = 1e-6                           # small perpendicular component
B_z = 1e-6                              # axial field (T)
B_theta = 0.8e-6                        # azimuthal field (T)
rho = 1e-24                             # plasma density (kg/m³)
cs = 1e5                                # sound speed (m/s)
mu0 = 4 * np.pi * 1e-7

vAz = B_z / np.sqrt(mu0 * rho)
omega2 = (k_par**2 * vAz**2) + (k_perp**2 * cs**2 * (1 - B_theta**2 / (B_z**2 + B_theta**2)))

plt.figure(figsize=(10,6))
plt.loglog(k_par, np.sqrt(np.abs(omega2)), 'gold', lw=4)
plt.axhline(0, color='red', ls='--', lw=2)
plt.xlabel('k_∥ (m⁻¹)')
plt.ylabel('ω (rad/s)')
plt.title('ALADIN ∞ C(t) — Theta-Pinch Dispersion Relation (Stable)')
plt.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('theta_pinch_dispersion.png', dpi=300, bbox_inches='tight')
print("theta_pinch_dispersion.png saved — stable plasma waves")
