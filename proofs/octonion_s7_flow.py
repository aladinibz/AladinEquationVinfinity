import numpy as np
import matplotlib.pyplot as plt

# Parameters
H0 = 75.2          # km/s/Mpc
omega = 2 * np.pi * 43  # 43 Hz
e_coeff = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])  # e0 to e7

# Time grid (0 to 13.8 Gyr)
t = np.linspace(0, 4.35e17, 1000)  # seconds
q_t = np.sum(e_coeff * np.cos(omega * t[:, None]), axis=1)  # real part only
a_t = np.abs(q_t) + 1e-10  # avoid zero

# Hubble H(t)
H_t = np.gradient(a_t, t) / a_t

# Redshift z(t)
z_t = np.maximum(1/a_t - 1, 1e-3)  # clip low z

# H(z)
H_z = H_t * (1 + z_t)

# Plot
z_plot = np.logspace(-2, 1, 300)
H_plot = H0 * (z_plot / (1 + z_plot))**0.25

plt.figure(figsize=(10,6))
plt.plot(z_plot, H_plot, 'gold', lw=3, label='Aladin v∞ (n=0.25)')
plt.plot(z_t, H_z, 'purple', lw=2, alpha=0.8, label='S(7) Octonionic Flow')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Octonionic Cosmic Flow — S(7) Extension')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('octonion_hubble.png', dpi=300)
plt.close()

print("S(7) flow saved: octonion_hubble.png")
