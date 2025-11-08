import numpy as np
import matplotlib.pyplot as plt

# Parameters
H0 = 75.2          # km/s/Mpc
omega = 2 * np.pi * 43  # 43 Hz

# Redshift grid
z = np.logspace(-2, 1, 300)
a = 1 / (1 + z)

# Octonionic scalar evolution: a(t) = exp(ω t / 4)
# → t = 4 ln(a) / ω
t = 4 * np.log(a) / omega

# H(t) = da/dt / a = ω/4
H_t = np.full_like(t, omega / 4)

# H(z) = H(t(z))
H_z = H_t

# Plot
plt.figure(figsize=(10,6))
plt.plot(z, H_z, 'purple', lw=3, label='Octonionic H(z) = ω/4')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Octonionic Hubble Law — Constant H(z)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('octonionic_hubble.png', dpi=300)
plt.close()

print("Octonionic Hubble plot saved: octonionic_hubble.png")
