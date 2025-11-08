import numpy as np
import matplotlib.pyplot as plt

# Clifford Cl(3,0) — 4D spacetime algebra
H0 = 75.2
omega = 2 * np.pi * 43

# Redshift grid
z = np.logspace(-2, 1, 300)
a = 1 / (1 + z)

# Clifford scalar: a(t) = exp(ω t / 4)
t = 4 * np.log(a) / omega

# H(t) = da/dt / a = ω/4
H_t = np.full_like(t, omega / 4)

# H(z)
H_z = H_t

# Plot
plt.figure(figsize=(10,6))
plt.plot(z, H_z, 'purple', lw=3, label='Cl(3,0) H(z) = ω/4')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Clifford Algebra Hubble Law')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('clifford_hubble.png', dpi=300)
plt.close()

print("Clifford Hubble plot saved: clifford_hubble.png")
