import numpy as np
import matplotlib.pyplot as plt

# Lorentz bivector expansion in Cl(1,3)
H0 = 75.2
omega = 2 * np.pi * 43

# Redshift grid
z = np.logspace(-2, 1, 300)
a = 1 / (1 + z)

# Bivector B = ω/4 * (γ0∧γ1 + γ0∧γ2 + γ0∧γ3)
# Expansion: a(t) = exp(B t)
# |B| = ω/4 → a(t) = exp((ω/4) t)

t = 4 * np.log(a) / omega
H_t = np.full_like(t, omega / 4)

# H(z)
H_z = H_t

# Plot
plt.figure(figsize=(10,6))
plt.plot(z, H_z, 'purple', lw=3, label='Lorentz Bivector H(z) = ω/4')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Lorentz Bivector Expansion — Cl(1,3)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lorentz_bivector_hubble.png', dpi=300)
plt.close()

print("Lorentz bivector plot saved: lorentz_bivector_hubble.png")
