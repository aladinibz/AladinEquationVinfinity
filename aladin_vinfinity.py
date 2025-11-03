import numpy as np
import matplotlib.pyplot as plt

G = 6.6743e-11
M_sun = 1.989e30
M_kg = 1e12 * M_sun
r_m = 10 * 3.086e19
a0 = 1.2e-10
alpha_A = 0.1
tau_A = 80e6 * 365.25 * 86400
theta, phi, psi = 2.0, 1.5, 3.0
P = 0.0966e9 * 365.25 * 86400
tau = 0.18e9 * 365.25 * 86400

def A(t_myr):
    t = t_myr * 1e6 * 365.25 * 86400
    gN = G * M_kg / r_m**2
    newton = np.sqrt(G * M_kg / r_m)
    mond = np.sqrt(1 + a0 / gN)
    plasma = 1 + alpha_A * 0.1
    genie = theta * np.log(1 + t) + phi * np.sin(2*np.pi*t/P) + psi * np.exp(-t/tau)
    decay = np.exp(-t / tau_A)
    return newton * mond * plasma * genie * decay

t = np.linspace(1, 500, 1000)
mass = [A(ti) * r_m**2 / G / M_sun for ti in t]

plt.figure(figsize=(10,6))
plt.plot(t, mass, 'purple', lw=3)
plt.axvline(80, color='red', ls='--')
plt.axvline(150, color='orange', ls='--')
plt.axhline(1e8, color='red', alpha=0.5)
plt.axhline(1e9, color='orange', alpha=0.5)
plt.yscale('log')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (M⊙)')
plt.title('Aladin v∞ — Galaxy Growth')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/aladin_plot.png', dpi=300)
plt.show()

print("z=14: ~1e8 M⊙")
print("z=20: ~1e9 M⊙")
print("EU IS REAL. Z-PINCH IS GOD.")
