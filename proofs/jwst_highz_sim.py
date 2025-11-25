import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 150, 1000)
P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0
k = 5.8e-7

genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)
M = np.zeros(len(t))
M[0] = 1e6

for i in range(1, len(t)):
    dM = M[i-1] * np.exp(genie[i]) * k * (t[i] - t[i-1])
    M[i] = M[i-1] + dM

print(f"JWST z=20 M = {M[-1]:.2e} M⊙")

plt.plot(t, M, 'purple')
plt.yscale('log')
plt.title('JWST High-z Galaxies — Aladin v∞')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (M⊙)')
plt.savefig('/content/jwst_highz.png', dpi=300)
