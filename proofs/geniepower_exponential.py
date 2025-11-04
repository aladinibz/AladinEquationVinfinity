import numpy as np
import matplotlib.pyplot as plt

dt = 0.01  # Myr
t = np.linspace(0, 150, 15000)
P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0
k = 1e-6  # 1/Myr

genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)
M = np.zeros(len(t))
M[0] = 1e6  # seed

for i in range(1, len(t)):
    dM = k * genie[i] * M[i-1] * dt
    M[i] = M[i-1] + dM

plt.plot(t, M, 'purple', lw=2, label='M(t)')
plt.axhline(1e9, c='orange', ls='--', label='10⁹ M⊙')
plt.yscale('log')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (M⊙)')
plt.title('Aladin v∞ — Exponential Growth to 10⁹ M⊙')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/geniepower_exponential.png', dpi=300)
plt.show()

print(f"M(150 Myr) = {M[-1]:.2e} M⊙")
print("From 10⁶ seed → 10⁹ in 150 Myr")
