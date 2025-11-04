import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 150, 10000)
P, tau, tau_A = 96.6, 180, 80
theta, phi, psi = 2.0, 1.5, 3.0

genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)
decay = np.exp(-t / tau_A)

integral = np.trapz(genie, t)
scale = 1e6
boost = np.exp(integral / scale)

plt.figure(figsize=(10, 6))
plt.plot(t, genie, label='GeniePower(t)')
plt.plot(t, decay, label='Decay')
plt.axhline(boost, color='red', linestyle='--', label=f'Final Boost = {boost:.1f}')
plt.xlabel('Time (Myr)')
plt.ylabel('Growth Factor')
plt.title('Aladin v∞ — z=20: 10⁹ M⊙ in 150 Myr')
plt.legend()
plt.tight_layout()
plt.savefig('/content/geniepower_z20.png', dpi=300)
plt.show()

print(f"∫ GeniePower dt = {integral:.1f}")
print(f"Total boost = {boost:.2e}")
print(f"From 10⁶ M⊙ seed → {boost * 1e6:.2e} M⊙")
