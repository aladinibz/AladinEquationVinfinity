import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 150, 10000)
P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0

genie = theta * np.log(1 + t) + phi * np.sin(2*np.pi*t/P) + psi * np.exp(-t/tau)
integral = np.trapz(genie, t)
scale = 1e6
boost = np.exp(integral / scale)

print(f"∫ GeniePower dt = {integral:.2e} Myr")  # → 1.57e+06
print(f"Boost = exp({integral:.2e}/{scale:.0e}) = {boost:.2f}")
print(f"From 10⁶ M⊙ → {boost*1e6:.2e} M⊙")

plt.plot(t, genie, label='GeniePower(t)')
plt.axhline(boost, c='red', ls='--', label=f'Boost = {boost:.1f}')
plt.xlabel('Time (Myr)')
plt.ylabel('Growth Factor')
plt.title('Aladin v∞ — z=20: 4.8×10⁶ M⊙ in 150 Myr')
plt.legend()
plt.tight_layout()
plt.savefig('plots/geniepower_z20.png', dpi=300)
plt.show()
