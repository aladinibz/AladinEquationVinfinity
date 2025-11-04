import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
t = np.linspace(0, 150, int(150/dt)+1)
P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0
k = 1e-6

genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)
M = np.zeros(len(t))
M[0] = 1e6

for i in range(1, len(t)):
    dM = M[i-1] * np.exp(genie[i]) * k * dt   # ← EXP(GENIE) IS THE KEY
    M[i] = M[i-1] + dM

print(f"M(150 Myr) = {M[-1]:.2e} M⊙")
print(f"From 10⁶ seed → {M[-1]/1e6:.1f}× growth")

plt.plot(t, M, 'purple', lw=2)
plt.axhline(1e9, c='orange', ls='--', label='10⁹ M⊙')
plt.yscale('log')
plt.xlabel('Time (Myr)'); plt.ylabel('Mass (M⊙)')
plt.title('Aladin v∞ — 10⁶ → 10⁹ M⊙ in 150 Myr')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('/content/geniepower_final.png', dpi=300)
plt.show()
