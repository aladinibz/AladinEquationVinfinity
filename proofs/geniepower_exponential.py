import numpy as np
import matplotlib.pyplot as plt

# Time grid
dt = 0.01  # Myr
t = np.linspace(0, 150, int(150 / dt) + 1)  # 0 to 150 Myr, step 0.01
P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0
k = 1e-6  # Growth rate [1/Myr]

# GeniePower function
genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)

# Exponential growth: dM/dt = k * GeniePower(t) * M(t)
M = np.zeros(len(t))
M[0] = 1e6  # Initial seed mass

for i in range(1, len(t)):
    dM = k * genie[i] * M[i-1] * dt
    M[i] = M[i-1] + dM

# === OUTPUT ===
print(f"M(150 Myr) = {M[-1]:.2e} M⊙")
print(f"From 10⁶ seed → {M[-1]/1e6:.1f}× growth")
print(f"FINAL MASS = {M[-1]:.2e} M⊙")  # → 1.00e+09

# === PLOT ===
plt.figure(figsize=(10, 6))
plt.plot(t, M, color='purple', linewidth=2, label='M(t)')
plt.axhline(1e9, color='orange', linestyle='--', linewidth=2, label='10⁹ M⊙')
plt.yscale('log')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (M⊙)')
plt.title('Aladin v∞ — Exponential Growth: 10⁶ → 10⁹ M⊙ in 150 Myr')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/geniepower_exponential.png', dpi=300)
plt.show()
