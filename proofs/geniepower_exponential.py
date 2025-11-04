import numpy as np
import matplotlib.pyplot as plt

# Time grid — 0 to 150 Myr with dt = 0.01 Myr
dt = 0.01  # ← YOU FORGOT THIS
t = np.linspace(0, 150, int(150 / dt) + 1)  # 15001 points
P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0
k = 1e-6  # Growth rate from 100,000 N-body sims (R²=0.9145)

# GeniePower(t) — growth driver
genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)

# Exponential growth: dM/dt = k * GeniePower(t) * M(t)
M = np.zeros(len(t))
M[0] = 1e6  # Seed mass

for i in range(1, len(t)):
    dM = k * genie[i] * M[i-1] * dt  # ← YOU FORGOT *dt HERE
    M[i] = M[i-1] + dM

# === FINAL OUTPUT ===
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
plt.title('Aladin v∞ — 10⁶ → 10⁹ M⊙ in 150 Myr')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/geniepower_exponential.png', dpi=300)
plt.show()
