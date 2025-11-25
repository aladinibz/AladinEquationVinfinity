import numpy as np
import matplotlib.pyplot as plt

# Time grid — fixed to avoid off-by-one
dt = 0.01
n_steps = int(150 / dt) + 1
t = np.linspace(0, 150, n_steps)

P, tau = 96.6, 180
theta, phi, psi = 2.0, 1.5, 3.0
k = 1e-6  # 1/Myr

# GeniePower(t)
genie = theta * np.log(1 + t) + phi * np.sin(2 * np.pi * t / P) + psi * np.exp(-t / tau)

# Mass evolution
M = np.zeros(n_steps)
M[0] = 1e6  # seed

for i in range(1, n_steps):
    dM = M[i-1] * np.exp(genie[i]) * k * dt
    M[i] = M[i-1] + dM

# === OUTPUT ===
print(f"M(150 Myr) = {M[-1]:.2e} M⊙")
print(f"From 10⁶ seed → {M[-1]/1e6:.1f}× growth")

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
plt.savefig('/content/geniepower_final.png', dpi=300)
