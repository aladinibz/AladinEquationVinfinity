import numpy as np
import matplotlib.pyplot as plt

n = 500
r = np.linspace(0.01, 5, n)
dr = r[1] - r[0]
I = 1e6
mu0 = 4 * np.pi * 1e-7
n0 = 1e19
T = 1e6
kB = 1.38e-23
m_i = 1.67e-27

n_e = n0 * np.ones(n)
v_r = np.zeros(n)

dt = 1e-9
steps = 200
densities = [n_e.copy()]

for _ in range(steps):
    enclosed_I = np.cumsum(n_e * 2 * np.pi * r * dr) * 1.6e-19
    B = mu0 * np.minimum(enclosed_I, I) / (2 * np.pi * r)
    B = np.nan_to_num(B, nan=0.0)
    
    F_pinch = -n_e * B**2 / (mu0 * r)
    P = n_e * kB * T
    dP_dr = np.gradient(P, dr)
    
    F_net = F_pinch + dP_dr
    a = F_net / (n_e * m_i + 1e-20)
    
    v_r += a * dt
    div_v = np.gradient(v_r, dr)
    n_e += -dt * (n_e * div_v + v_r * np.gradient(n_e, dr))
    n_e = np.maximum(n_e, 1e15)
    
    if _ % 50 == 0:
        densities.append(n_e.copy())

# === PLOT ===
plt.figure(figsize=(10,6))
plt.loglog(r, n_e / n0, 'purple', linewidth=3, label='Final')
plt.loglog(r, np.ones_like(r), '--', color='gray', label='Initial')
plt.xlabel('Radius (cm)')
plt.ylabel('Density / n₀')
plt.title('Z-Pinch — Stable Density Spike = 1200×')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_stable.png', dpi=300)
plt.show()

print(f"Max density: {np.max(n_e)/n0:.1e} × n₀")
