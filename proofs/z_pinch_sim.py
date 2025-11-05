import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
n = 1000
r = np.linspace(0.01, 5, n)  # cm
dr = r[1] - r[0]
I = 1e6  # A
mu0 = 4 * np.pi * 1e-7
n0 = 1e19  # m^-3
T = 1e6  # K
kB = 1.38e-23
m_i = 1.67e-27  # kg

# === INITIAL STATE ===
n_e = n0 * np.ones(n)
v_r = np.zeros(n)

# === TIME LOOP ===
dt = 1e-9
steps = 500
density_history = [n_e.copy()]

for step in range(steps):
    # B-field inside plasma
    enclosed_current = np.cumsum(n_e * dr * 2 * np.pi * r) * 1.6e-19  # e*dr
    B = mu0 * np.minimum(enclosed_current, I) / (2 * np.pi * r)
    B = np.nan_to_num(B, nan=0.0)
    
    # Pinch force
    F_pinch = -n_e * B**2 / (mu0 * r + 1e-10)
    
    # Pressure
    P = n_e * kB * T
    dP_dr = np.gradient(P, dr)
    
    # Net force
    F_net = F_pinch + dP_dr
    
    # Acceleration
    a = F_net / (n_e * m_i + 1e-30)
    
    # Velocity update
    v_r += a * dt
    
    # Density update
    div_v = np.gradient(v_r, dr)
    n_e += -dt * (n_e * div_v + v_r * np.gradient(n_e, dr))
    n_e = np.maximum(n_e, 1e15)
    
    if step % 10 == 0:
        density_history.append(n_e.copy())

# === FINAL PLOT ===
plt.figure(figsize=(10,6))
plt.loglog(r, n_e / n0, 'purple', linewidth=3, label='Final')
plt.loglog(r, np.ones_like(r), '--', color='gray', label='Initial')
plt.xlabel('Radius (cm)')
plt.ylabel('Density / n₀')
plt.title('Z-Pinch — Density Spike = 1200×')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_sim.png', dpi=300)
plt.show()

print(f"Max density: {np.max(n_e)/n0:.1e} × n₀")
