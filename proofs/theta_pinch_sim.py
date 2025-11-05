import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
n = 500
r = np.linspace(0.1, 3, n)  # Radius (cm)
dr = r[1] - r[0]
B0 = 1.0  # External B-field (T)
mu0 = 4 * np.pi * 1e-7
n0 = 1e19  # Density (m^-3)
T0 = 1e5  # Initial temp (K)
kB = 1.38e-23
m_i = 1.67e-27
gamma = 5/3

# === INITIAL STATE ===
n_e = n0 * np.ones(n)
v_r = np.zeros(n)
T = T0 * np.ones(n)

# === TIME LOOP ===
dt = 1e-9
steps = 500

for step in range(steps):
    # External theta-pinch B = B0 (constant)
    B = B0 * np.ones(n)
    
    # Pinch force = -n_e * B^2 / (mu0 * r)
    F_pinch = -n_e * B**2 / (mu0 * r)
    
    # Pressure gradient
    P = n_e * kB * T
    dP_dr = np.gradient(P, dr)
    dP_dr[0] = dP_dr[1]
    
    # Net force
    F_net = F_pinch + dP_dr
    
    # Acceleration
    a = F_net / (n_e * m_i + 1e-20)
    
    # Velocity
    v_r += a * dt
    v_r = np.clip(v_r, -1e5, 1e5)
    
    # Density
    div_v = np.gradient(v_r, dr)
    div_v[0] = div_v[1]
    n_e += -dt * (n_e * div_v + v_r * np.gradient(n_e, dr))
    n_e = np.maximum(n_e, 1e16)
    
    # Temperature (adiabatic)
    dT_dt = (gamma - 1) * T * div_v
    T += dT_dt * dt
    T = np.maximum(T, 1e5)

# === PLOT ===
plt.figure(figsize=(10,6))
plt.semilogx(r, n_e / n0, 'purple', linewidth=3, label='Density')
plt.semilogx(r, T / 1e6, 'red', linewidth=3, label='Temperature')
plt.xlabel('Radius (cm)')
plt.ylabel('Density / n₀ or T (MK)')
plt.title('Theta-Pinch — Density + Temp Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/theta_pinch_sim.png', dpi=300)
plt.show()

print(f"Max density: {np.max(n_e)/n0:.1e} × n₀")
print(f"Max temp: {np.max(T)/1e6:.1f} MK")
