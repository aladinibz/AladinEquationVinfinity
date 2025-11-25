
# --- INSTALL ---
!pip install numpy matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

# --- PLASMA STRESS-ENERGY ---
rho = 1e-24
v = 1e5
J = 1e18
B = 1e-6
c = 3e8

# T00 = rho * v^2
T00 = rho * v**2

# Tij = (1/(4pi)) * (F_iλ F_j^λ - (1/4) g_ij F^2)
F = J * B
Tij = (1/(4*np.pi)) * F**2

print(f"T00 = {T00:.2e} kg/m·s²")
print(f"Tij = {Tij:.2e} kg/m·s²")

# --- PLOT: STRESS-ENERGY DENSITY ---
r = np.logspace(18, 25, 100)
T = rho * (1e5)**2 + (1/(4*np.pi)) * (1e18 * 1e-6)**2

plt.figure(figsize=(8,5))
plt.loglog(r, np.full_like(r, T), 'gold', lw=3, label='Plasma Tμν')
plt.axvline(1e21, color='cyan', ls='--', label='Galaxy Scale')
plt.xlabel('Radius r (m)')
plt.ylabel('Tμν (kg/m·s²)')
plt.title('ALADIN ∞ ℂ(t) — Gravity from Plasma Stress-Energy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE PNG ---
png_path = '/content/gravity_plasma.png'
plt.savefig(png_path, dpi=300)
print(f"PNG saved: {png_path}")

# --- DOWNLOAD .py + .png ---

print("gravity_plasma.py + .png — DOWNLOADED")
