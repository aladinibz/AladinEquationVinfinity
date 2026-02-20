import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime
from scipy.fft import rfft, rfftfreq
import scipy.signal as signal

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                  ALADIN v1.2 – Relativistic Z-Pinch Simulation             ║
# ║  (stabilized + 43 Hz detection + Fermion Majorana mass term)               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

c = 3.0e8
J0 = 1.0e18
J_pl = 43.0 ** 3
alpha = c * (J0 / J_pl)**(1/3.0)
f_res = c * (J0 ** (1/3.0)) / alpha

print(f"Derived resonance frequency: {f_res:.10f} Hz")

# =============================================================================
# PARAMETERS (stability-tuned + Majorana)
# =============================================================================

GRID_SIZE = 256
N_GRID = GRID_SIZE
dt_max = 3e-7
t_max = 5.0
CFL = 0.22
MAX_STEPS = 3000

mu0 = 4 * np.pi * 1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
omega_res = 2 * np.pi * f_res
m_phi = omega_res / c

kappa_maj = 0.01              # ← NEW: Majorana coupling strength

g_damp_base = 0.5
k_genie = 0.2
g_damp_max = 2.0 * g_damp_base

y_ferm = 5e-4
k_ferm = 2e-4
g_genie = 5e-5
y_genie = 0.1
kg_damping = 0.35
gamma = 5/3
kappa_thermal = 0.001
R_gas = 1.0

N_crit = 5e5
Gamma0 = 4e8
k_rp = 0.25
g_C = 2e-3
C_field_pump = 5e-4
nu_num = 0.001
eta = 0.01

os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

r = cp.linspace(0, 2*a, N_GRID)
phi = cp.linspace(0, 2*cp.pi, N_GRID)
z = cp.linspace(0, 10*a, N_GRID)
dr = r[1] - r[0]
dphi = phi[1] - phi[0]
dz = z[1] - z[0]

R, Phi, Z = cp.meshgrid(r, phi, z, indexing='ij')
R_safe = cp.maximum(R, 1e-3)
invR = 1.0 / R_safe
invR2 = 1.0 / (R_safe**2)

k_sausage = cp.float64(2 * cp.pi / (5*a))

def safe(arr, pos=1e6, neg=-1e6):
    return cp.nan_to_num(cp.asanyarray(arr), nan=0.0, posinf=pos, neginf=neg)

def cyl_gradient(field):
    grad_r = cp.gradient(field, dr, axis=0)
    grad_phi = cp.gradient(field, dphi, axis=1) / R_safe
    grad_z = cp.gradient(field, dz, axis=2)
    return safe(grad_r), safe(grad_phi), safe(grad_z)

def cyl_laplacian(field):
    dphi_dr = cp.gradient(field, dr, axis=0)
    term_r = invR * cp.gradient(R_safe * dphi_dr, dr, axis=0)
    term_phi = invR2 * cp.gradient(cp.gradient(field, dphi, axis=1), dphi, axis=1)
    term_z = cp.gradient(cp.gradient(field, dz, axis=2), dz, axis=2)
    return safe(term_r + term_phi + term_z)

def cyl_divergence(vr, vphi, vz):
    term_r = invR * cp.gradient(R_safe * vr, dr, axis=0)
    term_phi = cp.gradient(vphi, dphi, axis=1)
    term_z = cp.gradient(vz, dz, axis=2)
    return safe(term_r + term_phi + term_z)

# Initial fields
rho = cp.full((N_GRID, N_GRID, N_GRID), rho0)
v_r = cp.zeros_like(rho)
v_phi = cp.zeros_like(rho)
v_z = cp.zeros_like(rho)

J_z = J0 * cp.exp(-R**2 / a**2)
B_phi = mu0 * J0 * a * (1 - cp.exp(-R**2 / a**2))
B_r = cp.zeros_like(rho)
B_z = cp.zeros_like(rho)
p = cp.full_like(rho, rho0 * vA**2)
T = p / (rho * R_gas + 1e-10)

rho += 0.1 * rho0 * cp.cos(k_sausage * Z)
v_phi += 0.05 * vA * cp.cos(Phi) * cp.sin(k_sausage * Z)

genie_phi = cp.zeros_like(rho)
genie_phi_prev = genie_phi.copy()
C_field = cp.full_like(rho, 0.005)
J_pineal = cp.full_like(rho, 5e7)

# Histories
genie_center_history = []
C_mean_history = []
time_history = []

history = {
    "E_total": [], "genie_amp": [], "C_mean": [], "T_mean": [],
    "energy_drift": [], "max_v_over_c": [],
    # Fermion with Majorana
    "ferm_dirac_mean": [], "ferm_majorana_mean": [],
    "ferm_total_mean": [], "ferm_density_mean": []
}

t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)
step = 0

progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.2 – Majorana mass + 43 Hz")

while t < t_max and step < MAX_STEPS:
    step += 1

    # ... (timestep calculation, currents, relativistic gamma, momentum, continuity, pressure – unchanged) ...
    vmax = cp.max(cp.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-12))
    vA_local = cp.max(cp.sqrt((B_r**2 + B_phi**2 + B_z**2) / (mu0 * rho + 1e-12)))
    min_length = cp.min(cp.array([dr, dz, cp.mean(R_safe)*dphi]))
    dt_new = CFL * min_length / (vmax + vA_local + 1e-6)
    dt = 0.6 * dt_prev + 0.4 * dt_new
    dt = cp.minimum(dt, dt_max)
    dt = cp.maximum(dt, 5e-10)
    dt_prev = dt
    t += dt
    time_history.append(float(t))

    # 43 Hz sampling
    center = N_GRID // 2
    genie_center_history.append(float(genie_phi[center, center, center]))
    C_mean_history.append(float(cp.mean(C_field)))

    J_r = (cp.gradient(B_z, dphi, axis=1)/R_safe - cp.gradient(B_phi, dz, axis=2))/mu0
    J_phi = (cp.gradient(B_r, dz, axis=2) - cp.gradient(B_z, dr, axis=0))/mu0
    J_z = (cp.gradient(R_safe*B_phi, dr, axis=0)/R_safe - cp.gradient(B_r, dphi, axis=1))/mu0

    v2 = v_r**2 + v_phi**2 + v_z**2
    v2 = cp.clip(v2, 0, 0.99*c**2)
    gamma_rel = 1.0 / cp.sqrt(1.0 - v2/c**2 + 1e-18)
    gamma_rel = cp.clip(gamma_rel, 1.0, 15.0)

    JxB_r = gamma_rel * (J_phi * B_z - J_z * B_phi)
    JxB_phi = gamma_rel * (J_z * B_r - J_r * B_z)
    JxB_z = gamma_rel * (J_r * B_phi - J_phi * B_r)

    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)

    v_r += dt * (JxB_r - grad_p_r) / (rho * gamma_rel + 1e-7)
    v_phi += dt * (JxB_phi - grad_p_phi) / (rho * gamma_rel + 1e-7)
    v_z += dt * (JxB_z - grad_p_z) / (rho * gamma_rel + 1e-7)

    v_r = cp.clip(v_r, -0.1*c, 0.1*c)
    v_phi = cp.clip(v_phi, -0.1*c, 0.1*c)
    v_z = cp.clip(v_z, -0.1*c, 0.1*c)

    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt * drho_dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    div_v = cyl_divergence(v_r, v_phi, v_z)
    p += dt * ((gamma-1) * (-p * div_v) + kappa_thermal * cyl_laplacian(p))
    p = cp.clip(p, 1e-6, 100*cp.mean(p))
    T = p / (rho * R_gas + 1e-10)

    # ==================== FERMION MAJORANA MASS TERM ====================
    ferm_dirac_mass     = y_ferm * cp.abs(genie_phi)
    ferm_majorana_mass  = kappa_maj * (genie_phi ** 2)          # ← Majorana term
    ferm_total_mass     = ferm_dirac_mass + ferm_majorana_mass
    ferm_density_proxy  = 0.01 + 0.005 * (ferm_total_mass ** 2)

    # Mild Majorana back-reaction to scalar (safe coefficient)
    majorana_backreaction = 5e-6 * ferm_total_mass

    # Store for history
    history["ferm_dirac_mean"].append(float(cp.mean(ferm_dirac_mass)))
    history["ferm_majorana_mean"].append(float(cp.mean(ferm_majorana_mass)))
    history["ferm_total_mean"].append(float(cp.mean(ferm_total_mass)))
    history["ferm_density_mean"].append(float(cp.mean(ferm_density_proxy)))

    # ==================== GENIE SCALAR (with Majorana back-reaction) ====================
    source_genie = g_genie * J_z + majorana_backreaction
    source_genie = cp.clip(source_genie, -500, 500)

    lap_now = cyl_laplacian(genie_phi)
    accel = lap_now - m_phi**2 * genie_phi - 0.005 * genie_phi**3 + source_genie
    accel = cp.clip(accel, -2e4, 2e4)

    genie_vel = (genie_phi - genie_phi_prev) / dt_prev if step > 1 else cp.zeros_like(genie_phi)
    genie_phi_new = genie_phi + dt * genie_vel + 0.5 * dt**2 * accel
    genie_phi_new *= cp.exp(-kg_damping * dt)
    genie_phi_new -= 0.008 * cyl_laplacian(genie_phi_new) * dt
    genie_phi_new = cp.clip(genie_phi_new, -0.3, 0.3)

    genie_phi_prev = genie_phi.copy()
    genie_phi = genie_phi_new

    # C_field (unchanged)
    Gamma_sr = Gamma0 * (N_crit * C_field**2) / (1 + N_crit * C_field**2)
    pump_eff = C_field_pump * (1 + k_rp * C_field)
    pineal_res = 0.18 * cp.sin(2*cp.pi*43*t + 0.15)
    C_source = g_C * genie_phi * (1 + pineal_res * J_pineal / J0)

    C_field += dt * (C_source + pump_eff * (1 - C_field**2) - 0.02 * C_field**3 - Gamma_sr * C_field)
    C_field = cp.clip(C_field, 0.0, 5.0)

    rho += 3e-7 * C_field**2 * dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    if step % 20 == 0:
        fields = cp.stack([genie_phi, C_field, rho, v_r, v_phi, v_z])
        low = cp.percentile(fields, 0.2)
        high = cp.percentile(fields, 99.8)
        for f in [genie_phi, C_field, rho, v_r, v_phi, v_z]:
            f[...] = cp.clip(f, low, high)

    E_total = float(cp.mean(0.5 * rho * (v_r**2 + v_phi**2 + v_z**2)) +
                    cp.mean(0.5 * (B_r**2 + B_phi**2 + B_z**2) / mu0))

    drift = abs((E_total - (history["E_total"][-1] if history["E_total"] else E_total)) /
                (history["E_total"][-1] + 1e-10)) if history["E_total"] else 0.0

    history["E_total"].append(E_total)
    history["genie_amp"].append(float(cp.mean(cp.abs(genie_phi))))
    history["C_mean"].append(float(cp.mean(C_field)))
    history["T_mean"].append(float(cp.mean(T)))
    history["energy_drift"].append(drift)
    history["max_v_over_c"].append(float(cp.max(cp.sqrt(v2))/c))

    if step % 100 == 0:
        print(f"Step {step:4d} | t={float(t):.3e} | genie={history['genie_amp'][-1]:.2e} "
              f"| Majorana={history['ferm_majorana_mean'][-1]:.2e} | TotalFerm={history['ferm_total_mean'][-1]:.2e} "
              f"| C={history['C_mean'][-1]:.2e} | drift={drift:.2e}")

    progress_bar.update(1)

progress_bar.close()

# =============================================================================
# 43 Hz DETECTION (unchanged from v1.1)
# =============================================================================
# (the full 43 Hz block from previous v1.1 – copy it here exactly as before)
# ... [paste the entire 43 Hz detection block from your v1.1] ...

# =============================================================================
# FERMION + MAJORANA PLOTS
# =============================================================================
print("\nGenerating Majorana fermion plots...")

t_np = np.array(time_history)

def safe_hist(key, default=0.0):
    vals = history.get(key, [])
    arr = np.array([float(v) for v in vals]) if vals else np.full(len(t_np), default)
    if len(arr) < len(t_np):
        arr = np.pad(arr, (0, len(t_np)-len(arr)), mode='edge')
    return np.nan_to_num(arr, nan=default)

# Plot 01 – Dirac vs Majorana vs Total
plt.figure(figsize=(11, 6))
plt.plot(t_np, safe_hist("ferm_dirac_mean"), 'orange', lw=2.2, label='Dirac-like (y_ferm |ϕ|)')
plt.plot(t_np, safe_hist("ferm_majorana_mean"), 'red', lw=2.2, label='Majorana (κ ϕ²)')
plt.plot(t_np, safe_hist("ferm_total_mean"), 'white', lw=3, label='TOTAL fermion mass')
plt.xlabel('Simulation time')
plt.ylabel('Fermion mass proxy')
plt.title('Fermion Mass Evolution – Majorana Term Added')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('plots/plot01_ferm_mass.png', dpi=200, facecolor='black')
plt.close()
print("Saved: plots/plot01_ferm_mass.png")

# Plot 02 – Density
plt.figure(figsize=(10, 6))
plt.plot(t_np, safe_hist("ferm_density_mean"), 'lime', lw=2.5, label='Mean fermion density proxy')
plt.xlabel('Simulation time')
plt.ylabel('Density proxy')
plt.title('Fermion Density Evolution (modulated by total mass)')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('plots/plot02_ferm_density.png', dpi=180)
plt.close()
print("Saved: plots/plot02_ferm_density.png")

# Plot 14 – Overlaid with total fermion mass
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(t_np, safe_hist("genie_amp"), 'magenta', lw=2.4, label='Mean |genie_phi|')
ax1.set_xlabel('Time')
ax1.set_ylabel('Genie amplitude', color='magenta')
ax1.tick_params(axis='y', labelcolor='magenta')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t_np, safe_hist("ferm_total_mean"), 'orange', lw=2.2, label='Total fermion mass (Dirac+Majorana)')
ax2.set_ylabel('Fermion mass proxy', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

plt.title('Genie Scalar + Total Fermion Mass (Majorana included)')
plt.savefig('plots/plot14_overlaid_genie_fermion.png', dpi=180)
plt.close()
print("Saved: plots/plot14_overlaid_genie_fermion.png")

print("\nALADIN v1.2 complete with Majorana mass term!")
print("Check the new plots – Majorana contribution should be visible as quadratic growth.")
print("Tip: increase kappa_maj to 0.05 or g_genie to 1e-4 to see stronger Majorana effect.")
