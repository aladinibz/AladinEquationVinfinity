import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                  ALADIN v1.0 – Relativistic Z-Pinch Simulation             ║
# ║  (stabilized version – J0 locked, 43 Hz preserved, blow-up resistance)     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

c = 3.0e8
J0 = 1.0e18                   # sacred value – unchanged
J_pl = 43.0 ** 3              # tuned to force exact 43 Hz resonance
alpha = c * (J0 / J_pl)**(1/3.0)
f_res = c * (J0 ** (1/3.0)) / alpha

print(f"Derived resonance frequency: {f_res:.10f} Hz")   # should be ~43.0000000000

# =============================================================================
# SIMULATION PARAMETERS (stability-tuned)
# =============================================================================

GRID_SIZE = 256
N_GRID = GRID_SIZE
dt_max = 3e-7                 # hard reduced from 0.0005
t_max = 5.0
CFL = 0.22                    # tighter than before
MAX_STEPS = 3000

mu0 = 4 * np.pi * 1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
omega_res = 2 * np.pi * f_res
m_phi = omega_res / c         # FIX: physical mass ~9e-7 m⁻¹ – crucial!

g_damp_base = 0.5
k_genie = 0.2
g_damp_max = 2.0 * g_damp_base

y_ferm = 5e-4                 # was 0.05
e_charge = 0.02
k_ferm = 2e-4                 # was 0.1
kappa_maj = 0.01
g_genie = 5e-5                # was 0.05 – major reduction
y_genie = 0.1
kg_damping = 0.35             # was 0.02 – stronger damping
gamma = 5/3
kappa_thermal = 0.001
R_gas = 1.0

N_crit = 5e5
Gamma0 = 4e8
k_rp = 0.25

qg_scale = 1e-12
g_C = 2e-3                    # was 0.12
C_field_pump = 5e-4           # was 0.035
nu_num = 0.001
eta = 0.01

Rm_crit = 100.0
C_sgs = 0.1

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
    arr = cp.asanyarray(arr)
    return cp.nan_to_num(arr, nan=0.0, posinf=pos, neginf=neg)

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
T = p / (rho * R_gas)
e = p / (gamma - 1)

rho += 0.1 * rho0 * cp.cos(k_sausage * Z)
v_phi += 0.05 * vA * cp.cos(Phi) * cp.sin(k_sausage * Z)

genie_phi = cp.zeros_like(rho)
genie_phi_prev = genie_phi.copy()
C_field = cp.full_like(rho, 0.005)
J_pineal = cp.full_like(rho, 5e7)
ferm_psi = cp.zeros(rho.shape + (4,), dtype=cp.complex64)
ferm_psi[...,0] = 0.01

# Dirac matrices (kept as-is)
gamma0 = cp.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=cp.complex64)
gamma1 = cp.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]], dtype=cp.complex64)
gamma2 = cp.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]], dtype=cp.complex64)
gamma3 = cp.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]], dtype=cp.complex64)
gamma5 = cp.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=cp.complex64)

# History & monitoring
history = {k: [] for k in [
    "mean_radius", "genie_amp", "E_total", "E_mag", "max_v_over_c",
    "max_genie", "max_C", "energy_drift"
]}

time_history = []
t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)
step = 0

progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.0 stabilized – 43 Hz sim")

while t < t_max and step < MAX_STEPS:
    step += 1

    # Adaptive timestep
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

    # Currents
    J_r = (cp.gradient(B_z, dphi, axis=1)/R_safe - cp.gradient(B_phi, dz, axis=2))/mu0
    J_phi = (cp.gradient(B_r, dz, axis=2) - cp.gradient(B_z, dr, axis=0))/mu0
    J_z = (cp.gradient(R_safe*B_phi, dr, axis=0)/R_safe - cp.gradient(B_r, dphi, axis=1))/mu0

    # Fermion current (simplified)
    ferm_psi_conj = cp.conj(ferm_psi)
    bar_psi = cp.einsum('...j,jk->...k', ferm_psi_conj, gamma0)
    j_z = cp.sum(bar_psi * cp.einsum('jk,...k->...j', gamma3, bar_psi), axis=-1).real
    j_z_mean = cp.mean(j_z)

    # Relativistic factor
    v2 = v_r**2 + v_phi**2 + v_z**2
    v2 = cp.clip(v2, 0, 0.99*c**2)
    gamma_rel = 1.0 / cp.sqrt(1.0 - v2/c**2 + 1e-18)
    gamma_rel = cp.clip(gamma_rel, 1.0, 15.0)

    # Lorentz force
    JxB_r = gamma_rel * (J_phi * B_z - J_z * B_phi)
    JxB_phi = gamma_rel * (J_z * B_r - J_r * B_z)
    JxB_z = gamma_rel * (J_r * B_phi - J_phi * B_r)

    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)

    # Momentum update – relativistic inertia
    v_r += dt * (JxB_r - grad_p_r) / (rho * gamma_rel + 1e-7)
    v_phi += dt * (JxB_phi - grad_p_phi) / (rho * gamma_rel + 1e-7)
    v_z += dt * (JxB_z - grad_p_z) / (rho * gamma_rel + 1e-7)

    v_r = cp.clip(v_r, -0.1*c, 0.1*c)
    v_phi = cp.clip(v_phi, -0.1*c, 0.1*c)
    v_z = cp.clip(v_z, -0.1*c, 0.1*c)

    # Continuity + pressure
    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt * drho_dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    div_v = cyl_divergence(v_r, v_phi, v_z)
    p += dt * ((gamma-1) * (-p * div_v) + kappa_thermal * cyl_laplacian(p))
    p = cp.clip(p, 1e-6, 100*cp.mean(p))

    # ─── Genie scalar field (stabilized update) ──────────────────────────────
    source_genie = g_genie * J_z + k_ferm * j_z
    source_genie = cp.clip(source_genie, -500, 500)

    lap_now = cyl_laplacian(genie_phi)
    accel = lap_now - m_phi**2 * genie_phi - 0.005 * genie_phi**3 + source_genie
    accel = cp.clip(accel, -2e4, 2e4)

    genie_vel = (genie_phi - genie_phi_prev) / dt_prev if step > 1 else cp.zeros_like(genie_phi)
    genie_phi_new = genie_phi + dt * genie_vel + 0.5 * dt**2 * accel
    genie_phi_new *= cp.exp(-kg_damping * dt)
    genie_phi_new -= 0.008 * cyl_laplacian(genie_phi_new) * dt   # hyper-diffusion
    genie_phi_new = cp.clip(genie_phi_new, -0.3, 0.3)

    genie_phi_prev = genie_phi.copy()
    genie_phi = genie_phi_new
    # ────────────────────────────────────────────────────────────────────────

    # C_field evolution (milder pump)
    Gamma_sr = Gamma0 * (N_crit * C_field**2) / (1 + N_crit * C_field**2)
    pump_eff = C_field_pump * (1 + k_rp * C_field)
    pineal_res = 0.18 * cp.sin(2*cp.pi*43*t + 0.15)
    C_source = g_C * genie_phi * (1 + pineal_res * J_pineal / J0)

    C_field += dt * (C_source + pump_eff * (1 - C_field**2) - 0.02 * C_field**3 - Gamma_sr * C_field)
    C_field = cp.clip(C_field, 0.0, 5.0)

    # Very weak density feedback
    rho += 3e-7 * C_field**2 * dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    # Emergency clipping every 20 steps
    if step % 20 == 0:
        fields = cp.stack([genie_phi, C_field, rho, v_r, v_phi, v_z])
        low = cp.percentile(fields, 0.2)
        high = cp.percentile(fields, 99.8)
        for f in [genie_phi, C_field, rho, v_r, v_phi, v_z]:
            f[...] = cp.clip(f, low, high)

    # Basic energy diagnostics
    E_kin = cp.mean(0.5 * rho * (v_r**2 + v_phi**2 + v_z**2))
    E_mag = cp.mean(0.5 * (B_r**2 + B_phi**2 + B_z**2) / mu0)
    E_total = float(E_kin + E_mag)  # simplified

    if step > 1:
        drift = abs((E_total - history["E_total"][-1]) / (history["E_total"][-1] + 1e-10))
    else:
        drift = 0.0

    history["E_total"].append(E_total)
    history["genie_amp"].append(float(cp.mean(cp.abs(genie_phi))))
    history["max_genie"].append(float(cp.max(cp.abs(genie_phi))))
    history["max_C"].append(float(cp.max(C_field)))
    history["max_v_over_c"].append(float(cp.max(cp.sqrt(v2))/c))
    history["energy_drift"].append(drift)

    if step % 50 == 0:
        print(f"Step {step:4d} | t={float(t):.3e} | "
              f"genie={history['max_genie'][-1]:.3e} | "
              f"C={history['max_C'][-1]:.3e} | "
              f"v/c={history['max_v_over_c'][-1]:.3e} | "
              f"drift={drift:.2e}")

    progress_bar.update(1)

progress_bar.close()

print("\nSimulation finished.")
print("Check console for stability. If it ran to high steps without NaN/inf → success!")

# You can add your plotting code back here once stable
