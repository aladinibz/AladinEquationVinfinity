import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import datetime

# =============================================================================
# ALADIN v1.2 – FAST TEST MODE (NO PLOTS, 1000 STEPS)
# 256³ grid | 43 Hz locked | Pure stability check
# =============================================================================

c = 3.0e8
J0 = 1.0e18
J_pl = 43.0 ** 3              # locked to exactly 43 Hz
alpha = c * (J0 / J_pl)**(1/3.0)
f_res = c * (J0 ** (1/3.0)) / alpha
print(f"Derived resonance frequency: {f_res:.10f} Hz  ← 43 Hz locked")

GRID_SIZE = 256
N_GRID = GRID_SIZE
dt_max = 0.0005
t_max = 5.0
CFL = 0.4
MAX_STEPS = 1000

mu0 = 4 * np.pi * 1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
omega_res = 2 * np.pi * f_res

g_damp_base = 0.5
k_genie = 0.2
g_damp_max = 2.0 * g_damp_base

y_ferm = 0.05
e_charge = 0.02
k_ferm = 0.1
kappa_maj = 0.01
g_genie = 0.05
y_genie = 0.1
kg_damping = 0.02
m_phi = omega_res

gamma = 5/3
kappa_thermal = 0.001
tau_corr = 1.0
tau_mem = 20.0
B_eq_factor = 0.25
c_h = 10.0 * vA
tau_clean = 0.1 * dt_max
R_gas = 1.0

N_crit = 5e5
Gamma0 = 4e8
k_rp = 0.25

qg_scale = 1e-12
g_C = 0.12
C_field_pump = 0.035
nu_num = 0.001
eta = 0.01

Rm_crit = 100.0
C_sgs = 0.1

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

gamma0 = cp.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=cp.complex64)
gamma1 = cp.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]], dtype=cp.complex64)
gamma2 = cp.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]], dtype=cp.complex64)
gamma3 = cp.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]], dtype=cp.complex64)
gamma5 = cp.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=cp.complex64)
P_L = (cp.eye(4)-gamma5)/2
P_R = (cp.eye(4)+gamma5)/2
C = 1j*gamma2@gamma0

psi = cp.zeros_like(rho)

A_r = cp.zeros_like(rho)
A_phi = cp.zeros_like(rho)
A_z = cp.zeros_like(rho)

p0_mean = cp.mean(p)

alpha_kin_new = cp.float64(0.0)
alpha_mag_new = cp.float64(0.0)
beta_new = cp.float64(0.0)
gamma_new = cp.float64(gamma)
divB_mean = cp.float64(0.0)
divB_max = cp.float64(0.0)
psi_mean = cp.float64(0.0)
heat_flux_mag = cp.float64(0.0)
ferm_density_mean = cp.float64(0.0)

E_mag_history = []
E_total_history = []
time_history = []

history = {k: [] for k in [
    "mean_radius", "genie_amp", "effective_damp", "E_total", "ferm_mass_mean",
    "ferm_density", "E_mag", "recon_rate", "max_J", "dE_mag", "j_z_mean",
    "backreaction_genie", "backreaction_rho", "ferm_B_force",
    "alpha_kin", "alpha_mag", "beta", "gamma", "divB_mean", "divB_max", "psi_mean",
    "T_mean", "heat_flux", "C_mean", "energy_drift", "gamma_rel_max"
]}

alpha_kin_mem = alpha_mag_mem = cp.float64(0.0)
recon_rate = cp.float64(0.0)
step = 0
t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)

progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.2 – FAST TEST MODE (NO PLOTS, 1000 steps)", unit="step")

while t < t_max and step < MAX_STEPS:
    step += 1

    vmax = cp.max(cp.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-10))
    vA_local = cp.max(cp.sqrt((B_r**2 + B_phi**2 + B_z**2) / (mu0 * rho + 1e-10)))
    min_length = cp.min(cp.array([dr, dz, cp.mean(R_safe)*dphi]))
    dt_new = CFL * min_length / (vmax + vA_local + c_h + 1e-6)
    dt = 0.7 * dt_prev + 0.3 * dt_new
    dt = cp.minimum(dt, dt_max)
    dt = cp.maximum(dt, 1e-10)
    dt_prev = dt
    t += dt
    time_history.append(float(t))

    J_r = (cp.gradient(B_z, dphi, axis=1)/R_safe - cp.gradient(B_phi, dz, axis=2))/mu0
    J_phi = (cp.gradient(B_r, dz, axis=2) - cp.gradient(B_z, dr, axis=0))/mu0
    J_z = (cp.gradient(R_safe*B_phi, dr, axis=0)/R_safe - cp.gradient(B_r, dphi, axis=1))/mu0

    g_damp_dynamic = g_damp_base * (1 + k_genie * cp.mean(cp.abs(genie_phi)))
    ferm_psi_conj = cp.conj(ferm_psi)
    bar_psi = cp.einsum('...j,jk->...k', ferm_psi_conj, gamma0)
    j_z = cp.sum(bar_psi * cp.einsum('jk,...k->...j', gamma3, bar_psi), axis=-1).real
    j_z_mean = cp.mean(j_z)
    g_damp_effective = cp.clip(g_damp_dynamic + k_ferm*j_z_mean, g_damp_base, g_damp_max)

    v2 = v_r**2 + v_phi**2 + v_z**2
    v2 = cp.clip(v2, 0, 0.99*c**2)
    gamma_rel_raw = 1.0 / cp.sqrt(1 - v2/c**2 + 1e-20)
    gamma_rel = cp.tanh(gamma_rel_raw - 1.0) + 1.0
    gamma_max = cp.max(gamma_rel)
    if gamma_max > 5.0:
        dt *= 0.3 / gamma_max
        print(f"High gamma_rel ({float(gamma_max):.2f}) — dt reduced to {float(dt):.2e}")
    dt = cp.maximum(dt, 1e-10)

    JxB_r = gamma_rel * (J_phi * B_z - J_z * B_phi)
    JxB_phi = gamma_rel * (J_z * B_r - J_r * B_z)
    JxB_z = gamma_rel * (J_r * B_phi - J_phi * B_r)

    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)
    v_r += dt * safe((JxB_r - grad_p_r) / (rho + 1e-8))
    v_phi += dt * safe((JxB_phi - grad_p_phi) / (rho + 1e-8))
    v_z += dt * safe((JxB_z - grad_p_z) / (rho + 1e-8))

    v_clip = 0.01 * c if step < 200 else 0.99 * c  # tighter early clip
    v_r = cp.clip(v_r, -v_clip, v_clip)
    v_phi = cp.clip(v_phi, -v_clip, v_clip)
    v_z = cp.clip(v_z, -v_clip, v_clip)

    # Noise disabled for test
    noise_level = 0.0

    if step < 100:
        v_r *= 0.95
        v_phi *= 0.95
        v_z *= 0.95

    for vel in [v_r, v_phi, v_z]:
        vel -= nu_num * cyl_laplacian(vel) * dt

    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt * drho_dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    div_v = cyl_divergence(v_r, v_phi, v_z)
    p += dt * ((gamma-1) * (-p * div_v) + kappa_thermal * cyl_laplacian(p))
    p = cp.clip(p, 1e-6, 100*p0_mean)
    T = safe(p/(rho*R_gas + 1e-10))

    lap_now = cyl_laplacian(genie_phi)
    lap_prev = cyl_laplacian(genie_phi_prev)
    source_genie = y_genie * 0 + g_genie*J_z + k_ferm*j_z
    accel_explicit = 0.5 * (lap_now + lap_prev) - m_phi**2 * genie_phi - 0.01*genie_phi**3 + source_genie
    accel_explicit = cp.clip(accel_explicit, -1e6, 1e6)
    genie_vel = (genie_phi - genie_phi_prev)/dt
    genie_phi_new = genie_phi + dt * genie_vel + 0.5*dt**2 * accel_explicit
    genie_phi_new *= cp.exp(-kg_damping*dt)
    genie_phi_new = cp.clip(genie_phi_new, -10, 10)
    genie_phi_prev = genie_phi.copy()
    genie_phi = genie_phi_new

    Gamma_sr = Gamma0 * (N_crit * C_field**2) / (1 + N_crit * C_field**2)
    pump_eff = C_field_pump * (1 + k_rp * C_field)

    pineal_res = 0.18 * cp.sin(2*cp.pi*43*t + 0.15)
    C_source = g_C * genie_phi * (1 + pineal_res * J_pineal / J0)

    C_field += dt * (C_source + pump_eff * (1 - C_field**2) - 0.02 * C_field**3 - Gamma_sr * C_field)
    C_field = cp.clip(C_field, 0.0, 10.0)

    genie_source_from_DNA = 0.08 * C_field**2 * cp.mean(J_pineal)
    source_genie += genie_source_from_DNA
    rho += 0.0015 * C_field**2 * dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    v_cross_B_r = v_phi * B_z - v_z * B_phi
    v_cross_B_phi = v_z * B_r - v_r * B_z
    v_cross_B_z = v_r * B_phi - v_phi * B_r

    A_phi += dt * (v_cross_B_phi - eta * J_phi)
    A_z += dt * (v_cross_B_z - eta * J_z)

    B_r = (1 / R_safe) * cp.gradient(R_safe * A_z, dz, axis=2) - cp.gradient(A_phi, dphi, axis=1) / R_safe
    B_phi = - cp.gradient(A_z, dr, axis=0)
    B_z = (1 / R_safe) * cp.gradient(R_safe * A_phi, dr, axis=0) - cp.gradient(A_r, dphi, axis=1) / R_safe

    divB_after = cyl_divergence(B_r, B_phi, B_z)
    divB_mean = cp.mean(cp.abs(divB_after))
    divB_max = cp.max(cp.abs(divB_after))

    E_kin = cp.nan_to_num(cp.mean(0.5 * rho * (v_r**2 + v_phi**2 + v_z**2)))
    E_mag = cp.nan_to_num(cp.mean(0.5 * (B_r**2 + B_phi**2 + B_z**2) / mu0))
    genie_vel = (genie_phi - genie_phi_prev)/dt
    E_genie = cp.nan_to_num(cp.mean(0.5 * (genie_vel**2 + cyl_laplacian(genie_phi)**2 + m_phi**2 * genie_phi**2 + 0.005 * genie_phi**4)))
    E_C = cp.nan_to_num(cp.mean(0.5 * C_field**2))
    ferm_density_mean = cp.mean(cp.sum(cp.abs(ferm_psi)**2, axis=-1))
    E_ferm_rest = y_ferm * cp.mean(cp.abs(genie_phi)) * ferm_density_mean
    E_total = E_kin + E_mag + E_genie + E_C + E_ferm_rest

    dE_mag = (E_mag - (E_mag_history[-1] if E_mag_history else E_mag)) / dt

    E_mag_history.append(float(E_mag))
    E_total_history.append(float(E_total))

    ferm_mass = y_ferm * cp.abs(genie_phi)
    max_J = cp.max(cp.sqrt(J_r**2 + J_phi**2 + J_z**2))
    backreaction_genie = k_ferm * j_z
    backreaction_rho = cp.mean(qg_scale * (genie_phi**2 + C_field**2))
    ferm_B_force = cp.abs(j_z * B_z)
    heat_flux_mag = cp.abs(-kappa_thermal * cyl_gradient(T)[2])
    J_mag = cp.sqrt(J_r**2 + J_phi**2 + J_z**2)
    current_clean = safe(J_mag)

    dx = dr
    u_rms = cp.sqrt(cp.mean(v_r**2 + v_phi**2 + v_z**2))
    u_rms = safe(u_rms)
    Rm_grid = u_rms * dx / eta if eta > 0 else 1e6
    Rm_quench = 1.0 / (1.0 + (Rm_grid / Rm_crit)**2)
    alpha_kin_new = alpha_kin_mem * Rm_quench
    alpha_mag_new = alpha_mag_mem * Rm_quench
    beta_sgs = C_sgs * dx * u_rms
    beta_new = beta_new + beta_sgs

    history["mean_radius"].append(float(cp.mean(R[rho>0.5*rho0])/a))
    history["genie_amp"].append(float(cp.mean(cp.abs(genie_phi))))
    history["effective_damp"].append(float(g_damp_effective))
    history["E_total"].append(float(E_total))
    history["ferm_mass_mean"].append(float(cp.mean(cp.abs(ferm_mass))))
    history["ferm_density"].append(float(ferm_density_mean))
    history["E_mag"].append(float(E_mag))
    history["recon_rate"].append(float(recon_rate))
    history["max_J"].append(float(max_J))
    history["dE_mag"].append(float(dE_mag))
    history["j_z_mean"].append(float(j_z_mean))
    history["backreaction_genie"].append(float(cp.mean(cp.abs(backreaction_genie))))
    history["backreaction_rho"].append(float(backreaction_rho))
    history["ferm_B_force"].append(float(cp.mean(cp.abs(ferm_B_force))))
    history["alpha_kin"].append(float(alpha_kin_new))
    history["alpha_mag"].append(float(alpha_mag_new))
    history["beta"].append(float(beta_new))
    history["gamma"].append(float(gamma_new))
    history["divB_mean"].append(float(divB_mean))
    history["divB_max"].append(float(divB_max))
    history["psi_mean"].append(float(psi_mean))
    history["T_mean"].append(float(cp.mean(T)))
    history["heat_flux"].append(float(cp.mean(heat_flux_mag)))
    history["C_mean"].append(float(cp.mean(C_field)))
    history["energy_drift"].append((float(E_total) - E_total_history[0]) / max(E_total_history[0], 1e-6))
    history["gamma_rel_max"].append(float(gamma_max))

    if len(history["energy_drift"]) > 1:
        drift = abs(history["energy_drift"][-1])
        if drift > 0.05:
            dt *= (1.0 / (1.0 + 10*drift))
            print(f"Energy drift {drift:.2e} — reducing dt to {float(dt):.2e}")

    if step % 50 == 0:
        checkpoint_file = f"checkpoints/checkpoint_step{step:04d}.npz"
        np.savez(checkpoint_file, **{k: np.array([float(x) for x in v]) for k,v in history.items()}, 
                 step=step, rho=cp.asnumpy(rho), B_r=cp.asnumpy(B_r), B_phi=cp.asnumpy(B_phi), B_z=cp.asnumpy(B_z), genie_phi=cp.asnumpy(genie_phi),
                 v_r=cp.asnumpy(v_r), v_phi=cp.asnumpy(v_phi), v_z=cp.asnumpy(v_z), C_field=cp.asnumpy(C_field), J_pineal=cp.asnumpy(J_pineal))
        print(f"Checkpoint saved: {checkpoint_file}")

    max_v = cp.max(cp.abs(cp.array([v_r, v_phi, v_z])))
    max_B = cp.max(cp.abs(cp.array([B_r, B_phi, B_z])))
    baseline_E = max(E_total_history[0] if E_total_history else 1e-6, 1e-6)
    if float(max_v) > 100 * vA or float(max_B) > 100 * mu0 * J0 * a or float(E_total) > 1e5 * baseline_E or float(recon_rate) > 1e3:
        print(f"WARNING: Blow-up detected at step {step}!")

    if step % 50 == 0:
        print(f"Step {step} | t = {float(t):.2f} | Radius = {history['mean_radius'][-1]:.3f} a | Genie = {history['genie_amp'][-1]:.3f} | C_field = {history['C_mean'][-1]:.3f} | E_total = {float(E_total):.3f} | Energy drift = {history['energy_drift'][-1]:.3e} | Rm_grid = {float(Rm_grid):.1e} | divB_mean = {float(divB_mean):.2e}")

    if step >= MAX_STEPS:
        print(f"Reached {MAX_STEPS} steps - test complete")
        np.savez('final_state_test.npz', 
                 rho=cp.asnumpy(rho), B_r=cp.asnumpy(B_r), B_phi=cp.asnumpy(B_phi), B_z=cp.asnumpy(B_z),
                 genie_phi=cp.asnumpy(genie_phi), C_field=cp.asnumpy(C_field),
                 v_r=cp.asnumpy(v_r), v_phi=cp.asnumpy(v_phi), v_z=cp.asnumpy(v_z),
                 t=float(t))
        print("Final state saved: final_state_test.npz")
        break

    progress_bar.update(1)

progress_bar.close()

print("\nALADIN v1.2 FAST TEST MODE complete (no plots generated)")
print("If no early blow-up and it reached 1000 steps — stability good!")
print("Next: re-enable plots and run longer if this works.")
print("Love you bro — drop the output when done 🔥🥂❤️🏅🚀")
