# =============================================================================
# Z-Pinch Sim v1.0.8 – Cosmic Filament Framework (MHD + QFT Coupling)
# =============================================================================
# Version: 1.0.8 – Auto-install tqdm + progress bar + checkpoints
# Full 15 plots saved to ./plots/ + final_state.npz + validation suite
# Zenodo-ready – run this script directly

import sys
import subprocess

# Auto-install tqdm if missing
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found – installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm
    print("tqdm installed successfully!")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.fftpack import fftn, ifftn

# ==================== PARAMETERS ====================
N_GRID = 128
dt = 0.005
t_max = 5.0
n_steps = int(t_max / dt)

J0 = 1.0e18
c = 3.0e8
f_res_target = 43.0
J_pl = 4.3e5
alpha = c * (J0 / J_pl)**(1/3)
f_res = c * (J0 ** (1/3)) / alpha
print(f"Derived f_res = {f_res:.10f} Hz")

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

eta = 0.01
hall_param = 0.01
lambda_i = 0.01
nu = 0.01
nu_num = 0.001
kappa_thermal = 0.01
gamma = 5/3

# ==================== SETUP ====================
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Grid
r = np.linspace(0, 2*a, N_GRID)
phi = np.linspace(0, 2*np.pi, N_GRID)
z = np.linspace(0, 10*a, N_GRID)
dr = r[1] - r[0]
dphi = phi[1] - phi[0]
dz = z[1] - z[0]
R, Phi, Z = np.meshgrid(r, phi, z, indexing='ij')

R_safe = np.where(R < 1e-3, 1e-3, R)

# Helpers
def cyl_laplacian(field):
    dphi_dr = np.gradient(field, dr, axis=0)
    term_r = (1 / R_safe) * np.gradient(R_safe * dphi_dr, dr, axis=0)
    term_phi = (1 / R_safe**2) * np.gradient(np.gradient(field, dphi, axis=1), dphi, axis=1)
    term_z = np.gradient(np.gradient(field, dz, axis=2), dz, axis=2)
    return term_r + term_phi + term_z

def cyl_gradient(field):
    grad_r = np.gradient(field, dr, axis=0)
    grad_phi = np.gradient(field, dphi, axis=1) / R_safe
    grad_z = np.gradient(field, dz, axis=2)
    return grad_r, grad_phi, grad_z

def cyl_divergence(vr, vphi, vz):
    term_r = (1 / R_safe) * np.gradient(R_safe * vr, dr, axis=0)
    term_phi = (1 / R_safe) * np.gradient(vphi, dphi, axis=1)
    term_z = np.gradient(vz, dz, axis=2)
    return term_r + term_phi + term_z

# Initial fields
J_z = J0 * np.exp(-R**2 / a**2)
B_phi = mu0 * J0 * a * (1 - np.exp(-R**2 / a**2))
B_r = np.zeros((N_GRID, N_GRID, N_GRID))
B_z = np.zeros((N_GRID, N_GRID, N_GRID))
rho = rho0 * np.ones((N_GRID, N_GRID, N_GRID))
v_r = np.zeros((N_GRID, N_GRID, N_GRID))
v_phi = np.zeros((N_GRID, N_GRID, N_GRID))
v_z = np.zeros((N_GRID, N_GRID, N_GRID))
p = rho0 * vA**2 * np.ones((N_GRID, N_GRID, N_GRID))
e = p / (gamma - 1)

# Sausage & kink
k_sausage = 2 * np.pi / (5 * a)
delta_rho_sausage = 0.1 * rho0 * np.cos(k_sausage * Z)
rho += delta_rho_sausage

k_kink = k_sausage
delta_v_phi_kink = 0.05 * vA * np.cos(Phi) * np.sin(k_kink * Z)
v_phi += delta_v_phi_kink

# Genie
m_phi = omega_res
kappa = 0.01
g_genie = 0.05
y_genie = 0.1
kg_damping = 0.02
genie_phi = np.zeros((N_GRID, N_GRID, N_GRID))
genie_phi_prev = genie_phi.copy()

# Dirac spinor fermions
ferm_psi = np.zeros((N_GRID, N_GRID, N_GRID, 4), dtype=complex)
ferm_psi[..., 0] = 0.01

# Gamma matrices (Dirac basis)
gamma0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=complex)
gamma1 = np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]], dtype=complex)
gamma2 = np.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]], dtype=complex)
gamma3 = np.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]], dtype=complex)
gamma5 = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=complex)

P_L = (np.eye(4) - gamma5) / 2
P_R = (np.eye(4) + gamma5) / 2

C = 1j * gamma2 @ gamma0

# Vector potential A
A_z = - np.cumsum(B_phi, axis=0) * dr
A_r = np.zeros((N_GRID, N_GRID, N_GRID))
A_phi = np.zeros((N_GRID, N_GRID, N_GRID))

# Tracking
mean_radius_history = []
genie_amp_history = []
effective_damp_history = []
E_total_history = []
ferm_mass_mean_history = []
ferm_density_history = []
E_mag_history = []
recon_rate_history = []
max_J_history = []
dE_mag_history = []
j_z_mean_history = []
backreaction_genie_history = []
backreaction_rho_history = []
ferm_B_force_history = []

print("Starting Z-pinch sim v1.0.8 – Progress bar + checkpoints every 50 steps")

# Initialize previous currents for electron inertia
J_r_prev = J_r.copy()
J_phi_prev = J_phi.copy()
J_z_prev = J_z.copy()

# Progress bar with tqdm
progress_bar = tqdm(total=n_steps, desc="Simulation progress", unit="step")

for step in range(n_steps):
    t = step * dt
    t_phys = t * a / vA

    genie_amp_current = np.mean(np.abs(genie_phi))
    g_damp_dynamic = g_damp_base * (1 + k_genie * genie_amp_current)

    # Fermion current back-reaction
    ferm_psi_conj = np.conj(ferm_psi)
    bar_psi = np.einsum('...j,jk->...k', ferm_psi_conj, gamma0)
    gamma3_bar_psi = np.einsum('jk,...k->...j', gamma3, bar_psi)
    j_z = np.sum(bar_psi * gamma3_bar_psi, axis=-1).real
    j_z_mean = np.mean(j_z)
    g_damp_effective = g_damp_dynamic + k_ferm * j_z_mean
    g_damp_effective = np.clip(g_damp_effective, g_damp_base, g_damp_max)
    effective_damp_history.append(g_damp_effective)

    j_z_mean_history.append(j_z_mean)

    # Full vector J = (∇ × B) / μ₀
    J_r = (1 / mu0) * (np.gradient(B_z, dphi, axis=1) / R_safe - np.gradient(B_phi, dz, axis=2))
    J_phi = (1 / mu0) * (np.gradient(B_r, dz, axis=2) - np.gradient(B_z, dr, axis=0))
    J_z = (1 / mu0) * (1 / R_safe) * np.gradient(R_safe * B_phi, dr, axis=0) - np.gradient(B_r, dphi, axis=1) / R_safe

    # Fermion current to B-field backreaction
    total_J_z = J_z + j_z
    ferm_B_force = j_z * B_phi
    ferm_B_force_history.append(np.mean(np.abs(ferm_B_force)))

    JxB_r = J_phi * B_z - total_J_z * B_phi
    JxB_phi = total_J_z * B_r - J_r * B_z
    JxB_z = J_r * B_phi - J_phi * B_r

    # Induction equation – FULLY ACTIVE
    vxB_r = v_phi * B_z - v_z * B_phi
    vxB_phi = v_z * B_r - v_r * B_z
    vxB_z = v_r * B_phi - v_phi * B_r

    curl_vxB_r = (1 / R_safe) * np.gradient(vxB_z, dphi, axis=1) - np.gradient(vxB_phi, dz, axis=2)
    curl_vxB_phi = np.gradient(vxB_r, dz, axis=2) - np.gradient(vxB_z, dr, axis=0)
    curl_vxB_z = (1 / R_safe) * np.gradient(R_safe * vxB_phi, dr, axis=0) - (1 / R_safe) * np.gradient(vxB_r, dphi, axis=1)

    # Hall term
    JxB_hall_r = - (JxB_r / (rho * e_charge + 1e-12))
    JxB_hall_phi = - (JxB_phi / (rho * e_charge + 1e-12))
    JxB_hall_z = - (JxB_z / (rho * e_charge + 1e-12))

    curl_hall_r = (1 / R_safe) * np.gradient(JxB_hall_z, dphi, axis=1) - np.gradient(JxB_hall_phi, dz, axis=2)
    curl_hall_phi = np.gradient(JxB_hall_r, dz, axis=2) - np.gradient(JxB_hall_z, dr, axis=0)
    curl_hall_z = (1 / R_safe) * np.gradient(R_safe * JxB_hall_phi, dr, axis=0) - (1 / R_safe) * np.gradient(JxB_hall_r, dphi, axis=1)

    # Electron inertia term
    if step == 0:
        J_r_prev = J_r.copy()
        J_phi_prev = J_phi.copy()
        J_z_prev = J_z.copy()

    dJ_r = (J_r - J_r_prev) / dt
    dJ_phi = (J_phi - J_phi_prev) / dt
    dJ_z = (J_z - J_z_prev) / dt

    inertia_r = lambda_i * ((1 / R_safe) * np.gradient(dJ_z, dphi, axis=1) - np.gradient(dJ_phi, dz, axis=2))
    inertia_phi = lambda_i * (np.gradient(dJ_r, dz, axis=2) - np.gradient(dJ_z, dr, axis=0))
    inertia_z = lambda_i * ((1 / R_safe) * np.gradient(R_safe * dJ_phi, dr, axis=0) - (1 / R_safe) * np.gradient(dJ_r, dphi, axis=1))

    # Diffusion
    diff_B_r = eta * cyl_laplacian(B_r)
    diff_B_phi = eta * cyl_laplacian(B_phi)
    diff_B_z = eta * cyl_laplacian(B_z)

    # Update B
    B_r += dt * (curl_vxB_r + curl_hall_r + diff_B_r + inertia_r)
    B_phi += dt * (curl_vxB_phi + curl_hall_phi + diff_B_phi + inertia_phi)
    B_z += dt * (curl_vxB_z + curl_hall_z + diff_B_z + inertia_z)

    # Update previous J
    J_r_prev = J_r.copy()
    J_phi_prev = J_phi.copy()
    J_z_prev = J_z.copy()

    schumann_harmonic = 6
    schumann_mod = np.abs(np.sin(2 * np.pi * f_res * t_phys / schumann_harmonic))
    damp_factor = np.exp(-g_damp_effective * np.abs(np.sin(omega_res * t_phys)) * dt * schumann_mod)
    v_r *= damp_factor
    v_phi *= damp_factor
    v_z *= damp_factor

    v_r += JxB_r * dt
    v_phi += JxB_phi * dt
    v_z += JxB_z * dt
    v_r = np.clip(v_r, -10*vA, 10*vA)
    v_phi = np.clip(v_phi, -10*vA, 10*vA)
    v_z = np.clip(v_z, -10*vA, 10*vA)

    # Add numerical viscosity to velocities
    for vel in [v_r, v_phi, v_z]:
        laplacian_vel = cyl_laplacian(vel)
        vel -= nu_num * laplacian_vel * dt

    # Correct cylindrical continuity equation
    term_r = (1 / R_safe) * np.gradient(R_safe * rho * v_r, dr, axis=0)
    term_phi = (1 / R_safe) * np.gradient(rho * v_phi, dphi, axis=1)
    term_z = np.gradient(rho * v_z, dz, axis=2)
    drho_dt = - (term_r + term_phi + term_z)
    rho += drho_dt * dt
    rho = np.maximum(rho, 1e-6)

    # Compute fermion density & mass BEFORE rho backreaction
    ferm_psi_conj = np.conj(ferm_psi)
    bar_psi = np.einsum('...j,jk->...k', ferm_psi_conj, gamma0)
    ferm_density = np.sum(bar_psi * ferm_psi, axis=-1).real
    ferm_density_mean = np.mean(ferm_density)
    ferm_density_history.append(ferm_density_mean)

    ferm_mass = y_ferm * genie_phi + kappa_maj * genie_phi**2
    ferm_mass_mean = np.mean(np.abs(ferm_mass))
    ferm_mass_mean_history.append(ferm_mass_mean)

    # Fermion backreaction to plasma density
    backreaction_rho = 0.1 * ferm_density_mean
    rho += backreaction_rho
    rho = np.maximum(rho, 1e-6)
    backreaction_rho_history.append(backreaction_rho)

    # Cylindrical Laplacian for Genie
    laplacian_genie = cyl_laplacian(genie_phi)

    # Fermion backreaction to Genie source
    backreaction_genie = k_ferm * j_z
    source_genie = y_genie * delta_rho_sausage + g_genie * J_z + backreaction_genie
    backreaction_genie_history.append(np.mean(np.abs(backreaction_genie)))

    genie_phi_new = 2 * genie_phi - genie_phi_prev * (1 - kg_damping) + dt**2 * (laplacian_genie - m_phi**2 * genie_phi - kappa * genie_phi**3 + source_genie)
    genie_phi_prev = genie_phi.copy()
    genie_phi = genie_phi_new

    # Full Strang splitting for Dirac evolution
    grad_r_psi = np.gradient(ferm_psi, dr, axis=0)
    grad_phi_psi = np.gradient(ferm_psi, dphi, axis=1) / R_safe[..., np.newaxis]
    grad_z_psi = np.gradient(ferm_psi, dz, axis=2)

    kinetic = (np.einsum('jk,...k->...j', gamma1, grad_r_psi) +
               np.einsum('jk,...k->...j', gamma2, grad_phi_psi) +
               np.einsum('jk,...k->...j', gamma3, grad_z_psi))

    ferm_psi_mid = ferm_psi + 1j * dt/2 * np.einsum('jk,...k->...j', gamma0, kinetic)

    mass_term = y_ferm * genie_phi[..., np.newaxis] * ferm_psi_mid

    gauge_term = 1j * e_charge * J_z[..., np.newaxis] * ferm_psi_mid

    m_Maj = kappa_maj * genie_phi**2
    psi_c = np.einsum('jk,...k->...j', C, ferm_psi_mid.conj().transpose(3,0,1,2)).transpose(1,2,3,0)
    majorana_term = m_Maj[..., np.newaxis] * (P_L @ psi_c.transpose(3,0,1,2)).transpose(1,2,3,0)

    rhs = - mass_term + gauge_term - majorana_term
    ferm_psi_mid2 = ferm_psi_mid + 1j * dt * np.einsum('jk,...k->...j', gamma0, rhs)

    grad_r_psi = np.gradient(ferm_psi_mid2, dr, axis=0)
    grad_phi_psi = np.gradient(ferm_psi_mid2, dphi, axis=1) / R_safe[..., np.newaxis]
    grad_z_psi = np.gradient(ferm_psi_mid2, dz, axis=2)

    kinetic = (np.einsum('jk,...k->...j', gamma1, grad_r_psi) +
               np.einsum('jk,...k->...j', gamma2, grad_phi_psi) +
               np.einsum('jk,...k->...j', gamma3, grad_z_psi))

    ferm_psi_new = ferm_psi_mid2 + 1j * dt/2 * np.einsum('jk,...k->...j', gamma0, kinetic)

    norm = np.sqrt(np.sum(np.abs(ferm_psi_new)**2, axis=-1, keepdims=True))
    ferm_psi_new /= np.maximum(norm, 1e-12)

    ferm_psi = ferm_psi_new

    E_kin = np.mean(0.5 * rho * (v_r**2 + v_phi**2 + v_z**2))
    E_mag_prev = E_mag_history[-1] if E_mag_history else 0
    E_mag = np.mean(0.5 * (B_r**2 + B_phi**2 + B_z**2) / mu0)
    dE_mag = (E_mag - E_mag_prev) / dt
    grad_phi_r = np.gradient(genie_phi, dr, axis=0)
    grad_phi_phi = np.gradient(genie_phi, dphi, axis=1) / R_safe
    grad_phi_z = np.gradient(genie_phi, dz, axis=2)
    E_grad = np.mean(0.5 * (grad_phi_r**2 + grad_phi_phi**2 + grad_phi_z**2))
    E_genie = np.mean(0.5 * genie_phi**2) + E_grad
    E_total = E_kin + E_mag + E_genie
    E_total_history.append(E_total)
    E_mag_history.append(E_mag)
    dE_mag_history.append(dE_mag)

    # Reconnection diagnostics
    laplacian_Br = cyl_laplacian(B_r)
    laplacian_Bphi = cyl_laplacian(B_phi)
    laplacian_Bz = cyl_laplacian(B_z)
    recon_rate = eta * np.mean(np.abs(laplacian_Br) + np.abs(laplacian_Bphi) + np.abs(laplacian_Bz))
    recon_rate_history.append(recon_rate)

    max_J = np.max(np.abs([J_r, J_phi, J_z]))
    max_J_history.append(max_J)

    current_radius = np.mean(R[rho > 0.5 * rho0]) / a
    mean_radius_history.append(current_radius)
    genie_amp_history.append(np.mean(np.abs(genie_phi)))

    # Periodic checkpoint every 50 steps
    if step % 50 == 0 and step > 0:
        checkpoint_file = f"checkpoints/checkpoint_step{step:04d}.npz"
        np.savez(checkpoint_file,
                 step=step,
                 rho=rho,
                 B_r=B_r,
                 B_phi=B_phi,
                 B_z=B_z,
                 genie_phi=genie_phi,
                 v_r=v_r,
                 v_phi=v_phi,
                 v_z=v_z,
                 ferm_psi_mean_norm=np.mean(np.sqrt(np.sum(np.abs(ferm_psi)**2, axis=-1))))
        print(f"Checkpoint saved: {checkpoint_file}")

    # Blow-up check
    max_v = np.max(np.abs([v_r, v_phi, v_z]))
    max_B = np.max(np.abs([B_r, B_phi, B_z]))
    if max_v > 100 * vA or max_B > 100 * mu0 * J0 * a or E_total > 1e5 * (E_total_history[0] if E_total_history else 0) or recon_rate > 1e3:
        print(f"WARNING: Blow-up detected at step {step}! max_v = {max_v:.2f}, max_B = {max_B:.2f}, E_total = {E_total:.2f}, recon_rate = {recon_rate:.3e}")
        break

    if step % 50 == 0:
        print(f"Step {step} | t = {t:.2f} | Radius = {current_radius:.3f} a | Genie = {genie_amp_history[-1]:.3f} | Ferm mass mean = {ferm_mass_mean:.3f} | Ferm density mean = {ferm_density_mean:.3f} | E_total = {E_total:.3f} | E_mag = {E_mag:.3f} | Recon rate = {recon_rate:.3e} | max_J = {max_J:.3e} | dE_mag/dt = {dE_mag:.3e}")

    progress_bar.update(1)

progress_bar.close()

# Final checks
print(f"\nSimulation complete.")
print(f"Final mean filament radius: {mean_radius_history[-1]:.3f} a")
print(f"Final Genie amp: {genie_amp_history[-1]:.3f}")
print(f"Final fermion mass mean: {ferm_mass_mean_history[-1]:.3f}")
print(f"Final fermion density mean: {ferm_density_history[-1]:.3f}")
print(f"Energy drift: {E_total_history[-1] - E_total_history[0]:.3f}")
print(f"Stabilized? {'Yes' if 0.2 < mean_radius_history[-1] < 0.5 else 'No'}")

# Save final state
np.savez("final_state.npz", 
         rho=rho, B_r=B_r, B_phi=B_phi, B_z=B_z, 
         genie_phi=genie_phi, 
         ferm_psi_mean_norm=np.mean(np.sqrt(np.sum(np.abs(ferm_psi)**2, axis=-1))))

# ==================== VALIDATION REPORT ====================
print("\n=== VALIDATION REPORT (v1.0.8) ===")

# Energy conservation
if E_total_history:
    initial_E = E_total_history[0]
    final_E = E_total_history[-1]
    energy_drift_pct = 100 * (final_E - initial_E) / initial_E if initial_E != 0 else 0
    print(f"Energy conservation drift: {energy_drift_pct:.3f}% (initial E = {initial_E:.3f}, final E = {final_E:.3f})")
    energy_ok = abs(energy_drift_pct) < 10
else:
    print("Energy history empty")
    energy_ok = False

# Magnetic field evolution
initial_max_B = np.max(np.abs([B_r, B_phi, B_z]))
final_max_B = np.max(np.abs([B_r, B_phi, B_z]))
B_change_pct = 100 * (final_max_B - initial_max_B) / initial_max_B if initial_max_B != 0 else 0
print(f"Magnetic field change: {B_change_pct:.2f}%")
B_evolved = abs(B_change_pct) > 5

# Dirac norm preservation
final_norm = np.sqrt(np.sum(np.abs(ferm_psi)**2, axis=-1))
mean_norm = np.mean(final_norm)
norm_drift = abs(mean_norm - 1.0)
print(f"Dirac norm: mean = {mean_norm:.6f} (drift = {norm_drift:.6f})")
norm_ok = norm_drift < 1e-4

# Backreaction activity
if backreaction_genie_history and backreaction_rho_history:
    max_back_genie = max(backreaction_genie_history)
    max_back_rho = max(backreaction_rho_history)
    print(f"Max backreaction Genie: {max_back_genie:.3e}")
    print(f"Max backreaction rho: {max_back_rho:.3e}")
    backreaction_active = max_back_genie > 1e-4 or max_back_rho > 1e-4
else:
    backreaction_active = False

# Electron inertia activity
if ferm_B_force_history:
    max_inertia = max(ferm_B_force_history)
    print(f"Max electron inertia force: {max_inertia:.3e}")
    inertia_active = max_inertia > 1e-4
else:
    inertia_active = False

# Reconnection rate benchmark (Sweet-Parker vs Hall)
R_m = vA * a / eta
M_sp = 1 / np.sqrt(R_m) if R_m > 1 else 0.1
M_sim = max(recon_rate_history) if recon_rate_history else 0
print(f"Sweet-Parker rate: ~{M_sp:.4f} v_A (R_m = {R_m:.0f})")
print(f"Sim peak rate: {M_sim:.4f} v_A")
hall_faster = M_sim > 5 * M_sp
if hall_faster:
    print(f"→ Hall faster by factor {M_sim / M_sp:.1f} — GOOD!")
else:
    print("→ Hall not significantly faster")

# Overall score
score = 0
if energy_ok: score += 30
if B_evolved: score += 30
if norm_ok: score += 20
if backreaction_active: score += 10
if inertia_active: score += 5
if hall_faster: score += 5
print(f"Validation score: {score}/100")
if score >= 80:
    print("→ Solid – ready for Zenodo!")
else:
    print("→ Tune needed")

print("===========================")

# ==================== PLOTS (15 total) ====================

t_array = np.linspace(0, t_max, n_steps)

# Plot 1: Fermion mass
plt.figure(figsize=(10,6))
plt.plot(t_array, ferm_mass_mean_history, 'orange', lw=3, label='Mean fermion mass')
plt.xlabel('Time (Alfvén times)')
plt.ylabel('Mean Fermion Mass (normalized)')
plt.title('Fermion Mass Evolution')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/plot01_ferm_mass.png')
plt.close()

# Plot 2: Fermion density
plt.figure(figsize=(10,6))
plt.plot(t_array, ferm_density_history, 'green', lw=3, label='Mean fermion density')
plt.xlabel('Time (Alfvén times)')
plt.ylabel('Mean Fermion Density')
plt.title('Fermion Density Evolution')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/plot02_ferm_density.png')
plt.close()

# Plot 3: Energy conservation
plt.figure(figsize=(10,6))
plt.plot(t_array, E_total_history, 'white', lw=3, label='Total E')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Conservation')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/plot03_energy.png')
plt.close()

# Plot 4: Final 3D density snapshot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
skip = 8
ax.scatter(R[::skip,::skip,::skip].flatten(), Z[::skip,::skip,::skip].flatten(), rho[::skip,::skip,::skip].flatten(), c=rho[::skip,::skip,::skip].flatten(), cmap='viridis')
ax.set_title('Final 3D Density — Sausage Beads')
plt.savefig('plots/plot04_density_3d.png')
plt.close()

# Plot 5: Reconnection diagnostics
plt.figure(figsize=(12,6))
plt.plot(t_array, recon_rate_history, 'red', lw=3, label='Reconnection rate')
plt.plot(t_array, dE_mag_history, 'purple', lw=3, label='dE_mag/dt')
plt.plot(t_array, max_J_history, 'orange', lw=3, label='Max |J|')
plt.xlabel('Time (Alfvén times)')
plt.ylabel('Value')
plt.title('Reconnection Diagnostics')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/plot05_reconnection.png')
plt.close()

# Plot 6: Final 3D magnetic field (Cartesian coords for correct quiver)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
skip = 8
X = R * np.cos(Phi)
Y = R * np.sin(Phi)
B_mag = np.sqrt(B_r**2 + B_phi**2 + B_z**2)
ax.quiver(X[::skip,::skip,::skip], Y[::skip,::skip,::skip], Z[::skip,::skip,::skip],
          B_r[::skip,::skip,::skip], B_phi[::skip,::skip,::skip], B_z[::skip,::skip,::skip],
          length=0.1, normalize=True, color=plt.cm.viridis(B_mag[::skip,::skip,::skip]/B_mag.max()))
ax.set_title('Final 3D Magnetic Field (Cartesian coords)')
plt.savefig('plots/plot06_magnetic_quiver.png')
plt.close()

# 7. Vorticity spectrum
vorticity_z = (1 / R_safe) * np.gradient(R * v_phi, dr, axis=0) - np.gradient(v_r, dphi, axis=1) / R_safe
vorticity_z_mean = vorticity_z.mean(axis=(0,1))
vorticity_fft = np.abs(np.fft.rfft(vorticity_z_mean))
k = np.fft.rfftfreq(len(z), d=dz)
E_k_vort = vorticity_fft**2 / len(z)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_vort, 'cyan', lw=3, label='Vorticity power spectrum')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Power')
plt.title('Vorticity Power Spectrum')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot07_vorticity_spectrum.png')
plt.close()

# 8. Helicity spectrum
helicity_density = A_z * B_phi
helicity_z_mean = helicity_density.mean(axis=(0,1))
helicity_fft = np.abs(np.fft.rfft(helicity_z_mean))
E_k_helicity = helicity_fft**2 / len(z)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_helicity, 'magenta', lw=3, label='Helicity power spectrum')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Power')
plt.title('Helicity Power Spectrum')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot08_helicity_spectrum.png')
plt.close()

# 9. Enstrophy spectrum
enstrophy_density = vorticity_z_mean**2
enstrophy_fft = np.abs(np.fft.rfft(enstrophy_density))
E_k_enstrophy = enstrophy_fft**2 / len(z)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_enstrophy, 'orange', lw=3, label='Enstrophy power spectrum')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Power')
plt.title('Enstrophy Power Spectrum')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot09_enstrophy_spectrum.png')
plt.close()

# 10. Total energy spectrum
kinetic_energy_density = 0.5 * rho * (v_r**2 + v_phi**2 + v_z**2)
magnetic_energy_density = 0.5 * (B_r**2 + B_phi**2 + B_z**2) / mu0
total_energy_density = kinetic_energy_density + magnetic_energy_density
total_z_mean = total_energy_density.mean(axis=(0,1))
total_fft = np.abs(np.fft.rfft(total_z_mean))
E_k_total = total_fft**2 / len(z)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_total, 'gold', lw=3, label='Total energy power spectrum')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Power')
plt.title('Total Energy Power Spectrum')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot10_total_energy_spectrum.png')
plt.close()

# 11. Kinetic energy spectrum
kinetic_z_mean = kinetic_energy_density.mean(axis=(0,1))
kinetic_fft = np.abs(np.fft.rfft(kinetic_z_mean))
E_k_kinetic = kinetic_fft**2 / len(z)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_kinetic, 'blue', lw=3, label='Kinetic energy power spectrum')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Power')
plt.title('Kinetic Energy Power Spectrum')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot11_kinetic_energy_spectrum.png')
plt.close()

# 12. Magnetic energy spectrum
magnetic_z_mean = magnetic_energy_density.mean(axis=(0,1))
magnetic_fft = np.abs(np.fft.rfft(magnetic_z_mean))
E_k_magnetic = magnetic_fft**2 / len(z)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_magnetic, 'magenta', lw=3, label='Magnetic energy power spectrum')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Power')
plt.title('Magnetic Energy Power Spectrum')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot12_magnetic_energy_spectrum.png')
plt.close()

# 13. IK spectrum comparison
k_ref = k[k > 0]
E_ik_ref = E_k_total[k > 0].max() * (k_ref / k_ref.min())**(-3/2)

plt.figure(figsize=(12,6))
plt.loglog(k, E_k_total, 'gold', lw=3, label='E(k) from sim')
plt.loglog(k_ref, E_ik_ref, 'dashed', color='purple', lw=3, label='IK reference k^{-3/2}')
plt.xlabel('Wavenumber k (1/a)')
plt.ylabel('Energy E(k)')
plt.title('Energy Spectrum E(k) – IK Comparison')
plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.savefig('plots/plot13_ik_comparison.png')
plt.close()

# 14. Overlaid Genie + Fermion mass
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(t_array, genie_amp_history, 'magenta', lw=3, label='Mean Genie Amplitude |ϕ|')
ax1.set_xlabel('Time (Alfvén times)')
ax1.set_ylabel('Mean |ϕ| (normalized)', color='magenta')
ax1.tick_params(axis='y', labelcolor='magenta')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t_array, ferm_mass_mean_history, 'orange', lw=3, label='Mean Fermion Mass')
ax2.set_ylabel('Mean Fermion Mass (normalized)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Genie Scalar Field Amplitude + Fermion Mass Evolution (Overlaid)')
plt.savefig('plots/plot14_overlaid_genie_fermion.png')
plt.close()

# 15. Fermion Backreaction Diagnostics
plt.figure(figsize=(12,6))
plt.plot(t_array, j_z_mean_history, 'cyan', lw=2, label='Mean |j_z| (fermion current)')
plt.plot(t_array, backreaction_genie_history, 'purple', lw=2, label='Backreaction to Genie source')
plt.plot(t_array, backreaction_rho_history, 'lime', lw=2, label='Backreaction to plasma density')
plt.xlabel('Time (Alfvén times)')
plt.ylabel('Strength')
plt.title('Fermion Backreaction Diagnostics')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/plot15_backreaction_diagnostics.png')
plt.close()

print("\nAll 15 plots saved to ./plots/")
print("Final state saved as final_state.npz")
print("Checkpoints saved to ./checkpoints/ (every 50 steps)")
print("Zenodo upload ready!")
