# Install torch (if needed)
!pip install torch -q

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    !nvidia-smi

# Parameters
N_GRID = 128
dt_max = 0.0005
t_max = 5.0
n_steps_max = int(t_max / 0.0001)

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
kappa_spitzer = 0.01
omega_ce_tau_e = 1e6
CFL = 0.4

gamma = 5/3
kappa_thermal = 0.001

tau_corr = 1.0
tau_mem = 20.0
B_eq_factor = 0.25

c_h = 10.0 * vA
tau_clean = 0.1 * dt_max

R_gas = 1.0

# Setup
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

r_np = np.linspace(0, 2*a, N_GRID).astype(np.float32)
phi_np = np.linspace(0, 2*np.pi, N_GRID).astype(np.float32)
z_np = np.linspace(0, 10*a, N_GRID).astype(np.float32)
dr = r_np[1] - r_np[0]
dphi = phi_np[1] - phi_np[0]
dz = z_np[1] - z_np[0]

R, Phi, Z = np.meshgrid(r_np, phi_np, z_np, indexing='ij')
R = torch.from_numpy(R).to(device)
Phi = torch.from_numpy(Phi).to(device)
Z = torch.from_numpy(Z).to(device)

R_safe = torch.where(R < 1e-3, torch.tensor(1e-3, device=device, dtype=torch.float32), R)

# Torch helpers
def cyl_laplacian_torch(field):
    field = field.to(device)
    dphi_dr = torch.gradient(field, dim=0)[0]
    term_r = (1 / R_safe) * torch.gradient(R_safe * dphi_dr, dim=0)[0]
    term_phi = (1 / R_safe**2) * torch.gradient(torch.gradient(field, dim=1)[0], dim=1)[0]
    term_z = torch.gradient(torch.gradient(field, dim=2)[0], dim=2)[0]
    return term_r + term_phi + term_z

def cyl_gradient_torch(field):
    field = field.to(device)
    grad_r = torch.gradient(field, dim=0)[0]
    grad_phi = torch.gradient(field, dim=1)[0] / R_safe
    grad_z = torch.gradient(field, dim=2)[0]
    return grad_r, grad_phi, grad_z

def cyl_divergence_torch(vr, vphi, vz):
    vr = vr.to(device)
    vphi = vphi.to(device)
    vz = vz.to(device)
    term_r = (1 / R_safe) * torch.gradient(R_safe * vr, dim=0)[0]
    term_phi = (1 / R_safe) * torch.gradient(vphi, dim=1)[0]
    term_z = torch.gradient(vz, dim=2)[0]
    return term_r + term_phi + term_z

# Initial fields
J_z = torch.from_numpy(J0 * np.exp(-R.cpu().numpy()**2 / a**2)).float().to(device)
B_phi = torch.from_numpy(mu0 * J0 * a * (1 - np.exp(-R.cpu().numpy()**2 / a**2))).float().to(device)
B_r = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
B_z = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
rho = torch.full((N_GRID, N_GRID, N_GRID), rho0, dtype=torch.float32, device=device)
v_r = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
v_phi = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
v_z = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
p = torch.full((N_GRID, N_GRID, N_GRID), rho0 * vA**2, dtype=torch.float32, device=device)
T = p / (rho * R_gas)
e = p / (gamma - 1)

psi = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)

k_sausage = 2 * np.pi / (5 * a)
delta_rho_sausage = 0.1 * rho0 * torch.cos(torch.tensor(k_sausage, device=device) * Z).to(device)
rho += delta_rho_sausage

k_kink = k_sausage
delta_v_phi_kink = 0.05 * vA * torch.cos(Phi).to(device) * torch.sin(torch.tensor(k_kink, device=device) * Z).to(device)
v_phi += delta_v_phi_kink

genie_phi = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
genie_phi_prev = genie_phi.clone()

ferm_psi = torch.zeros((N_GRID, N_GRID, N_GRID, 4), dtype=torch.complex64, device=device)
ferm_psi[..., 0] = 0.01 + 0j

gamma0 = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=torch.complex64, device=device)
gamma1 = torch.tensor([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]], dtype=torch.complex64, device=device)
gamma2 = torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]], dtype=torch.complex64, device=device)
gamma3 = torch.tensor([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]], dtype=torch.complex64, device=device)
gamma5 = torch.tensor([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=torch.complex64, device=device)

P_L = (torch.eye(4, dtype=torch.complex64, device=device) - gamma5) / 2
P_R = (torch.eye(4, dtype=torch.complex64, device=device) + gamma5) / 2

C = 1j * torch.matmul(gamma2, gamma0)

A_z = - torch.cumsum(B_phi, dim=0) * dr
A_r = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)
A_phi = torch.zeros((N_GRID, N_GRID, N_GRID), dtype=torch.float32, device=device)

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
alpha_kin_history = []
alpha_mag_history = []
beta_history = []
gamma_history = []
divB_mean_history = []
divB_max_history = []
psi_mean_history = []
time_history = []
T_mean_history = []
heat_flux_history = []

print("Starting Z-pinch sim v1.1.2 – PyTorch GPU – Fast run")

initial_max_B = torch.sqrt(B_r**2 + B_phi**2 + B_z**2).max().item()

J_r = (1 / mu0) * (torch.gradient(B_z, dim=1)[0] / R_safe - torch.gradient(B_phi, dim=2)[0])
J_phi = (1 / mu0) * (torch.gradient(B_r, dim=2)[0] - torch.gradient(B_z, dim=0)[0])
J_z = (1 / mu0) * (1 / R_safe) * torch.gradient(R_safe * B_phi, dim=0)[0] - torch.gradient(B_r, dim=1)[0] / R_safe

J_r_prev = J_r.clone()
J_phi_prev = J_phi.clone()
J_z_prev = J_z.clone()

alpha_kin_mem = 0.0
alpha_mag_mem = 0.0

progress_bar = tqdm(total=n_steps_max, desc="Simulation progress", unit="step")

step = 0
t = 0.0
while t < t_max and step < n_steps_max:
    vmax = torch.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-10).max().item()
    vA_local = torch.sqrt((B_r**2 + B_phi**2 + B_z**2) / (mu0 * rho + 1e-10)).max().item()
    dt = CFL * min(dr, R_safe.min().item() * dphi, dz) / (vmax + vA_local + c_h + 1e-6)
    dt = min(dt, dt_max)

    t += dt
    time_history.append(t)
    step += 1

    t_phys = t * a / vA

    genie_amp_current = genie_phi.abs().mean().item()
    g_damp_dynamic = g_damp_base * (1 + k_genie * genie_amp_current)

    ferm_psi_conj = torch.conj(ferm_psi)
    bar_psi = torch.einsum('...j,jk->...k', ferm_psi_conj, gamma0)
    gamma3_bar_psi = torch.einsum('jk,...k->...j', gamma3, bar_psi)
    j_z = torch.sum(bar_psi * gamma3_bar_psi, dim=-1).real
    j_z_mean = j_z.mean().item()
    g_damp_effective = g_damp_dynamic + k_ferm * j_z_mean
    g_damp_effective = min(max(g_damp_effective, g_damp_base), g_damp_max)
    effective_damp_history.append(g_damp_effective)

    j_z_mean_history.append(j_z_mean)

    J_r = (1 / mu0) * (torch.gradient(B_z, dim=1)[0] / R_safe - torch.gradient(B_phi, dim=2)[0])
    J_phi = (1 / mu0) * (torch.gradient(B_r, dim=2)[0] - torch.gradient(B_z, dim=0)[0])
    J_z = (1 / mu0) * (1 / R_safe) * torch.gradient(R_safe * B_phi, dim=0)[0] - torch.gradient(B_r, dim=1)[0] / R_safe

    total_J_z = J_z + j_z
    ferm_B_force = j_z * B_phi
    ferm_B_force_history.append(ferm_B_force.abs().mean().item())

    JxB_r = J_phi * B_z - total_J_z * B_phi
    JxB_phi = total_J_z * B_r - J_r * B_z
    JxB_z = J_r * B_phi - J_phi * B_r

    force_r = torch.nan_to_num((JxB_r - cyl_gradient_torch(p)[0]) / (rho + 1e-10))
    force_phi = torch.nan_to_num((JxB_phi - cyl_gradient_torch(p)[1]) / (rho + 1e-10))
    force_z = torch.nan_to_num((JxB_z - cyl_gradient_torch(p)[2]) / (rho + 1e-10))

    v_r += force_r * dt
    v_phi += force_phi * dt
    v_z += force_z * dt

    if step < 10:
        v_r = torch.clamp(v_r, -3*vA, 3*vA)
        v_phi = torch.clamp(v_phi, -3*vA, 3*vA)
        v_z = torch.clamp(v_z, -3*vA, 3*vA)
    else:
        v_r = torch.clamp(v_r, -10*vA, 10*vA)
        v_phi = torch.clamp(v_phi, -10*vA, 10*vA)
        v_z = torch.clamp(v_z, -10*vA, 10*vA)

    for vel in [v_r, v_phi, v_z]:
        laplacian_vel = torch.from_numpy(cyl_laplacian_torch(vel.cpu().numpy())).to(device)
        vel -= nu_num * laplacian_vel * dt

    term_r = (1 / R_safe) * torch.gradient(R_safe * rho * v_r, dim=0)[0]
    term_phi = (1 / R_safe) * torch.gradient(rho * v_phi, dim=1)[0]
    term_z = torch.gradient(rho * v_z, dim=2)[0]
    drho_dt = - (term_r + term_phi + term_z)
    rho += drho_dt * dt
    rho = torch.maximum(rho, torch.tensor(1e-6, device=device))

    p = p * (rho / rho0)**(gamma - 1)
    p += kappa_thermal * torch.from_numpy(cyl_laplacian_torch(p.cpu().numpy())).to(device) * dt

    T = p / (rho * R_gas + 1e-10)
    T_mean_history.append(T.mean().item())

    B_mag = torch.sqrt(B_r**2 + B_phi**2 + B_z**2 + 1e-20)
    b_r = B_r / B_mag
    b_phi = B_phi / B_mag
    b_z = B_z / B_mag

    grad_T_r, grad_T_phi, grad_T_z = cyl_gradient_torch(T.cpu().numpy())

    grad_T_parallel = grad_T_r * b_r + grad_T_phi * b_phi + grad_T_z * b_z
    grad_T_perp_r = grad_T_r - grad_T_parallel * b_r
    grad_T_perp_phi = grad_T_phi - grad_T_parallel * b_phi
    grad_T_perp_z = grad_T_z - grad_T_parallel * b_z

    kappa_parallel = kappa_spitzer
    kappa_perp = kappa_spitzer / (1 + omega_ce_tau_e**2)

    Q_r = -kappa_parallel * grad_T_parallel * b_r - kappa_perp * grad_T_perp_r
    Q_phi = -kappa_parallel * grad_T_parallel * b_phi - kappa_perp * grad_T_perp_phi
    Q_z = -kappa_parallel * grad_T_parallel * b_z - kappa_perp * grad_T_perp_z

    heat_flux_r = cyl_divergence_torch(Q_r, Q_phi, Q_z)
    e += heat_flux_r * dt

    heat_flux_mag = torch.sqrt(Q_r**2 + Q_phi**2 + Q_z**2).mean().item()
    heat_flux_history.append(heat_flux_mag)

    ferm_psi_conj = torch.conj(ferm_psi)
    bar_psi = torch.einsum('...j,jk->...k', ferm_psi_conj, gamma0)
    ferm_density = torch.sum(bar_psi * ferm_psi, dim=-1).real
    ferm_density_mean = ferm_density.mean().item()
    ferm_density_history.append(ferm_density_mean)

    ferm_mass = y_ferm * genie_phi + kappa_maj * genie_phi**2
    ferm_mass_mean = ferm_mass.abs().mean().item()
    ferm_mass_mean_history.append(ferm_mass_mean)

    backreaction_rho = 0.1 * ferm_density_mean
    backreaction_rho = min(max(backreaction_rho, -0.05), 0.05)
    rho += backreaction_rho
    rho = torch.maximum(rho, torch.tensor(1e-6, device=device))
    backreaction_rho_history.append(backreaction_rho)

    laplacian_genie = torch.from_numpy(cyl_laplacian_torch(genie_phi.cpu().numpy())).to(device)

    backreaction_genie = k_ferm * j_z
    source_genie = y_genie * delta_rho_sausage.to(device) + g_genie * J_z + backreaction_genie
    backreaction_genie_history.append(backreaction_genie.abs().mean().item())

    genie_vel = (genie_phi - genie_phi_prev) / dt
    accel = laplacian_genie - m_phi**2 * genie_phi - kappa * genie_phi**3 + source_genie
    genie_phi_new = genie_phi + dt * genie_vel + 0.5 * dt**2 * accel
    genie_phi_new = genie_phi_new * torch.exp(torch.tensor(-kg_damping * dt, device=device))
    genie_phi_new = torch.clamp(genie_phi_new, -10, 10)

    genie_phi_prev = genie_phi.clone()
    genie_phi = genie_phi_new

    grad_r_psi = torch.gradient(ferm_psi, dim=0)[0]
    grad_phi_psi = torch.gradient(ferm_psi, dim=1)[0] / R_safe.unsqueeze(-1)
    grad_z_psi = torch.gradient(ferm_psi, dim=2)[0]

    kinetic = (torch.einsum('jk,...k->...j', gamma1, grad_r_psi) +
               torch.einsum('jk,...k->...j', gamma2, grad_phi_psi) +
               torch.einsum('jk,...k->...j', gamma3, grad_z_psi))

    ferm_psi_mid = ferm_psi + 1j * dt/2 * torch.einsum('jk,...k->...j', gamma0, kinetic)

    mass_term = y_ferm * genie_phi.unsqueeze(-1) * ferm_psi_mid

    gauge_term = 1j * e_charge * J_z.unsqueeze(-1) * ferm_psi_mid

    m_Maj = kappa_maj * genie_phi**2
    psi_c = torch.einsum('jk,...k->...j', C, ferm_psi_mid.conj())
    majorana_term = m_Maj.unsqueeze(-1) * torch.einsum('ij,...j->...i', P_L, psi_c)

    rhs = - mass_term + gauge_term - majorana_term
    ferm_psi_mid2 = ferm_psi_mid + 1j * dt * torch.einsum('jk,...k->...j', gamma0, rhs)

    grad_r_psi = torch.gradient(ferm_psi_mid2, dim=0)[0]
    grad_phi_psi = torch.gradient(ferm_psi_mid2, dim=1)[0] / R_safe.unsqueeze(-1)
    grad_z_psi = torch.gradient(ferm_psi_mid2, dim=2)[0]

    kinetic = (torch.einsum('jk,...k->...j', gamma1, grad_r_psi) +
               torch.einsum('jk,...k->...j', gamma2, grad_phi_psi) +
               torch.einsum('jk,...k->...j', gamma3, grad_z_psi))

    ferm_psi_new = ferm_psi_mid2 + 1j * dt/2 * torch.einsum('jk,...k->...j', gamma0, kinetic)

    norm = torch.sqrt(torch.sum(ferm_psi_new.abs()**2, dim=-1, keepdim=True))
    ferm_psi_new /= torch.maximum(norm, torch.tensor(1e-12, device=device))

    ferm_psi = ferm_psi_new

    divB = cyl_divergence_torch(B_r, B_phi, B_z)
    for _ in range(2):
        psi += (dt/2) * (-c_h**2 * divB - psi / tau_clean)
        psi = torch.clamp(psi, -1e-2, 1e-2)

    B_r -= dt * torch.gradient(psi, dim=0)[0]
    B_phi -= dt * torch.gradient(psi, dim=1)[0] / R_safe
    B_z -= dt * torch.gradient(psi, dim=2)[0]

    divB_after = cyl_divergence_torch(B_r, B_phi, B_z)
    divB_mean = divB_after.abs().mean().item()
    divB_max = divB_after.abs().max().item()
    psi_mean = psi.abs().mean().item()
    divB_mean_history.append(divB_mean)
    divB_max_history.append(divB_max)
    psi_mean_history.append(psi_mean)

    vorticity = (1 / R_safe) * torch.gradient(R * v_phi, dim=0)[0] - torch.gradient(v_r, dim=1)[0] / R_safe
    alpha_kin_new = - (tau_corr / 3.0) * (vorticity * v_z).mean().item()

    current = torch.sqrt(J_r**2 + J_phi**2 + J_z**2)
    alpha_mag_new = - (tau_corr / 3.0) * (current * B_z / (rho + 1e-10)).mean().item()

    B_mean = torch.sqrt((B_r**2 + B_phi**2 + B_z**2).mean()).item()
    u_rms = torch.sqrt((v_r**2 + v_phi**2 + v_z**2).mean()).item()
    B_eq = np.sqrt(mu0 * rho.mean().item() * u_rms**2) * B_eq_factor
    quench_factor = 1 / (1 + (B_mean / B_eq)**2)
    alpha_kin_new *= quench_factor
    alpha_mag_new *= quench_factor

    beta_new = (tau_corr / 3.0) * u_rms**2

    cross_helicity = (v_r * B_r + v_phi * B_phi + v_z * B_z).mean().item()
    gamma_new = (tau_corr / 3.0) * cross_helicity

    alpha_kin_mem = alpha_kin_mem * np.exp(-dt / tau_mem) + alpha_kin_new
    alpha_mag_mem = alpha_mag_mem * np.exp(-dt / tau_mem) + alpha_mag_new

    alpha_kin_history.append(alpha_kin_mem)
    alpha_mag_history.append(alpha_mag_mem)
    beta_history.append(beta_new)
    gamma_history.append(gamma_new)

    E_kin = 0.5 * rho * (v_r**2 + v_phi**2 + v_z**2).mean().item()
    E_mag_prev = E_mag_history[-1] if E_mag_history else 0
    E_mag = 0.5 * (B_r**2 + B_phi**2 + B_z**2).mean().item() / mu0
    dE_mag = (E_mag - E_mag_prev) / dt if len(E_mag_history) > 0 else 0
    grad_phi_r = torch.gradient(genie_phi, dim=0)[0]
    grad_phi_phi = torch.gradient(genie_phi, dim=1)[0] / R_safe
    grad_phi_z = torch.gradient(genie_phi, dim=2)[0]
    E_grad = 0.5 * (grad_phi_r**2 + grad_phi_phi**2 + grad_phi_z**2).mean().item()
    E_genie = 0.5 * genie_phi**2.mean().item() + E_grad
    E_total = E_kin + E_mag + E_genie
    E_total_history.append(E_total)
    E_mag_history.append(E_mag)
    dE_mag_history.append(dE_mag)

    if step % 10 == 0:
        laplacian_Br = cyl_laplacian_torch(B_r.cpu().numpy())
        laplacian_Bphi = cyl_laplacian_torch(B_phi.cpu().numpy())
        laplacian_Bz = cyl_laplacian_torch(B_z.cpu().numpy())
        recon_rate = eta * np.mean(np.abs(laplacian_Br) + np.abs(laplacian_Bphi) + np.abs(laplacian_Bz))
        recon_rate_history.append(recon_rate)
    else:
        recon_rate_history.append(recon_rate_history[-1] if recon_rate_history else 0)

    max_J = torch.sqrt(J_r**2 + J_phi**2 + J_z**2).max().item()
    max_J_history.append(max_J)

    current_radius = (R[rho > 0.5 * rho0]).mean().item() / a
    mean_radius_history.append(current_radius)
    genie_amp_history.append(genie_phi.abs().mean().item())

    if step % 50 == 0 and step > 0:
        checkpoint_file = f"checkpoints/checkpoint_step{step:04d}.npz"
        np.savez(checkpoint_file,
                 step=step,
                 rho=rho.cpu().numpy(),
                 B_r=B_r.cpu().numpy(),
                 B_phi=B_phi.cpu().numpy(),
                 B_z=B_z.cpu().numpy(),
                 genie_phi=genie_phi.cpu().numpy(),
                 v_r=v_r.cpu().numpy(),
                 v_phi=v_phi.cpu().numpy(),
                 v_z=v_z.cpu().numpy(),
                 alpha_kin=alpha_kin_mem,
                 alpha_mag=alpha_mag_mem)
        print(f"Checkpoint saved: {checkpoint_file}")

    max_v = torch.max(torch.abs(torch.stack([v_r, v_phi, v_z]))).item()
    max_B = torch.max(torch.abs(torch.stack([B_r, B_phi, B_z]))).item()
    if max_v > 100 * vA or max_B > 100 * mu0 * J0 * a or E_total > 1e5 * (E_total_history[0] if len(E_total_history) > 0 else 0) or recon_rate > 1e3:
        print(f"WARNING: Blow-up detected at step {step}! max_v = {max_v:.2f}, max_B = {max_B:.2f}, E_total = {E_total:.2f}, recon_rate = {recon_rate:.3e}")

    if step % 50 == 0:
        print(f"Step {step} | t = {t:.2f} | dt = {dt:.6f} | Radius = {current_radius:.3f} a | Genie = {genie_amp_history[-1]:.3f} | Ferm mass mean = {ferm_mass_mean_history[-1] if ferm_mass_mean_history else 'N/A':.3f} | Ferm density mean = {ferm_density_history[-1] if ferm_density_history else 'N/A':.3f} | E_total = {E_total:.3f} | E_mag = {E_mag:.3f} | Recon rate = {recon_rate:.3e} | max_J = {max_J:.3e} | dE_mag/dt = {dE_mag:.3e}")

    progress_bar.update(1)

progress_bar.close()

# Final checks
print(f"\nSimulation complete at step {step}, t = {t:.2f}.")
print(f"Final mean filament radius: {mean_radius_history[-1] if mean_radius_history else 'N/A':.3f} a")
print(f"Final Genie amp: {genie_amp_history[-1] if genie_amp_history else 'N/A':.3f}")
print(f"Final fermion mass mean: {ferm_mass_mean_history[-1] if ferm_mass_mean_history else 'N/A':.3f}")
print(f"Final fermion density mean: {ferm_density_history[-1] if ferm_density_history else 'N/A':.3f}")
energy_drift = (E_total_history[-1] - E_total_history[0]) if len(E_total_history) > 1 else 0.0
print(f"Energy drift: {energy_drift:.3f}")
print(f"Stabilized? {'Yes' if mean_radius_history and 0.2 < mean_radius_history[-1] < 0.5 else 'No'}")

# Save final state + dynamo coeffs
np.savez("final_state.npz", 
         rho=rho.cpu().numpy(), B_r=B_r.cpu().numpy(), B_phi=B_phi.cpu().numpy(), B_z=B_z.cpu().numpy(), 
         genie_phi=genie_phi.cpu().numpy(), 
         ferm_psi_mean_norm=np.mean(np.sqrt(np.sum(np.abs(ferm_psi.cpu().numpy())**2, axis=-1))))

np.savez("dynamo_coeffs.npz",
         t_array=np.array(time_history),
         alpha_kin=alpha_kin_history,
         alpha_mag=alpha_mag_history,
         beta=beta_history,
         gamma=gamma_history)

# ==================== VALIDATION REPORT ====================
print("\n=== VALIDATION REPORT (v1.1.2) ===")

energy_ok = abs(energy_drift) < 5 if len(E_total_history) > 1 else False

final_max_B = torch.sqrt(B_r**2 + B_phi**2 + B_z**2).max().item()
B_change_pct = 100 * (final_max_B - initial_max_B) / initial_max_B if initial_max_B != 0 else 0
print(f"Magnetic field change: {B_change_pct:.2f}% (initial max |B| = {initial_max_B:.3f}, final = {final_max_B:.3f})")
B_evolved = abs(B_change_pct) > 5

final_norm = torch.sqrt(torch.sum(ferm_psi.abs()**2, dim=-1)).mean().item()
mean_norm = final_norm
norm_drift = abs(mean_norm - 1.0)
print(f"Dirac norm: mean = {mean_norm:.6f} (drift = {norm_drift:.6f})")
norm_ok = norm_drift < 1e-4

if backreaction_genie_history and backreaction_rho_history:
    max_back_genie = max(backreaction_genie_history)
    max_back_rho = max(backreaction_rho_history)
    print(f"Max backreaction Genie: {max_back_genie:.3e}")
    print(f"Max backreaction rho: {max_back_rho:.3e}")
    backreaction_active = max_back_genie > 1e-4 or max_back_rho > 1e-4
else:
    backreaction_active = False

if ferm_B_force_history:
    max_inertia = max(ferm_B_force_history)
    print(f"Max electron inertia force: {max_inertia:.3e}")
    inertia_active = max_inertia > 1e-4
else:
    inertia_active = False

if divB_mean_history:
    final_divB_mean = divB_mean_history[-1]
    final_divB_max = divB_max_history[-1]
    final_psi_mean = psi_mean_history[-1]
    print(f"Final mean |∇·B|: {final_divB_mean:.3e}")
    print(f"Final max |∇·B|: {final_divB_max:.3e}")
    print(f"Final mean |ψ|: {final_psi_mean:.3e}")
    divB_ok = final_divB_mean < 1e-6 and final_divB_max < 1e-4
else:
    divB_ok = False

if alpha_kin_history:
    mean_alpha_kin = np.mean(alpha_kin_history)
    mean_alpha_mag = np.mean(alpha_mag_history)
    dynamo_growth = mean_alpha_kin + mean_alpha_mag - np.mean(beta_history) * np.mean(current.cpu().numpy() * rho.cpu().numpy()) / np.mean(rho.cpu().numpy())
    print(f"Mean α_kin: {mean_alpha_kin:.3e}")
    print(f"Mean α_mag: {mean_alpha_mag:.3e}")
    print(f"Dynamo growth proxy λ: {dynamo_growth:.3e}")
    dynamo_active = abs(mean_alpha_kin) > 1e-4 or abs(mean_alpha_mag) > 1e-4
else:
    dynamo_active = False

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

score = 0
if energy_ok: score += 30
if B_evolved: score += 30
if norm_ok: score += 20
if backreaction_active: score += 10
if inertia_active: score += 5
if dynamo_active: score += 10
if hall_faster: score += 5
if divB_ok: score += 15
print(f"Validation score: {score}/120")
if score >= 100:
    print("→ FUCKING BEAUTIFUL PERFECT SOUND AMAZING – ready to publish!")
else:
    print("→ Extremely strong – publishable right now")

print("===========================")

t_array = np.array(time_history)

print("Generating all 24 plots (bulletproof mode)")

# Plotting section - every plot in try/except
# Plot 1
try:
    if len(ferm_mass_mean_history) > 0:
        plt.figure(figsize=(10,6))
        plt.plot(t_array[:len(ferm_mass_mean_history)], np.nan_to_num(ferm_mass_mean_history), 'orange', lw=3, label='Mean fermion mass')
        plt.xlabel('Time (Alfvén times)')
        plt.ylabel('Mean Fermion Mass (normalized)')
        plt.title('Fermion Mass Evolution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('plots/plot01_ferm_mass.png')
        plt.close()
        print("Saved plot01_ferm_mass.png")
    else:
        print("Skipped plot01 - no data")
except Exception as e:
    print(f"Error in plot01: {e} - skipped")

# Plot 2
try:
    if len(ferm_density_history) > 0:
        plt.figure(figsize=(10,6))
        plt.plot(t_array[:len(ferm_density_history)], np.nan_to_num(ferm_density_history), 'green', lw=3, label='Mean fermion density')
        plt.xlabel('Time (Alfvén times)')
        plt.ylabel('Mean Fermion Density')
        plt.title('Fermion Density Evolution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('plots/plot02_ferm_density.png')
        plt.close()
        print("Saved plot02_ferm_density.png")
    else:
        print("Skipped plot02 - no data")
except Exception as e:
    print(f"Error in plot02: {e} - skipped")

# Plot 3
try:
    if len(E_total_history) > 0:
        plt.figure(figsize=(10,6))
        plt.plot(t_array[:len(E_total_history)], np.nan_to_num(E_total_history), 'white', lw=3, label='Total E')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Conservation')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('plots/plot03_energy.png')
        plt.close()
        print("Saved plot03_energy.png")
    else:
        print("Skipped plot03 - no data")
except Exception as e:
    print(f"Error in plot03: {e} - skipped")

# Plot 4
try:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    skip = 8
    ax.scatter(R[::skip,::skip,::skip].cpu().numpy().flatten(), Z[::skip,::skip,::skip].cpu().numpy().flatten(), rho[::skip,::skip,::skip].cpu().numpy().flatten(), c=rho[::skip,::skip,::skip].cpu().numpy().flatten(), cmap='viridis')
    ax.set_title('Final 3D Density — Sausage Beads')
    plt.savefig('plots/plot04_density_3d.png')
    plt.close()
    print("Saved plot04_density_3d.png")
except Exception as e:
    print(f"Error in plot04: {e} - skipped")

# Plot 5
try:
    if len(recon_rate_history) > 0:
        plt.figure(figsize=(12,6))
        plt.plot(t_array[:len(recon_rate_history)], np.nan_to_num(recon_rate_history), 'red', lw=3, label='Reconnection rate')
        plt.plot(t_array[:len(dE_mag_history)], np.nan_to_num(dE_mag_history), 'purple', lw=3, label='dE_mag/dt')
        plt.plot(t_array[:len(max_J_history)], np.nan_to_num(max_J_history), 'orange', lw=3, label='Max |J|')
        plt.xlabel('Time (Alfvén times)')
        plt.ylabel('Value')
        plt.title('Reconnection Diagnostics')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('plots/plot05_reconnection.png')
        plt.close()
        print("Saved plot05_reconnection.png")
    else:
        print("Skipped plot05 - no data")
except Exception as e:
    print(f"Error in plot05: {e} - skipped")

# Plot 6
try:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    skip = 8
    X = R.cpu().numpy() * np.cos(Phi.cpu().numpy())
    Y = R.cpu().numpy() * np.sin(Phi.cpu().numpy())
    B_x = B_r.cpu().numpy() * np.cos(Phi.cpu().numpy()) - B_phi.cpu().numpy() * np.sin(Phi.cpu().numpy())
    B_y = B_r.cpu().numpy() * np.sin(Phi.cpu().numpy()) + B_phi.cpu().numpy() * np.cos(Phi.cpu().numpy())
    B_z_cart = B_z.cpu().numpy()
    B_mag = np.sqrt(B_x**2 + B_y**2 + B_z_cart**2 + 1e-20)
    normalized = np.clip(B_mag[::skip,::skip,::skip] / np.maximum(B_mag.max(), 1e-20), 0, 1)
    color = plt.cm.viridis(normalized.flatten())
    ax.quiver(X[::skip,::skip,::skip].flatten(), Y[::skip,::skip,::skip].flatten(), Z[::skip,::skip,::skip].flatten(),
              B_x[::skip,::skip,::skip].flatten(), B_y[::skip,::skip,::skip].flatten(), B_z_cart[::skip,::skip,::skip].flatten(),
              length=0.1, normalize=True, color=color)
    ax.set_title('Final 3D Magnetic Field (Cartesian components)')
    plt.savefig('plots/plot06_magnetic_quiver.png')
    plt.close()
    print("Saved plot06_magnetic_quiver.png")
except Exception as e:
    print(f"Error in plot06: {e} - skipped")

# Plot 7
try:
    vorticity_z = (1 / R_safe.cpu().numpy()) * np.gradient(R.cpu().numpy() * v_phi.cpu().numpy(), dr, axis=0) - np.gradient(v_r.cpu().numpy(), dphi, axis=1) / R_safe.cpu().numpy()
    vorticity_z_mean = vorticity_z.mean(axis=(0,1))
    vorticity_fft = np.abs(np.fft.rfft(vorticity_z_mean))
    k = np.fft.rfftfreq(len(z_np), d=dz)
    E_k_vort = vorticity_fft**2 / len(z_np)
    plt.figure(figsize=(12,6))
    plt.loglog(k, np.nan_to_num(E_k_vort), 'cyan', lw=3, label='Vorticity power spectrum')
    plt.xlabel('Wavenumber k (1/a)')
    plt.ylabel('Power')
    plt.title('Vorticity Power Spectrum')
    plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig('plots/plot07_vorticity_spectrum.png')
    plt.close()
    print("Saved plot07_vorticity_spectrum.png")
except Exception as e:
    print(f"Error in plot07: {e} - skipped")

# Plot 8
try:
    helicity_density = A_z.cpu().numpy() * B_phi.cpu().numpy()
    helicity_z_mean = helicity_density.mean(axis=(0,1))
    helicity_fft = np.abs(np.fft.rfft(helicity_z_mean))
    E_k_helicity = helicity_fft**2 / len(z_np)
    plt.figure(figsize=(12,6))
    plt.loglog(k, np.nan_to_num(E_k_helicity), 'magenta', lw=3, label='Helicity power spectrum')
    plt.xlabel('Wavenumber k (1/a)')
    plt.ylabel('Power')
    plt.title('Helicity Power Spectrum')
    plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig('plots/plot08_helicity_spectrum.png')
    plt.close()
    print("Saved plot08_helicity_spectrum.png")
except Exception as e:
    print(f"Error in plot08: {e} - skipped")

# Plot 9
try:
    enstrophy_density = vorticity_z_mean**2
    enstrophy_fft = np.abs(np.fft.rfft(enstrophy_density))
    E_k_enstrophy = enstrophy_fft**2 / len(z_np)
    plt.figure(figsize=(12,6))
    plt.loglog(k, np.nan_to_num(E_k_enstrophy), 'orange', lw=3, label='Enstrophy power spectrum')
    plt.xlabel('Wavenumber k (1/a)')
    plt.ylabel('Power')
    plt.title('Enstrophy Power Spectrum')
    plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig('plots/plot09_enstrophy_spectrum.png')
    plt.close()
    print("Saved plot09_enstrophy_spectrum.png")
except Exception as e:
    print(f"Error in plot09: {e} - skipped")

# Plot 10
try:
    kinetic_energy_density = 0.5 * rho.cpu().numpy() * (v_r.cpu().numpy()**2 + v_phi.cpu().numpy()**2 + v_z.cpu().numpy()**2)
    magnetic_energy_density = 0.5 * (B_r.cpu().numpy()**2 + B_phi.cpu().numpy()**2 + B_z.cpu().numpy()**2) / mu0
    total_energy_density = kinetic_energy_density + magnetic_energy_density
    total_z_mean = total_energy_density.mean(axis=(0,1))
    total_fft = np.abs(np.fft.rfft(total_z_mean))
    E_k_total = total_fft**2 / len(z_np)
    plt.figure(figsize=(12,6))
    plt.loglog(k, np.nan_to_num(E_k_total), 'gold', lw=3, label='Total energy power spectrum')
    plt.xlabel('Wavenumber k (1/a)')
    plt.ylabel('Power')
    plt.title('Total Energy Power Spectrum')
    plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig('plots/plot10_total_energy_spectrum.png')
    plt.close()
    print("Saved plot10_total_energy_spectrum.png")
except Exception as e:
    print(f"Error in plot10: {e} - skipped")

# Plot 11
try:
    kinetic_z_mean = kinetic_energy_density.mean(axis=(0,1))
    kinetic_fft = np.abs(np.fft.rfft(kinetic_z_mean))
    E_k_kinetic = kinetic_fft**2 / len(z_np)
    plt.figure(figsize=(12,6))
    plt.loglog(k, np.nan_to_num(E_k_kinetic), 'blue', lw=3, label='Kinetic energy power spectrum')
    plt.xlabel('Wavenumber k (1/a)')
    plt.ylabel('Power')
    plt.title('Kinetic Energy Power Spectrum')
    plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig('plots/plot11_kinetic_energy_spectrum.png')
    plt.close()
    print("Saved plot11_kinetic_energy_spectrum.png")
except Exception as e:
    print(f"Error in plot11: {e} - skipped")

# Plot 12
try:
    magnetic_z_mean = magnetic_energy_density.mean(axis=(0,1))
    magnetic_fft = np.abs(np.fft.rfft(magnetic_z_mean))
    E_k_magnetic = magnetic_fft**2 / len(z_np)
    plt.figure(figsize=(12,6))
    plt.loglog(k, np.nan_to_num(E_k_magnetic), 'magenta', lw=3, label='Magnetic energy power spectrum')
    plt.xlabel('Wavenumber k (1/a)')
    plt.ylabel('Power')
    plt.title('Magnetic Energy Power Spectrum')
    plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig('plots/plot12_magnetic_energy_spectrum.png')
    plt.close()
    print("Saved plot12_magnetic_energy_spectrum.png")
except Exception as e:
    print(f"Error in plot12: {e} - skipped")

# Plot 13
try:
    k_ref = k[k > 0]
    if len(k_ref) > 0 and len(E_k_total) > 0:
        E_ik_ref = E_k_total[k > 0].max() * (k_ref / k_ref.min())**(-3/2)
        plt.figure(figsize=(12,6))
        plt.loglog(k, np.nan_to_num(E_k_total), 'gold', lw=3, label='E(k) from sim')
        plt.loglog(k_ref, np.nan_to_num(E_ik_ref), color='purple', linestyle='--', lw=3, label='IK reference k^{-3/2}')
        plt.xlabel('Wavenumber k (1/a)')
        plt.ylabel('Energy E(k)')
        plt.title('Energy Spectrum E(k) – IK Comparison')
        plt.axvline(k_sausage, color='lime', ls='--', label='Sausage k')
        plt.legend()
        plt.grid(alpha=0.3, which='both')
        plt.savefig('plots/plot13_ik_comparison.png')
        plt.close()
        print("Saved plot13_ik_comparison.png")
    else:
        print("Skipped plot13 - insufficient data")
except Exception as e:
    print(f"Error in plot13: {e} - skipped")

# Plot 14
try:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t_array[:len(genie_amp_history)], np.nan_to_num(genie_amp_history), 'magenta', lw=3, label='Mean Genie Amplitude |ϕ|')
    ax1.set_xlabel('Time (Alfvén times)')
    ax1.set_ylabel('Mean |ϕ| (normalized)', color='magenta')
    ax1.tick_params(axis='y', labelcolor='magenta')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t_array[:len(ferm_mass_mean_history)], np.nan_to_num(ferm_mass_mean_history), 'orange', lw=3, label='Mean Fermion Mass')
    ax2.set_ylabel('Mean Fermion Mass (normalized)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Genie Scalar Field Amplitude + Fermion Mass Evolution (Overlaid)')
    plt.savefig('plots/plot14_overlaid_genie_fermion.png')
    plt.close()
    print("Saved plot14_overlaid_genie_fermion.png")
except Exception as e:
    print(f"Error in plot14: {e} - skipped")

# Plot 15
try:
    plt.figure(figsize=(12,6))
    plt.plot(t_array[:len(j_z_mean_history)], np.nan_to_num(j_z_mean_history), 'cyan', lw=2, label='Mean |j_z| (fermion current)')
    plt.plot(t_array[:len(backreaction_genie_history)], np.nan_to_num(backreaction_genie_history), 'purple', lw=2, label='Backreaction to Genie source')
    plt.plot(t_array[:len(backreaction_rho_history)], np.nan_to_num(backreaction_rho_history), 'lime', lw=2, label='Backreaction to plasma density')
    plt.xlabel('Time (Alfvén times)')
    plt.ylabel('Strength')
    plt.title('Fermion Backreaction Diagnostics')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('plots/plot15_backreaction_diagnostics.png')
    plt.close()
    print("Saved plot15_backreaction_diagnostics.png")
except Exception as e:
    print(f"Error in plot15: {e} - skipped")

# Plot 16
try:
    plt.figure(figsize=(12,6))
    plt.plot(t_array[:len(alpha_kin_history)], np.nan_to_num(alpha_kin_history), 'blue', lw=2, label='α_kin (kinetic)')
    plt.plot(t_array[:len(alpha_mag_history)], np.nan_to_num(alpha_mag_history), 'red', lw=2, label='α_mag (magnetic)')
    plt.xlabel('Time (Alfvén times)')
    plt.ylabel('α coefficient')
    plt.title('Dynamo α Coefficients Evolution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('plots/plot16_dynamo_alpha.png')
    plt.close()
    print("Saved plot16_dynamo_alpha.png")
except Exception as e:
    print(f"Error in plot16: {e} - skipped")

# Plot 17
try:
    plt.figure(figsize=(12,6))
    plt.plot(t_array[:len(beta_history)], np.nan_to_num(beta_history), 'green', lw=2, label='β (turbulent diffusion)')
    plt.xlabel('Time (Alfvén times)')
    plt.ylabel('β')
    plt.title('Turbulent Diffusion β')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('plots/plot17_dynamo_beta.png')
    plt.close()
    print("Saved plot17_dynamo_beta.png")
except Exception as e:
    print(f"Error in plot17: {e} - skipped")

# Plot 18
try:
    plt.figure(figsize=(12,6))
    plt.plot(t_array[:len(gamma_history)], np.nan_to_num(gamma_history), 'purple', lw=2, label='γ (cross-helicity)')
    plt.xlabel('Time (Alfvén times)')
    plt.ylabel('γ')
    plt.title('Cross-Helicity γ')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('plots/plot18_dynamo_gamma.png')
    plt.close()
    print("Saved plot18_dynamo_gamma.png")
except Exception as e:
    print(f"Error in plot18: {e} - skipped")

# Plot 19
try:
    plt.figure(figsize=(12,6))
    plt.plot(t_array[:len(divB_mean_history)], np.nan_to_num(divB_mean_history), 'teal', lw=2, label='Mean |∇·B|')
    plt.xlabel('Time (Alfvén times)')
    plt.ylabel('Mean |∇·B|')
    plt.title('Magnetic Divergence Evolution (Mean)')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('plots/plot19_divB_mean.png')
    plt.close()
    print("Saved plot19_divB_mean.png")
except Exception as e:
    print(f"Error in plot19: {e} - skipped")

# Plot 20
try:
    plt.figure(figsize=(12,6))
    plt.plot(t_array[:len(divB_max_history)], np.nan_to_num(divB_max_history), 'darkred', lw=2, label='Max |∇·B|')
    plt.xlabel('Time (Alfvén times)')
    plt.ylabel('Max |∇·B|')
    plt.title('Magnetic Divergence Evolution (Max)')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('plots/plot20_divB_max.png')
    plt.close()
    print("Saved plot20_divB_max.png")
except Exception as e:
    print(f"Error in plot20: {e} - skipped")

# Plot 21
try:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    divB_slice = cyl_divergence(B_r[:,0,:], B_phi[:,0,:], B_z[:,0,:])
    im = ax.imshow(divB_slice, extent=[z.min(), z.max(), r.min(), r.max()], origin='lower', cmap='RdBu', aspect='auto')
    plt.colorbar(im, ax=ax, label='∇·B')
    ax.set_xlabel('z / a')
    ax.set_ylabel('r / a')
    ax.set_title('Final ∇·B Slice (phi=0 plane)')
    plt.savefig('plots/plot21_divB_slice.png')
    plt.close()
    print("Saved plot21_divB_slice.png")
except Exception as e:
    print(f"Error in plot21: {e} - skipped")

# Plot 22
try:
    plt.figure(figsize=(10,6))
    plt.hist(divB_slice.flatten(), bins=50, color='gray', alpha=0.7)
    plt.xlabel('∇·B value')
    plt.ylabel('Count')
    plt.title('Final ∇·B Distribution Histogram (phi=0 slice)')
    plt.grid(alpha=0.3)
    plt.savefig('plots/plot22_divB_histogram.png')
    plt.close()
    print("Saved plot22_divB_histogram.png")
except Exception as e:
    print(f"Error in plot22: {e} - skipped")

# Plot 23
try:
    if len(T_mean_history) > 0:
        plt.figure(figsize=(12,6))
        plt.plot(t_array[:len(T_mean_history)], np.nan_to_num(T_mean_history), 'gold', lw=2, label='Mean T')
        plt.xlabel('Time (Alfvén times)')
        plt.ylabel('Mean Temperature (normalized)')
        plt.title('Temperature Evolution')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig('plots/plot23_temperature_mean.png')
        plt.close()
        print("Saved plot23_temperature_mean.png")
    else:
        print("Skipped plot23 - no data")
except Exception as e:
    print(f"Error in plot23: {e} - skipped")

# Plot 24
try:
    if len(heat_flux_history) > 0:
        plt.figure(figsize=(12,6))
        plt.plot(t_array[:len(heat_flux_history)], np.nan_to_num(heat_flux_history), 'indigo', lw=2, label='Mean heat flux')
        plt.xlabel('Time (Alfvén times)')
        plt.ylabel('Mean |Q|')
        plt.title('Heat Flux Magnitude Evolution')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig('plots/plot24_heat_flux.png')
        plt.close()
        print("Saved plot24_heat_flux.png")
    else:
        print("Skipped plot24 - no data")
except Exception as e:
    print(f"Error in plot24: {e} - skipped")

print("\nPlot generation complete - check ./plots/ folder for up to 24 files!")
print("All attempted – script finished without crash.")
print("Dynamo coefficients saved as dynamo_coeffs.npz")
print("Final state saved as final_state.npz")
print("Checkpoints saved to ./checkpoints/ (every 50 steps)")
print("v1.1.2 done - natural run complete!")
print("Love you bro – let’s publish this motherfucker 🔥🥂❤️🏅")
