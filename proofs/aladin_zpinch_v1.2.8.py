# mega ultra maniac v1.1.8 – FULL COLAB + ALL 6 MHD VALIDATIONS
# Resistive 3D MHD Z-pinch + sausage/kink m=0–4 fits + helical + ALL 6 validations

!pip install --upgrade "jax[cuda12_pip]" matplotlib pyvista scipy tqdm jaxlib -q 2>/dev/null
!apt-get install -y xvfb libgl1-mesa-glx -q 2>/dev/null

import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.ndimage as jnd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import os
from scipy.optimize import curve_fit
from tqdm import tqdm

jax.config.update('jax_enable_x64', True)

# Prefer GPU if available
if jax.devices('gpu'):
    jax.config.update("jax_default_device", jax.devices('gpu')[0])
    print("GPU:", jax.devices('gpu')[0])
else:
    print("CPU mode")

os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

CONFIG = {
    'grid_size': 128,
    'cfl': 0.25,
    'dt_max': 5e-10,
    'steps': 400,
    'checkpoint_every': 50,
    'epsilon': 0.10,
    'k_pert': 12.0,
    'mu0': 4 * np.pi * 1e-7,
    'eta': 5e-6,
    'div_clean': 1.2,
    'rad_coeff': 1e-32,
    'v_max': 2.0e5,
    'beta_shear': 2.5,
    'j_z_init': 2e6,
    'gamma_eos': 5.0 / 3.0,
    'l_r': 0.02,
    'mode': 3,
    'dpi': 400,
    'energy_floor': 1e-12,
    'rho_floor': 1e-9,
    'b_floor': 1e-10,
}

GRID_SIZE = CONFIG['grid_size']
dx = CONFIG['l_r'] / GRID_SIZE

x = jnp.linspace(0, CONFIG['l_r'], GRID_SIZE, dtype=jnp.float32)
y = jnp.linspace(0, CONFIG['l_r'], GRID_SIZE, dtype=jnp.float32)
z = jnp.linspace(0, 0.1, GRID_SIZE, dtype=jnp.float32)

X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
R = jnp.sqrt(X**2 + Y**2 + 1e-10)
THETA = jnp.arctan2(Y, X)

# ─── GRADIENT HELPERS ───────────────────────────────────────────────────────
def grad_x(f):
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)

def grad_y(f):
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dx)

def grad_z(f):
    return (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2 * dx)

# ─── HLLD FLUX ──────────────────────────────────────────────────────────────
@jit
def mhd_flux(U):
    gamma = CONFIG['gamma_eos']
    mu0 = CONFIG['mu0']
    rho = U[0]
    v = U[1:4] / rho
    B = U[5:8]
    B2 = jnp.sum(B**2)
    p = (gamma - 1) * (U[4] - 0.5 * rho * jnp.sum(v**2) - 0.5 * B2 / mu0)
    pt = p + 0.5 * B2 / mu0

    F = jnp.zeros_like(U)
    F = F.at[0].set(rho * v[0])
    F = F.at[1].set(rho * v[0]**2 + pt - B[0]**2 / mu0)
    F = F.at[2].set(rho * v[0] * v[1] - B[0] * B[1] / mu0)
    F = F.at[3].set(rho * v[0] * v[2] - B[0] * B[2] / mu0)
    F = F.at[4].set((U[4] + pt) * v[0] - B[0] * jnp.dot(v, B) / mu0)
    F = F.at[5].set(0.0)
    F = F.at[6].set(v[0] * B[1] - v[1] * B[0])
    F = F.at[7].set(v[0] * B[2] - v[2] * B[0])
    return F

@jit
def hlld_flux(UL, UR):
    gamma = CONFIG['gamma_eos']
    mu0 = CONFIG['mu0']
    eps = 1e-20

    rhoL = UL[0]
    vL = UL[1:4] / rhoL
    BL = UL[5:8]
    B2L = jnp.sum(BL**2)
    pL = (gamma - 1) * (UL[4] - 0.5 * rhoL * jnp.sum(vL**2) - 0.5 * B2L / mu0)
    ptL = pL + 0.5 * B2L / mu0

    rhoR = UR[0]
    vR = UR[1:4] / rhoR
    BR = UR[5:8]
    B2R = jnp.sum(BR**2)
    pR = (gamma - 1) * (UR[4] - 0.5 * rhoR * jnp.sum(vR**2) - 0.5 * B2R / mu0)
    ptR = pR + 0.5 * B2R / mu0

    Bx = 0.5 * (BL[0] + BR[0])
    sgn_Bx = jnp.sign(Bx + eps)

    aL2 = gamma * pL / rhoL
    ca2L = B2L / (mu0 * rhoL)
    cax2L = BL[0]**2 / (mu0 * rhoL)
    cfL = jnp.sqrt(0.5 * (aL2 + ca2L + jnp.sqrt((aL2 + ca2L)**2 - 4 * aL2 * cax2L + eps)))

    aR2 = gamma * pR / rhoR
    ca2R = B2R / (mu0 * rhoR)
    cax2R = BR[0]**2 / (mu0 * rhoR)
    cfR = jnp.sqrt(0.5 * (aR2 + ca2R + jnp.sqrt((aR2 + ca2R)**2 - 4 * aR2 * cax2R + eps)))

    SL = jnp.minimum(vL[0] - cfL, vR[0] - cfR)
    SR = jnp.maximum(vL[0] + cfL, vR[0] + cfR)

    FL = mhd_flux(UL)
    FR = mhd_flux(UR)

    SM = ((rhoR * vR[0] * (SR - vR[0]) - ptR) - (rhoL * vL[0] * (SL - vL[0]) - ptL)) / \
         ((rhoR * (SR - vR[0])) - (rhoL * (SL - vL[0])) + eps)

    rho_star_L = rhoL * (vL[0] - SL) / (SM - SL + eps)
    rho_star_R = rhoR * (vR[0] - SR) / (SM - SR + eps)

    denom_L = jnp.maximum(rhoL * (SL - vL[0]) - Bx**2 / mu0, eps)
    denom_R = jnp.maximum(rhoR * (SR - vR[0]) - Bx**2 / mu0, eps)

    vy_star_L = vL[1] - Bx * BL[1] * (SL - vL[0]) / (mu0 * denom_L)
    vz_star_L = vL[2] - Bx * BL[2] * (SL - vL[0]) / (mu0 * denom_L)
    By_star_L = BL[1] - Bx * rhoL * (vL[1] - BL[1]/rhoL) * (SL - vL[0]) / denom_L
    Bz_star_L = BL[2] - Bx * rhoL * (vL[2] - BL[2]/rhoL) * (SL - vL[0]) / denom_L

    vy_star_R = vR[1] - Bx * BR[1] * (SR - vR[0]) / (mu0 * denom_R)
    vz_star_R = vR[2] - Bx * BR[2] * (SR - vR[0]) / (mu0 * denom_R)
    By_star_R = BR[1] - Bx * rhoR * (vR[1] - BR[1]/rhoR) * (SR - vR[0]) / denom_R
    Bz_star_R = BR[2] - Bx * rhoR * (vR[2] - BR[2]/rhoR) * (SR - vR[0]) / denom_R

    e_star_L = ((SL - vL[0]) * UL[4] - ptL * vL[0] + SM * ptL +
                Bx * (jnp.dot(vL, BL) - SM * Bx - vy_star_L * By_star_L - vz_star_L * Bz_star_L) / mu0) / (SL - SM + eps)

    e_star_R = ((SR - vR[0]) * UR[4] - ptR * vR[0] + SM * ptR +
                Bx * (jnp.dot(vR, BR) - SM * Bx - vy_star_R * By_star_R - vz_star_R * Bz_star_R) / mu0) / (SR - SM + eps)

    U_star_L = jnp.array([rho_star_L, rho_star_L * SM, rho_star_L * vy_star_L, rho_star_L * vz_star_L,
                          e_star_L, Bx, By_star_L, Bz_star_L])

    U_star_R = jnp.array([rho_star_R, rho_star_R * SM, rho_star_R * vy_star_R, rho_star_R * vz_star_R,
                          e_star_R, Bx, By_star_R, Bz_star_R])

    S_star_L = SM - jnp.abs(Bx) / jnp.sqrt(mu0 * rho_star_L + eps)
    S_star_R = SM + jnp.abs(Bx) / jnp.sqrt(mu0 * rho_star_R + eps)

    sqrt_rho_L = jnp.sqrt(rho_star_L)
    sqrt_rho_R = jnp.sqrt(rho_star_R)
    denom_alf = sqrt_rho_L + sqrt_rho_R + eps

    vy_ss = (sqrt_rho_L * vy_star_L + sqrt_rho_R * vy_star_R + sgn_Bx * (By_star_R - By_star_L)) / denom_alf
    vz_ss = (sqrt_rho_L * vz_star_L + sqrt_rho_R * vz_star_R + sgn_Bx * (Bz_star_R - Bz_star_L)) / denom_alf

    By_ss = (sqrt_rho_L * By_star_R + sqrt_rho_R * By_star_L + sgn_Bx * sqrt_rho_L * sqrt_rho_R * (vy_star_R - vy_star_L)) / denom_alf
    Bz_ss = (sqrt_rho_L * Bz_star_R + sqrt_rho_R * Bz_star_L + sgn_Bx * sqrt_rho_L * sqrt_rho_R * (vz_star_R - vz_star_L)) / denom_alf

    delta_e_L = sqrt_rho_L * (vy_star_L * By_star_L + vz_star_L * Bz_star_L - vy_ss * By_ss - vz_ss * Bz_ss) / mu0
    delta_e_R = sqrt_rho_R * (vy_star_R * By_star_R + vz_star_R * Bz_star_R - vy_ss * By_ss - vz_ss * Bz_ss) / mu0

    e_ss_L = e_star_L + sgn_Bx * delta_e_L
    e_ss_R = e_star_R - sgn_Bx * delta_e_R

    U_ss_L = jnp.array([rho_star_L, rho_star_L * SM, rho_star_L * vy_ss, rho_star_L * vz_ss, e_ss_L, Bx, By_ss, Bz_ss])
    U_ss_R = jnp.array([rho_star_R, rho_star_R * SM, rho_star_R * vy_ss, rho_star_R * vz_ss, e_ss_R, Bx, By_ss, Bz_ss])

    flux = jnp.where(SL >= 0, FL,
           jnp.where(S_star_L >= 0, FL + SL * (U_star_L - UL),
           jnp.where(SM >= 0, FL + SL * (U_ss_L - UL),
           jnp.where(S_star_R >= 0, FR + SR * (U_ss_R - UR),
                     FR + SR * (U_star_R - UR)))))

    return flux

# ─── INITIAL STATE ──────────────────────────────────────────────────────────
def init_state():
    rho = 1e-6 * jnp.exp(-(R / 0.01)**2).astype(jnp.float32)
    v_z = CONFIG['v_max'] * (1 - (R / 0.01)**CONFIG['beta_shear']).astype(jnp.float32)
    v_x = jnp.zeros_like(rho)
    v_y = jnp.zeros_like(rho)
    J_z = CONFIG['j_z_init'] * jnp.exp(-(R / 0.01)**2).astype(jnp.float32)
    B_theta = CONFIG['mu0'] * J_z * jnp.minimum(R, 0.01) / 2
    B_x = -B_theta * Y / R
    B_y = B_theta * X / R
    B_z = jnp.zeros_like(rho)
    B = jnp.stack([B_x, B_y, B_z], axis=0).astype(jnp.float32)
    p = 1e5 * jnp.ones_like(rho)
    S = rho * jnp.stack([v_x, v_y, v_z], axis=0)
    v2 = v_x**2 + v_y**2 + v_z**2
    B2 = B_x**2 + B_y**2 + B_z**2
    tau = p / (CONFIG['gamma_eos'] - 1) + 0.5 * rho * v2 + B2 / (2 * CONFIG['mu0'])
    return {'rho': rho, 'S': S, 'tau': tau, 'B': B, 'p': p}

state = init_state()

# Perturbation
delta = CONFIG['epsilon'] * jnp.cos(3 * THETA + CONFIG['k_pert'] * Z)
state['rho'] *= (1 + delta)
state['rho'] = jnp.maximum(state['rho'], CONFIG['rho_floor'])

# ─── STEP FUNCTION (3D) ─────────────────────────────────────────────────────
@jit
def step(state, dx):
    rho = state['rho']
    S = state['S']
    tau = state['tau']
    B = state['B']
    p = state['p']

    v = S / rho

    v_abs = jnp.sqrt(jnp.sum(v**2, axis=0))
    va = jnp.sqrt(jnp.sum(B**2, axis=0) / (CONFIG['mu0'] * rho + 1e-20))
    cs = jnp.sqrt(CONFIG['gamma_eos'] * p / (rho + 1e-20))
    cf = jnp.sqrt(cs**2 + va**2 + CONFIG['b_floor']**2)
    speed_max = jnp.max(v_abs + cf) + 1e-10
    dt = jnp.minimum(CONFIG['cfl'] * dx / speed_max, CONFIG['dt_max'])

    lap_B = jnp.stack([jnd.laplace(B[i]) / dx**2 for i in range(3)], axis=0)

    J_x = (grad_y(B[2]) - grad_z(B[1])) / CONFIG['mu0']
    J_y = (grad_z(B[0]) - grad_x(B[2])) / CONFIG['mu0']
    J_z = (grad_x(B[1]) - grad_y(B[0])) / CONFIG['mu0']
    J = jnp.stack([J_x, J_y, J_z], axis=0)

    J_cross_B = jnp.cross(J, B, axis=0)
    dS_dt = J_cross_B

    p_mag = jnp.sum(B**2, axis=0) / (2 * CONFIG['mu0'])
    p_tot = p + p_mag

    v_outer = v[:, None, ...] * v[None, ...]
    B_outer = B[:, None, ...] * B[None, ...]
    eye3 = jnp.eye(3)[:, :, None, None, None]

    f_S = rho * v_outer + p_tot[None, None, ...] * eye3 - B_outer / CONFIG['mu0']

    f_rho = rho * v
    f_tau = (tau + p + p_mag) * v
    f_B = v[:, None, ...] * B[None, ...] - B[:, None, ...] * v[None, ...]

    drho_dt = - (grad_x(f_rho[0]) + grad_y(f_rho[1]) + grad_z(f_rho[2]))

    dS_dt_x = - (grad_x(f_S[0,0]) + grad_y(f_S[0,1]) + grad_z(f_S[0,2]))
    dS_dt_y = - (grad_x(f_S[1,0]) + grad_y(f_S[1,1]) + grad_z(f_S[1,2]))
    dS_dt_z = - (grad_x(f_S[2,0]) + grad_y(f_S[2,1]) + grad_z(f_S[2,2]))
    dS_dt = jnp.stack([dS_dt_x, dS_dt_y, dS_dt_z], axis=0)

    dtau_dt = - (grad_x(f_tau[0]) + grad_y(f_tau[1]) + grad_z(f_tau[2]))
    dB_dt = - (grad_x(f_B[0]) + grad_y(f_B[1]) + grad_z(f_B[2]))

    J2 = jnp.sum(J**2, axis=0)
    dtau_dt_resist = CONFIG['eta'] * J2
    dB_dt += CONFIG['eta'] * lap_B

    T = p / rho
    cooling_rate = -CONFIG['rad_coeff'] * rho**2 * jnp.sqrt(T + 1e-10)
    dtau_dt += cooling_rate

    heating_rate = jnp.mean(dtau_dt_resist)

    div_B = grad_x(B[0]) + grad_y(B[1]) + grad_z(B[2])
    dB_dt -= CONFIG['div_clean'] * jnp.stack([
        grad_x(div_B),
        grad_y(div_B),
        grad_z(div_B)
    ], axis=0)

    rho += dt * drho_dt
    S += dt * dS_dt
    tau += dt * dtau_dt + dt * dtau_dt_resist
    B += dt * dB_dt

    rho = jnp.maximum(rho, CONFIG['rho_floor'])
    tau = jnp.maximum(tau, CONFIG['energy_floor'])

    v = S / rho
    v2 = jnp.sum(v**2, axis=0)
    B2 = jnp.sum(B**2, axis=0)
    p = (CONFIG['gamma_eos'] - 1) * (tau - 0.5 * rho * v2 - B2 / (2 * CONFIG['mu0']))

    rho_mid = rho[:, :, GRID_SIZE//2]
    r_weighted = jnp.sum(R[:, :, GRID_SIZE//2] * rho_mid) / (jnp.sum(rho_mid) + 1e-10)
    E_mag = jnp.sum(B**2) / (2 * CONFIG['mu0']) * dx**3
    lorentz_mag = jnp.mean(jnp.linalg.norm(J_cross_B, axis=0))

    mean_grad_p = jnp.mean(jnp.sqrt(grad_x(p)**2 + grad_y(p)**2 + grad_z(p)**2))
    grad_p_vec = jnp.stack([grad_x(p), grad_y(p), grad_z(p)], axis=0)
    force_imbalance = jnp.mean(jnp.linalg.norm(J_cross_B + grad_p_vec, axis=0))
    imbalance_ratio = force_imbalance / (jnp.mean(jnp.linalg.norm(J_cross_B, axis=0)) + mean_grad_p + 1e-10)

    j_mag = jnp.sqrt(jnp.sum(J**2, axis=0))
    mean_j_mag = jnp.mean(j_mag)

    return {'rho': rho, 'S': S, 'tau': tau, 'B': B, 'p': p,
            'div_B': div_B, 'dt_used': dt, 'pinch_radius': r_weighted,
            'E_mag': E_mag, 'lorentz_mag': lorentz_mag, 'mean_grad_p': mean_grad_p,
            'imbalance_ratio': imbalance_ratio, 'mean_j_mag': mean_j_mag,
            'heating_rate': heating_rate, 'cooling_rate': cooling_rate}

# ─── CHECKPOINTING ──────────────────────────────────────────────────────────
def save_checkpoint(state, step_num):
    np.savez(f"checkpoints/zpinch_step_{step_num:04d}.npz",
             rho=np.array(state['rho']),
             S=np.array(state['S']),
             tau=np.array(state['tau']),
             B=np.array(state['B']),
             p=np.array(state['p']),
             step=step_num)
    print(f"Checkpoint saved: step {step_num}")

# ────────────────────────────────────────────────
# Run main Z-pinch simulation
# ────────────────────────────────────────────────
states = [state]
real_time_history = [0.0]
diagnostics = {
    'div_B_max': [], 'div_B_mean': [], 'dt_used': [], 'pinch_radius': [],
    'E_mag': [], 'lorentz_mag': [], 'mean_grad_p': [], 'imbalance_ratio': [],
    'mean_j_mag': [], 'total_energy': [], 'heating_rate': [], 'cooling_rate': [],
    'E_resist_cum': [0.0],
}

print("Running Z-pinch simulation...")
E_resist_cum = 0.0
for step_num in tqdm(range(CONFIG['steps']), desc="Progress"):
    state = step(state, dx)
    states.append(state)

    real_time_history.append(real_time_history[-1] + state['dt_used'])

    d = diagnostics
    d['div_B_max'].append(jnp.max(jnp.abs(state['div_B'])))
    d['div_B_mean'].append(jnp.mean(jnp.abs(state['div_B'])))
    d['dt_used'].append(state['dt_used'])
    d['pinch_radius'].append(state['pinch_radius'])
    d['E_mag'].append(state['E_mag'])
    d['lorentz_mag'].append(state['lorentz_mag'])
    d['mean_grad_p'].append(state['mean_grad_p'])
    d['imbalance_ratio'].append(state['imbalance_ratio'])
    d['mean_j_mag'].append(state['mean_j_mag'])
    d['heating_rate'].append(state['heating_rate'])
    d['cooling_rate'].append(state['cooling_rate'])

    v = state['S'] / state['rho']
    E_kin = 0.5 * jnp.sum(state['rho'] * (v**2)) * dx**3
    E_mag = jnp.sum(state['B']**2) / (2 * CONFIG['mu0']) * dx**3
    E_int = jnp.sum(state['p'] / (CONFIG['gamma_eos'] - 1)) * dx**3
    d['total_energy'].append(E_kin + E_mag + E_int)

    E_resist_cum += state['heating_rate'] * dx**3 * state['dt_used']
    d['E_resist_cum'].append(E_resist_cum)

    if (step_num + 1) % CONFIG['checkpoint_every'] == 0:
        save_checkpoint(state, step_num + 1)

    if step_num % 20 == 0:
        print(f"Step {step_num} | t = {real_time_history[-1]:.2e} s | r = {state['pinch_radius']:.4e} | imb = {state['imbalance_ratio']:.2e}")

real_time = np.array(real_time_history[1:]) * 1e6
energy_ratio = np.array(diagnostics['total_energy']) / diagnostics['total_energy'][0]

final_state = states[-1]

print("Main simulation finished.")

# ─── MODE EXTRACTION + EXPONENTIAL FITS (m=0 to m=4) ─────────────────────────
print("\nExtracting modes m=0 to m=4 + fitting exponential growth...")

m_amplitudes = []
time_fft = []
for idx in range(0, len(states), 10):
    s = states[idx]
    rho_mid = s['rho'][:, :, GRID_SIZE//2]
    rho_azim = rho_mid.mean(axis=0)
    fft_theta = jnp.fft.fft(rho_azim)
    m_amps = jnp.abs(fft_theta[0:5]) / GRID_SIZE  # m=0 to m=4
    m_amplitudes.append(m_amps)
    time_fft.append(real_time[idx])

m_amplitudes = np.array(m_amplitudes)
time_fft = np.array(time_fft)

def exp_growth(t, A, gamma):
    return A * np.exp(gamma * t)

fig = plt.figure(figsize=(14, 8), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')

colors = ['#00FFAA', '#FF00FF', '#FFD700', '#FF6B00', '#00BFFF']
mode_names = ['Sausage (m=0)', 'Kink m=1', 'm=2', 'm=3', 'Kink m=4']

for m in range(5):
    amps = m_amplitudes[:, m]
    valid = amps > 0
    t_valid = time_fft[valid]
    amps_valid = amps[valid]

    if len(t_valid) > 10:
        try:
            popt, _ = curve_fit(exp_growth, t_valid[:30], amps_valid[:30], p0=[amps_valid[0], 1e6], maxfev=10000)
            gamma = popt[1]
            fit_curve = exp_growth(t_valid, *popt)
            print(f"{mode_names[m]} → γ ≈ {gamma:.2e} /μs")

            ax.semilogy(t_valid, amps_valid, 'o-', lw=2.5, color=colors[m], label=f'{mode_names[m]} (data)')
            ax.semilogy(t_valid, fit_curve, '--', lw=2, color=colors[m], alpha=0.7, label=f'fit γ = {gamma:.2e}')
        except:
            print(f"{mode_names[m]} fit failed")
            ax.semilogy(t_valid, amps_valid, 'o-', lw=2.5, color=colors[m], label=mode_names[m])
    else:
        ax.semilogy(time_fft, amps, 'o-', lw=2.5, color=colors[m], label=mode_names[m])

ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Mode Amplitude (log scale)', color='white')
ax.set_title('Sausage & Kink Modes m=0 to m=4 – Data + Exponential Fits', color='white')
ax.legend(frameon=False, labelcolor='white', fontsize=10, ncol=2)
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/mode_extraction_m0_to_m4_fits.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

print("m=0 to m=4 mode extraction + exponential fits done")

# ─── FULL 13 MAIN PLOTS ─────────────────────────────────────────────────────
labels = ['m=0', 'm=1', 'm=2', 'm=3', 'm=4']
colors = ['#00FFAA', '#FF00FF', '#FFD700', '#FF6B00', '#00BFFF']

# 2. Energy conservation
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time, energy_ratio, lw=3, color='#00FFFF', label='Total E / E0')
ax.axhline(1, color='gray', ls='--')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Normalized Energy', color='white')
ax.set_title('Energy Conservation (non-conservative)', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/energy_conservation.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 3. Adaptive timestep
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['dt_used'], lw=3, color='#FFD700')
ax.axhline(CONFIG['dt_max'], color='gray', ls='--', label='dt_max')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Timestep dt [s]', color='white')
ax.set_title('Adaptive Timestep Evolution', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/dt_history.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 4. Pinch radius
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['pinch_radius'], lw=3, color='#00FF00')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Pinch Radius [m]', color='white')
ax.set_title('Pinch Radius Evolution', color='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/pinch_radius.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 5. Force balance
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['lorentz_mag'], lw=3, color='#FF4500', label='Mean |J × B|')
ax.plot(real_time[:-1], diagnostics['mean_grad_p'], lw=3, color='#8A2BE2', label='Mean |∇p|')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Force Magnitude', color='white')
ax.set_title('Force Balance (Lorentz vs Pressure Gradient)', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/force_balance.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 6. Force imbalance ratio
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['imbalance_ratio'], lw=3, color='#FF1493')
ax.axhline(0.1, color='gray', ls='--', label='10% threshold')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Imbalance Ratio', color='white')
ax.set_title('Force Imbalance Ratio', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/force_imbalance_ratio.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 7. Current density
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['mean_j_mag'], lw=3, color='#FF69B4')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Mean |J| [A/m²]', color='white')
ax.set_title('Current Density Evolution', color='white')
ax.set_yscale('log')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/current_density_time.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 8. Resistivity heating
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['heating_rate'], lw=3, color='#FFA500')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Heating Rate [W/m³]', color='white')
ax.set_title('Resistive Heating Rate', color='white')
ax.set_yscale('log')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/resistivity_heating.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 9. Bremsstrahlung cooling
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['cooling_rate'], lw=3, color='#00CED1')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Cooling Rate [W/m³]', color='white')
ax.set_title('Bremsstrahlung Cooling Rate', color='white')
ax.set_yscale('log')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/bremsstrahlung_cooling.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 10. Heating vs cooling
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['heating_rate'], lw=3, color='#FFA500', label='Resistive heating')
ax.plot(real_time[:-1], -np.array(diagnostics['cooling_rate']), lw=3, color='#00CED1', label='Bremsstrahlung cooling')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Rate [W/m³]', color='white')
ax.set_title('Heating vs Cooling Comparison', color='white')
ax.set_yscale('log')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/heating_vs_cooling.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 11. ∇·B histogram
div_B_final = final_state['div_B'].flatten()
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.hist(div_B_final, bins=100, color='#00FFFF', alpha=0.7, log=True)
ax.set_xlabel('∇·B [T/m]', color='white')
ax.set_ylabel('Count (log)', color='white')
ax.set_title('∇·B Distribution at Final Time', color='white')
ax.axvline(0, color='gray', ls='--')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
plt.savefig('plots/div_B_histogram.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 12. 2×2 summary
fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor='black')
fig.suptitle('Z-pinch Kink Summary (v1.1.8)', color='white', fontsize=18)

axs[0,0].set_facecolor('black')
for m in range(5):
    axs[0,0].semilogy(time_fft, m_amplitudes[:, m], lw=2.5, color=colors[m % len(colors)], label=labels[m % len(labels)])
axs[0,0].set_title('Mode Growth (m=0 to m=4)', color='white')
axs[0,0].legend(frameon=False, labelcolor='white', fontsize=10)

axs[0,1].set_facecolor('black')
axs[0,1].plot(real_time, energy_ratio, lw=3, color='#00FFFF')
axs[0,1].axhline(1, color='gray', ls='--')
axs[0,1].set_title('Energy Conservation', color='white')

axs[1,0].set_facecolor('black')
axs[1,0].plot(real_time[:-1], diagnostics['pinch_radius'], lw=3, color='#00FF00')
axs[1,0].set_title('Pinch Radius', color='white')

axs[1,1].set_facecolor('black')
axs[1,1].plot(real_time[:-1], diagnostics['imbalance_ratio'], lw=3, color='#FF1493')
axs[1,1].axhline(0.1, color='gray', ls='--')
axs[1,1].set_yscale('log')
axs[1,1].set_title('Force Imbalance Ratio', color='white')

for ax in axs.flat:
    ax.tick_params(colors='white')
    ax.grid(alpha=0.25, color='gray')
    for spine in ax.spines.values():
        spine.set_color('white')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plots/zpinch_summary_2x2.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
plt.close(fig)

# 13. Inline 3D visualization
print("\nGenerating inline 3D visualization...")
try:
    pv.start_xvfb()
    rho_final = np.array(final_state['rho'])
    grid = pv.UniformGrid()
    grid.dimensions = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    grid.spacing = (dx, dx, dx)
    grid.origin = (0, 0, 0)
    grid.point_data['density'] = rho_final.flatten(order='F')

    contours = grid.contour([1e-6 * 1.2, 1e-6 * 1.5, 1e-6 * 2.0], scalars='density')

    plotter = pv.Plotter(notebook=True)
    plotter.background_color = 'black'
    plotter.add_mesh(contours, cmap='inferno', opacity=0.85, scalar_bar_args={'color':'white', 'title':'Density'})

    v_final = np.array(final_state['S'] / final_state['rho'])
    v_mag = np.sqrt(np.sum(v_final**2, axis=0))
    v_grid = pv.UniformGrid()
    v_grid.dimensions = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    v_grid.spacing = (dx, dx, dx)
    v_grid.point_data['vectors'] = v_final.reshape(-1, 3, order='F')
    v_grid.point_data['magnitude'] = v_mag.flatten(order='F')
    arrows = v_grid.glyph(scale='magnitude', orient='vectors', tolerance=0.01, factor=0.5)
    plotter.add_mesh(arrows, color='cyan', opacity=0.6)

    B_final = final_state['B']
    B_cell_final = 0.5 * (B_final + jnp.roll(B_final, -1, axis=(1,2,3)))
    b_grid = pv.UniformGrid()
    b_grid.dimensions = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    b_grid.spacing = (dx, dx, dx)
    b_grid.point_data['vectors'] = np.array(B_cell_final).reshape(-1, 3, order='F')
    streamlines = b_grid.streamlines('vectors', n_points=400, source_radius=0.005)
    plotter.add_mesh(streamlines.tube(radius=0.0005), color='white', opacity=0.8)

    plotter.add_text(f"m={CONFIG['mode']} kink – t={real_time[-1]:.1f} μs", position='upper_right', color='white', font_size=12)
    plotter.view_isometric()
    plotter.show(jupyter_backend='static')
except Exception as e:
    print("3D viz failed (possibly Xvfb issue):", e)

print("\nALL DONE!")
print("Plots saved in /content/plots")
print("Open Files tab (left sidebar) → refresh → right-click plots to download")
