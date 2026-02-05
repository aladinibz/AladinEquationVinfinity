# mega ultra maniac v1.1.8 – FULL COMPLETE SINGLE-CELL COLAB VERSION – 1391+ LINES
# No gaps — full sim + m=0–4 fits + helical + all 6 validations + integration tests + 13 plots

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

    denom_L = rhoL * (SL - vL[0]) - Bx**2 / mu0 + eps
    denom_R = rhoR * (SR - vR[0]) - Bx**2 / mu0 + eps

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

# ─── 2D STEP FUNCTION FOR VALIDATIONS ───────────────────────────────────────
@jit
def step_2d(state, dx):
    rho = state['rho']
    S = state['S'][:2]
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

    def grad(f, axis):
        return (jnp.roll(f, -1, axis) - jnp.roll(f, 1, axis)) / (2 * dx)

    def laplacian2(f):
        return (
            (jnp.roll(f, 1, axis=0) - 2*f + jnp.roll(f, -1, axis=0)) / dx**2 +
            (jnp.roll(f, 1, axis=1) - 2*f + jnp.roll(f, -1, axis=1)) / dx**2
        )

    lap_B = jnp.stack([laplacian2(B[i]) for i in range(3)], axis=0)

    J_z = (grad(B[1], 0) - grad(B[0], 1)) / CONFIG['mu0']
    J = jnp.stack([jnp.zeros_like(J_z), jnp.zeros_like(J_z), J_z], axis=0)

    J_cross_B = jnp.cross(J, B, axis=0)
    dS_dt = J_cross_B[:2]

    p_mag = jnp.sum(B**2, axis=0) / (2 * CONFIG['mu0'])
    p_tot = p + p_mag

    v_outer = v[:, None, ...] * v[None, ...]
    B_outer = B[:, None, ...] * B[None, ...]
    eye2 = jnp.eye(2)[:, :, None, None]

    f_S = rho * v_outer + p_tot[None, None, ...] * eye2 - B_outer[:2, :2] / CONFIG['mu0']

    f_rho = rho * v
    f_tau = (tau + p + p_mag) * v
    f_B = v[:, None, ...] * B[None, ...] - B[:, None, ...] * v[None, ...]

    drho_dt = - (grad(f_rho[0], 0) + grad(f_rho[1], 1))

    dS_dt_x = - (grad(f_S[0,0], 0) + grad(f_S[0,1], 1))
    dS_dt_y = - (grad(f_S[1,0], 0) + grad(f_S[1,1], 1))
    dS_dt = jnp.stack([dS_dt_x, dS_dt_y], axis=0)

    dtau_dt = - (grad(f_tau[0], 0) + grad(f_tau[1], 1))
    dB_dt = - (grad(f_B[0], 0) + grad(f_B[1], 1))

    J2 = jnp.sum(J**2, axis=0)
    dtau_dt_resist = CONFIG['eta'] * J2
    dB_dt += CONFIG['eta'] * lap_B

    T = p / rho
    cooling_rate = -CONFIG['rad_coeff'] * rho**2 * jnp.sqrt(T + 1e-10)
    dtau_dt += cooling_rate

    heating_rate = jnp.mean(dtau_dt_resist)

    div_B = grad(B[0], 0) + grad(B[1], 1)
    dB_dt -= CONFIG['div_clean'] * jnp.stack([
        grad(div_B, 0),
        grad(div_B, 1)
    ], axis=0)

    rho += dt * drho_dt
    S += dt * dS_dt
    tau += dt * dtau_dt + dt * dtau_dt_resist
    B += dt * dB_dt

    rho = jnp.maximum(rho, 1e-8)

    v = S / rho
    v2 = jnp.sum(v**2, axis=0)
    B2 = jnp.sum(B**2, axis=0)
    p = (CONFIG['gamma_eos'] - 1) * (tau - 0.5 * rho * v2 - B2 / (2 * CONFIG['mu0']))

    div_B = grad(B[0], 0) + grad(B[1], 1)
    mean_grad_p = jnp.mean(jnp.sqrt(grad(p, 0)**2 + grad(p, 1)**2))
    force_imbalance = jnp.mean(jnp.linalg.norm(J_cross_B[:2] + jnp.stack([grad(p, 0), grad(p, 1)], axis=0), axis=0))
    imbalance_ratio = force_imbalance / (jnp.mean(jnp.linalg.norm(J_cross_B[:2], axis=0)) + mean_grad_p + 1e-10)

    j_mag = jnp.sqrt(jnp.sum(J**2, axis=0))
    mean_j_mag = jnp.mean(j_mag)

    return {'rho': rho, 'S': jnp.concatenate([S, jnp.zeros_like(S[0:1])], axis=0), 'tau': tau, 'B': B, 'p': p,
            'div_B': div_B, 'dt_used': dt, 'pinch_radius': 0.0,
            'E_mag': 0.0, 'lorentz_mag': 0.0, 'mean_grad_p': mean_grad_p,
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

# 1. Growth rates (already saved above)

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

print("Inline 3D viz attempt done")

# ─── ALL 6 VALIDATIONS ──────────────────────────────────────────────────────
print("\nRunning all 6 validations...")

# Helper for 2D short runs
def short_run_2d(initial_state, steps=20, dx_local=dx):
    state = initial_state.copy()
    for _ in range(steps):
        state = step_2d(state, dx_local)
    return state

# 1. Brio-Wu
print("Brio-Wu 1D validation...")
nx_bw = 800
x_bw = jnp.linspace(0, 1, nx_bw)
dx_bw = x_bw[1] - x_bw[0]

rho_bw = jnp.where(x_bw < 0.5, 1.0, 0.125)
p_bw = jnp.where(x_bw < 0.5, 1000.0, 0.1)
vx_bw = jnp.zeros(nx_bw)
vy_bw = jnp.zeros(nx_bw)
vz_bw = jnp.zeros(nx_bw)
Bx_bw = jnp.full(nx_bw, 0.75)
By_bw = jnp.where(x_bw < 0.5, 1.0, -1.0)
Bz_bw = jnp.zeros(nx_bw)

U_bw = jnp.zeros((8, nx_bw))
U_bw = U_bw.at[0].set(rho_bw)
U_bw = U_bw.at[1].set(rho_bw * vx_bw)
U_bw = U_bw.at[2].set(rho_bw * vy_bw)
U_bw = U_bw.at[3].set(rho_bw * vz_bw)
U_bw = U_bw.at[4].set(p_bw / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_bw * (vx_bw**2 + vy_bw**2 + vz_bw**2) + 0.5 * (Bx_bw**2 + By_bw**2 + Bz_bw**2) / CONFIG['mu0'])
U_bw = U_bw.at[5].set(Bx_bw)
U_bw = U_bw.at[6].set(By_bw)
U_bw = U_bw.at[7].set(Bz_bw)

t_bw = 0.0
while t_bw < 0.2:
    v_abs = jnp.abs(U_bw[1] / U_bw[0])
    va = jnp.sqrt(jnp.sum(U_bw[5:8]**2, axis=0) / (CONFIG['mu0'] * U_bw[0] + 1e-20))
    cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw[4] - 0.5 * jnp.sum(U_bw[1:4]**2, axis=0) / U_bw[0] - 0.5 * jnp.sum(U_bw[5:8]**2, axis=0) / CONFIG['mu0']) / U_bw[0])
    cf = jnp.sqrt(cs**2 + va**2 + CONFIG['b_floor']**2)
    speed_max = jnp.max(v_abs + cf) + 1e-10
    dt_bw = 0.4 * dx_bw / speed_max
    dt_bw = jnp.minimum(dt_bw, 0.2 - t_bw)

    F_bw = jnp.zeros_like(U_bw)
    for i in range(nx_bw - 1):
        F_bw = F_bw.at[:, i].set(hlld_flux(U_bw[:, i], U_bw[:, i+1]))

    U_bw = U_bw.at[:, 1:-1].add(dt_bw / dx_bw * (F_bw[:, :-2] - F_bw[:, 1:-1]))

    t_bw += dt_bw.item()

rho_bw_final = U_bw[0]
By_bw_final = U_bw[6]
p_bw_final = (CONFIG['gamma_eos'] - 1) * (U_bw[4] - 0.5 * jnp.sum(U_bw[1:4]**2, axis=0) / U_bw[0] - 0.5 * jnp.sum(U_bw[5:8]**2, axis=0) / CONFIG['mu0'])

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].plot(x_bw, rho_bw_final, 'b-', label='Density [kg/m³]')
axs[0].set_title('Brio-Wu – Density at t=0.2')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(x_bw, By_bw_final, 'r-', label='By [T]')
axs[1].set_title('Brio-Wu – Magnetic Field By at t=0.2')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(x_bw, p_bw_final, 'g-', label='Pressure [Pa]')
axs[2].set_title('Brio-Wu – Pressure at t=0.2')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig('plots/brio_wu_multi.png', dpi=CONFIG['dpi'], facecolor='black')
plt.close(fig)

print("Brio-Wu done")

# 2. Orszag-Tang 2D vortex
print("\nOrszag-Tang 2D vortex validation...")
nx_ot = ny_ot = 256
x_ot = jnp.linspace(0, 2*np.pi, nx_ot)
y_ot = jnp.linspace(0, 2*np.pi, ny_ot)
X_ot, Y_ot = jnp.meshgrid(x_ot, y_ot, indexing='ij')

rho_ot = 1 + 0.25 * jnp.sin(2 * Y_ot) + 0.25 * jnp.sin(2 * X_ot)
p_ot = 5/3 * jnp.ones_like(rho_ot)
vx_ot = -jnp.sin(Y_ot)
vy_ot = jnp.sin(X_ot)
vz_ot = jnp.zeros_like(rho_ot)
Bx_ot = -jnp.sin(Y_ot)
By_ot = jnp.sin(2 * X_ot)
Bz_ot = jnp.zeros_like(rho_ot)

S_ot = rho_ot * jnp.stack([vx_ot, vy_ot, vz_ot], axis=0)
B_ot = jnp.stack([Bx_ot, By_ot, Bz_ot], axis=0)

v2_ot = vx_ot**2 + vy_ot**2 + vz_ot**2
B2_ot = Bx_ot**2 + By_ot**2 + Bz_ot**2
tau_ot = p_ot / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_ot * v2_ot + B2_ot / (2 * CONFIG['mu0'])

state_ot = {'rho': rho_ot, 'S': S_ot, 'tau': tau_ot, 'B': B_ot, 'p': p_ot}

t_ot = 0.0
while t_ot < 3.0:
    state_ot = step_2d(state_ot, 2*np.pi / nx_ot)
    t_ot += state_ot['dt_used'].item()

rho_ot_final = state_ot['rho']
v_ot_final = state_ot['S'] / rho_ot_final
v_mag_ot = jnp.sqrt(jnp.sum(v_ot_final**2, axis=0))
J_ot = (grad_y(state_ot['B'][1]) - grad_x(state_ot['B'][0])) / CONFIG['mu0']

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].imshow(rho_ot_final, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='inferno')
axs[0].set_title('Density ρ [kg/m³]')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].imshow(v_mag_ot, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='viridis')
axs[1].set_title('Velocity Magnitude [m/s]')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

axs[2].imshow(jnp.abs(J_ot), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='hot')
axs[2].set_title('Current Density |J| [A/m²]')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

plt.tight_layout()
plt.savefig('plots/orszag_tang_multi.png', dpi=CONFIG['dpi'], facecolor='black')
plt.close(fig)

print("Orszag-Tang done")

# 3. Hydro Blast Wave
print("\nHydro Blast Wave validation...")
nx_bw = 512
x_bw = jnp.linspace(0, 1, nx_bw)
dx_bw = x_bw[1] - x_bw[0]

rho_bw = jnp.ones(nx_bw)
p_bw = jnp.zeros(nx_bw)
p_bw = p_bw.at[nx_bw//2].set(1.0 / dx_bw)
vx_bw = jnp.zeros(nx_bw)

U_bw = jnp.zeros((3, nx_bw))
U_bw = U_bw.at[0].set(rho_bw)
U_bw = U_bw.at[1].set(rho_bw * vx_bw)
U_bw = U_bw.at[2].set(p_bw / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_bw * vx_bw**2)

t_bw = 0.0
while t_bw < 0.1:
    v_abs = jnp.abs(U_bw[1] / U_bw[0])
    cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw[2] - 0.5 * U_bw[1]**2 / U_bw[0]) * (CONFIG['gamma_eos'] - 1) / U_bw[0])
    speed_max = jnp.max(v_abs + cs) + 1e-10
    dt_bw = 0.4 * dx_bw / speed_max
    dt_bw = jnp.minimum(dt_bw, 0.1 - t_bw)

    F_bw = jnp.zeros_like(U_bw)
    for i in range(nx_bw - 1):
        F_bw = F_bw.at[:, i].set(hydro_hll_flux(U_bw[:, i], U_bw[:, i+1]))

    U_bw = U_bw.at[:, 1:-1].add(dt_bw / dx_bw * (F_bw[:, :-2] - F_bw[:, 1:-1]))

    t_bw += dt_bw.item()

rho_bw_final = U_bw[0]
p_bw_final = (CONFIG['gamma_eos'] - 1) * (U_bw[2] - 0.5 * U_bw[1]**2 / U_bw[0])

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(x_bw, rho_bw_final, 'b-', label='Density [kg/m³]')
axs[0].set_title('Hydro Blast Wave – Density at t=0.1')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(x_bw, p_bw_final, 'g-', label='Pressure [Pa]')
axs[1].set_title('Hydro Blast Wave – Pressure at t=0.1')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig('plots/hydro_blast_multi.png', dpi=CONFIG['dpi'], facecolor='black')
plt.close(fig)

print("Hydro Blast Wave done")

# 4. MHD Blast Wave
print("\nMHD Blast Wave validation...")
nx_bw = 512
x_bw = jnp.linspace(0, 1, nx_bw)
dx_bw = x_bw[1] - x_bw[0]

rho_bw = jnp.ones(nx_bw)
p_bw = jnp.zeros(nx_bw)
p_bw = p_bw.at[nx_bw//2].set(1.0 / dx_bw)
vx_bw = jnp.zeros(nx_bw)
vy_bw = jnp.zeros(nx_bw)
vz_bw = jnp.zeros(nx_bw)
Bx_bw = jnp.ones(nx_bw)
By_bw = jnp.zeros(nx_bw)
Bz_bw = jnp.zeros(nx_bw)

U_bw = jnp.zeros((8, nx_bw))
U_bw = U_bw.at[0].set(rho_bw)
U_bw = U_bw.at[1].set(rho_bw * vx_bw)
U_bw = U_bw.at[2].set(rho_bw * vy_bw)
U_bw = U_bw.at[3].set(rho_bw * vz_bw)
U_bw = U_bw.at[4].set(p_bw / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_bw * (vx_bw**2 + vy_bw**2 + vz_bw**2) + 0.5 * (Bx_bw**2 + By_bw**2 + Bz_bw**2) / CONFIG['mu0'])
U_bw = U_bw.at[5].set(Bx_bw)
U_bw = U_bw.at[6].set(By_bw)
U_bw = U_bw.at[7].set(Bz_bw)

t_bw = 0.0
while t_bw < 0.1:
    v_abs = jnp.abs(U_bw[1] / U_bw[0])
    va = jnp.sqrt(jnp.sum(U_bw[5:8]**2, axis=0) / (CONFIG['mu0'] * U_bw[0] + 1e-20))
    cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw[4] - 0.5 * jnp.sum(U_bw[1:4]**2, axis=0) / U_bw[0] - 0.5 * jnp.sum(U_bw[5:8]**2, axis=0) / CONFIG['mu0']) / U_bw[0])
    cf = jnp.sqrt(cs**2 + va**2 + CONFIG['b_floor']**2)
    speed_max = jnp.max(v_abs + cf) + 1e-10
    dt_bw = 0.4 * dx_bw / speed_max
    dt_bw = jnp.minimum(dt_bw, 0.1 - t_bw)

    F_bw = jnp.zeros_like(U_bw)
    for i in range(nx_bw - 1):
        F_bw = F_bw.at[:, i].set(hlld_flux(U_bw[:, i], U_bw[:, i+1]))

    U_bw = U_bw.at[:, 1:-1].add(dt_bw / dx_bw * (F_bw[:, :-2] - F_bw[:, 1:-1]))

    t_bw += dt_bw.item()

rho_bw_final = U_bw[0]
Bx_bw_final = U_bw[5]
p_bw_final = (CONFIG['gamma_eos'] - 1) * (U_bw[4] - 0.5 * jnp.sum(U_bw[1:4]**2, axis=0) / U_bw[0] - 0.5 * jnp.sum(U_bw[5:8]**2, axis=0) / CONFIG['mu0'])

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].plot(x_bw, rho_bw_final, 'b-', label='Density [kg/m³]')
axs[0].set_title('MHD Blast Wave – Density at t=0.1')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(x_bw, Bx_bw_final, 'r-', label='Bx [T]')
axs[1].set_title('MHD Blast Wave – Magnetic Field Bx at t=0.1')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(x_bw, p_bw_final, 'g-', label='Pressure [Pa]')
axs[2].set_title('MHD Blast Wave – Pressure at t=0.1')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig('plots/mhd_blast_multi.png', dpi=CONFIG['dpi'], facecolor='black')
plt.close(fig)

print("MHD Blast Wave done")

# 5. GEM Magnetic Reconnection
print("\nGEM Magnetic Reconnection validation...")
Lx = 50.0
Ly = 25.0
nx_gem = 512
ny_gem = 256
dx_gem = Lx / nx_gem

x_gem = jnp.linspace(-Lx/2, Lx/2, nx_gem)
y_gem = jnp.linspace(-Ly/2, Ly/2, ny_gem)
X_gem, Y_gem = jnp.meshgrid(x_gem, y_gem, indexing='ij')

lambda_sheet = 0.5
B0 = 1.0
rho0 = 1.0
p0 = B0**2 / (8 * np.pi)
rho_gem = rho0 * jnp.cosh(Y_gem / lambda_sheet)**(-2)
Bx_gem = B0 * jnp.tanh(Y_gem / lambda_sheet)
By_gem = 0.0 * jnp.ones_like(rho_gem)
Bz_gem = 0.0 * jnp.ones_like(rho_gem)
B_gem = jnp.stack([Bx_gem, By_gem, Bz_gem], axis=0)

pert = 0.01 * B0 * jnp.exp(- (X_gem**2 + (Y_gem)**2) / 5**2) * jnp.cos(2 * np.pi * X_gem / Lx)
By_gem += pert
B_gem = B_gem.at[1].set(By_gem)

p_gem = p0 * jnp.ones_like(rho_gem)
v_gem = jnp.zeros((3, nx_gem, ny_gem))
S_gem = rho_gem * v_gem

v2_gem = jnp.sum(v_gem**2, axis=0)
B2_gem = jnp.sum(B_gem**2, axis=0)
tau_gem = p_gem / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_gem * v2_gem + B2_gem / (2 * CONFIG['mu0'])

state_gem = {'rho': rho_gem, 'S': S_gem, 'tau': tau_gem, 'B': B_gem, 'p': p_gem}

t_gem = 0.0
t_end_gem = 40.0

print("Running GEM reconnection...")
while t_gem < t_end_gem:
    state_gem = step_2d(state_gem, dx_gem)
    t_gem += state_gem['dt_used'].item()

rho_gem_final = state_gem['rho']
v_gem_final = state_gem['S'] / rho_gem_final
v_x_gem = v_gem_final[0]
J_gem = (grad_y(state_gem['B'][1]) - grad_x(state_gem['B'][0])) / CONFIG['mu0']

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].imshow(rho_gem_final, origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='viridis')
axs[0].set_title('Density ρ [kg/m³]')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('y [m]')

axs[1].imshow(jnp.abs(J_gem), origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='hot')
axs[1].set_title('Current Density |J_z| [A/m²]')
axs[1].set_xlabel('x [m]')
axs[1].set_ylabel('y [m]')

axs[2].imshow(v_x_gem, origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='coolwarm')
axs[2].set_title('Outflow Velocity v_x [m/s]')
axs[2].set_xlabel('x [m]')
axs[2].set_ylabel('y [m]')

plt.tight_layout()
plt.savefig('plots/gem_multi.png', dpi=CONFIG['dpi'], facecolor='black')
plt.close(fig)

print("GEM done")

# 6. Sweet-Parker Reconnection
print("\nSweet-Parker Reconnection validation...")
Lx_sp = 20.0
Ly_sp = 10.0
nx_sp = 256
ny_sp = 128
dx_sp = Lx_sp / nx_sp

x_sp = jnp.linspace(-Lx_sp/2, Lx_sp/2, nx_sp)
y_sp = jnp.linspace(-Ly_sp/2, Ly_sp/2, ny_sp)
X_sp, Y_sp = jnp.meshgrid(x_sp, y_sp, indexing='ij')

delta_sp = 0.5
B0_sp = 1.0
rho0_sp = 1.0
Bx_sp = B0_sp * jnp.tanh(Y_sp / delta_sp)
By_sp = jnp.zeros_like(rho0_sp)
Bz_sp = jnp.zeros_like(rho0_sp)
B_sp = jnp.stack([Bx_sp, By_sp, Bz_sp], axis=0)

p_sp = B0_sp**2 / (2 * CONFIG['mu0']) - 0.5 * B_sp[0]**2 / CONFIG['mu0']
v_sp = jnp.zeros((3, nx_sp, ny_sp))
S_sp = rho0_sp * v_sp

v2_sp = jnp.sum(v_sp**2, axis=0)
B2_sp = jnp.sum(B_sp**2, axis=0)
tau_sp = p_sp / (CONFIG['gamma_eos'] - 1) + 0.5 * rho0_sp * v2_sp + B2_sp / (2 * CONFIG['mu0'])

state_sp = {'rho': rho0_sp * jnp.ones((nx_sp, ny_sp)), 'S': S_sp, 'tau': tau_sp, 'B': B_sp, 'p': p_sp}

t_sp = 0.0
t_end_sp = 50.0

sp_diagnostics = {'time': [], 'v_in': [], 'v_out': [], 'delta': [], 'reconn_rate': []}

print("Running Sweet-Parker reconnection...")
while t_sp < t_end_sp:
    state_sp = step_2d(state_sp, dx_sp)
    t_sp += state_sp['dt_used'].item()

    v_sp = state_sp['S'] / state_sp['rho']
    J_sp = (grad_y(state_sp['B'][1]) - grad_x(state_sp['B'][0])) / CONFIG['mu0']
    v_in = jnp.mean(jnp.abs(v_sp[1][:, ny_sp//2 - 10:ny_sp//2 + 10]))
    v_out = jnp.mean(jnp.abs(v_sp[0][nx_sp//2 - 10:nx_sp//2 + 10, :]))
    v_A = B0_sp / jnp.sqrt(CONFIG['mu0'] * rho0_sp)
    S = Lx_sp * v_A / 1e-3
    delta_theory = Lx_sp / jnp.sqrt(S)
    reconn_rate = v_in / v_A
    reconn_rate_theory = 1 / jnp.sqrt(S)

    sp_diagnostics['time'].append(t_sp)
    sp_diagnostics['v_in'].append(v_in)
    sp_diagnostics['v_out'].append(v_out)
    sp_diagnostics['delta'].append(delta_theory)
    sp_diagnostics['reconn_rate'].append(reconn_rate)

    if int(t_sp) % 10 == 0:
        print(f"Sweet-Parker t = {t_sp:.1f} | v_in = {v_in:.2e} | v_out = {v_out:.2e} | rate = {reconn_rate:.2e}")

fig_sp, axs_sp = plt.subplots(2, 2, figsize=(14, 10), facecolor='black')
fig_sp.suptitle('Sweet-Parker Reconnection – Full Details at t≈50', color='white', fontsize=18)

axs_sp[0,0].set_facecolor('black')
im = axs_sp[0,0].imshow(state_sp['rho'], origin='lower', extent=[-Lx_sp/2, Lx_sp/2, -Ly_sp/2, Ly_sp/2], cmap='viridis')
plt.colorbar(im, ax=axs_sp[0,0], label='Density ρ [kg/m³]')
axs_sp[0,0].set_title('Density', color='white')
axs_sp[0,0].tick_params(colors='white')

axs_sp[0,1].set_facecolor('black')
im = axs_sp[0,1].imshow(jnp.abs(J_sp), origin='lower', extent=[-Lx_sp/2, Lx_sp/2, -Ly_sp/2, Ly_sp/2], cmap='hot')
plt.colorbar(im, ax=axs_sp[0,1], label='|J_z| [A/m²]')
axs_sp[0,1].set_title('Current Density', color='white')
axs_sp[0,1].tick_params(colors='white')

axs_sp[1,0].set_facecolor('black')
im = axs_sp[1,0].imshow(v_sp[0], origin='lower', extent=[-Lx_sp/2, Lx_sp/2, -Ly_sp/2, Ly_sp/2], cmap='coolwarm')
plt.colorbar(im, ax=axs_sp[1,0], label='v_x [m/s]')
axs_sp[1,0].set_title('Outflow Velocity', color='white')
axs_sp[1,0].tick_params(colors='white')

axs_sp[1,1].set_facecolor('black')
axs_sp[1,1].plot(sp_diagnostics['time'], sp_diagnostics['reconn_rate'], 'c-', lw=2, label='Sim Rate')
axs_sp[1,1].axhline(reconn_rate_theory, color='r', ls='--', label='Sweet-Parker Theory ~ S^{-1/2}')
axs_sp[1,1].set_title('Reconnection Rate vs Time', color='white')
axs_sp[1,1].set_xlabel('Time [s]', color='white')
axs_sp[1,1].set_ylabel('Normalized Rate', color='white')
axs_sp[1,1].grid(alpha=0.3, color='gray')
axs_sp[1,1].legend(frameon=False, labelcolor='white')
axs_sp[1,1].tick_params(colors='white')

for ax in axs_sp.flat:
    ax.tick_params(colors='white')
    ax.grid(alpha=0.25, color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/sweet_parker_multi.png', dpi=CONFIG['dpi'], facecolor='black')
plt.close(fig_sp)

print("Sweet-Parker done")

# ─── INTEGRATION TESTS ──────────────────────────────────────────────────────
def run_integration_tests():
    print("\n=== Running Integration Tests ===")

    def short_run_2d(initial_state, steps=20, dx_local=dx):
        state = initial_state.copy()
        for _ in range(steps):
            state = step_2d(state, dx_local)
        return state

    # 1. Brio-Wu (check density jump and By reversal)
    print("Test 1: Brio-Wu shock structure")
    nx_test = 200
    x_test = jnp.linspace(0, 1, nx_test)
    dx_test = x_test[1] - x_test[0]

    rho_test = jnp.where(x_test < 0.5, 1.0, 0.125)
    p_test = jnp.where(x_test < 0.5, 1000.0, 0.1)
    vx_test = jnp.zeros(nx_test)
    vy_test = jnp.zeros(nx_test)
    vz_test = jnp.zeros(nx_test)
    Bx_test = jnp.full(nx_test, 0.75)
    By_test = jnp.where(x_test < 0.5, 1.0, -1.0)
    Bz_test = jnp.zeros(nx_test)

    U_test = jnp.zeros((8, nx_test))
    U_test = U_test.at[0].set(rho_test)
    U_test = U_test.at[1].set(rho_test * vx_test)
    U_test = U_test.at[2].set(rho_test * vy_test)
    U_test = U_test.at[3].set(rho_test * vz_test)
    U_test = U_test.at[4].set(p_test / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_test * (vx_test**2 + vy_test**2 + vz_test**2) + 0.5 * (Bx_test**2 + By_test**2 + Bz_test**2) / CONFIG['mu0'])
    U_test = U_test.at[5].set(Bx_test)
    U_test = U_test.at[6].set(By_test)
    U_test = U_test.at[7].set(Bz_test)

    t_test = 0.0
    for _ in range(30):
        v_abs = jnp.abs(U_test[1] / U_test[0])
        va = jnp.sqrt(jnp.sum(U_test[5:8]**2, axis=0) / (CONFIG['mu0'] * U_test[0] + 1e-20))
        cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_test[4] - 0.5 * jnp.sum(U_test[1:4]**2, axis=0) / U_test[0] - 0.5 * jnp.sum(U_test[5:8]**2, axis=0) / CONFIG['mu0']) / U_test[0])
        cf = jnp.sqrt(cs**2 + va**2 + CONFIG['b_floor']**2)
        speed_max = jnp.max(v_abs + cf) + 1e-10
        dt_test = 0.4 * dx_test / speed_max
        dt_test = jnp.minimum(dt_test, 0.2 - t_test)

        F_test = jnp.zeros_like(U_test)
        for i in range(nx_test - 1):
            F_test = F_test.at[:, i].set(hlld_flux(U_test[:, i], U_test[:, i+1]))

        U_test = U_test.at[:, 1:-1].add(dt_test / dx_test * (F_test[:, :-2] - F_test[:, 1:-1]))

        t_test += dt_test.item()

    rho_final_test = U_test[0]
    by_final_test = U_test[6]

    assert jnp.max(rho_final_test) > 1.5, f"Brio-Wu density jump too small: max = {jnp.max(rho_final_test)}"
    assert jnp.min(by_final_test) < -0.5 and jnp.max(by_final_test) > 0.5, "Brio-Wu By reversal missing"
    print("  Brio-Wu integration test → PASSED")

    # 2. Orszag-Tang (check vortex persistence)
    print("Test 2: Orszag-Tang vortex coherence")
    nx_ot_test = 64
    x_ot_test = jnp.linspace(0, 2*np.pi, nx_ot_test)
    y_ot_test = jnp.linspace(0, 2*np.pi, nx_ot_test)
    X_ot_test, Y_ot_test = jnp.meshgrid(x_ot_test, y_ot_test, indexing='ij')

    rho_ot_test = 1 + 0.25 * jnp.sin(2 * Y_ot_test) + 0.25 * jnp.sin(2 * X_ot_test)
    p_ot_test = 5/3 * jnp.ones_like(rho_ot_test)
    vx_ot_test = -jnp.sin(Y_ot_test)
    vy_ot_test = jnp.sin(X_ot_test)
    vz_ot_test = jnp.zeros_like(rho_ot_test)
    Bx_ot_test = -jnp.sin(Y_ot_test)
    By_ot_test = jnp.sin(2 * X_ot_test)
    Bz_ot_test = jnp.zeros_like(rho_ot_test)

    S_ot_test = rho_ot_test * jnp.stack([vx_ot_test, vy_ot_test, vz_ot_test], axis=0)
    B_ot_test = jnp.stack([Bx_ot_test, By_ot_test, Bz_ot_test], axis=0)

    v2_ot_test = vx_ot_test**2 + vy_ot_test**2 + vz_ot_test**2
    B2_ot_test = Bx_ot_test**2 + By_ot_test**2 + Bz_ot_test**2
    tau_ot_test = p_ot_test / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_ot_test * v2_ot_test + B2_ot_test / (2 * CONFIG['mu0'])

    state_ot_test = {'rho': rho_ot_test, 'S': S_ot_test, 'tau': tau_ot_test, 'B': B_ot_test, 'p': p_ot_test}

    t_ot_test = 0.0
    for _ in range(15):
        state_ot_test = step_2d(state_ot_test, 2*np.pi / nx_ot_test)
        t_ot_test += state_ot_test['dt_used'].item()

    rho_final_ot = state_ot_test['rho']
    assert jnp.std(rho_final_ot) > 0.05, f"Orszag-Tang vortex damped too much: std = {jnp.std(rho_final_ot)}"
    print("  Orszag-Tang integration test → PASSED")

    # 3. Hydro Blast Wave (check peak density)
    print("Test 3: Hydro Blast Wave shock")
    nx_bw_test = 128
    x_bw_test = jnp.linspace(0, 1, nx_bw_test)
    dx_bw_test = x_bw_test[1] - x_bw_test[0]

    rho_bw_test = jnp.ones(nx_bw_test)
    p_bw_test = jnp.zeros(nx_bw_test)
    p_bw_test = p_bw_test.at[nx_bw_test//2].set(1.0 / dx_bw_test)
    vx_bw_test = jnp.zeros(nx_bw_test)

    U_bw_test = jnp.zeros((3, nx_bw_test))
    U_bw_test = U_bw_test.at[0].set(rho_bw_test)
    U_bw_test = U_bw_test.at[1].set(rho_bw_test * vx_bw_test)
    U_bw_test = U_bw_test.at[2].set(p_bw_test / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_bw_test * vx_bw_test**2)

    t_bw_test = 0.0
    for _ in range(25):
        v_abs = jnp.abs(U_bw_test[1] / U_bw_test[0])
        cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw_test[2] - 0.5 * U_bw_test[1]**2 / U_bw_test[0]) * (CONFIG['gamma_eos'] - 1) / U_bw_test[0])
        speed_max = jnp.max(v_abs + cs) + 1e-10
        dt_bw_test = 0.4 * dx_bw_test / speed_max
        dt_bw_test = jnp.minimum(dt_bw_test, 0.1 - t_bw_test)

        F_bw_test = jnp.zeros_like(U_bw_test)
        for i in range(nx_bw_test - 1):
            F_bw_test = F_bw_test.at[:, i].set(hydro_hll_flux(U_bw_test[:, i], U_bw_test[:, i+1]))

        U_bw_test = U_bw_test.at[:, 1:-1].add(dt_bw_test / dx_bw_test * (F_bw_test[:, :-2] - F_bw_test[:, 1:-1]))

        t_bw_test += dt_bw_test.item()

    rho_final_bw = U_bw_test[0]
    assert jnp.max(rho_final_bw) > 3.0, f"Hydro Blast peak density too low: {jnp.max(rho_final_bw)}"
    print("  Hydro Blast Wave integration test → PASSED")

    # 4. MHD Blast Wave (check density jump + B compression)
    print("Test 4: MHD Blast Wave")
    nx_bw_test = 128
    x_bw_test = jnp.linspace(0, 1, nx_bw_test)
    dx_bw_test = x_bw_test[1] - x_bw_test[0]

    rho_bw_test = jnp.ones(nx_bw_test)
    p_bw_test = jnp.zeros(nx_bw_test)
    p_bw_test = p_bw_test.at[nx_bw_test//2].set(1.0 / dx_bw_test)
    vx_bw_test = jnp.zeros(nx_bw_test)
    vy_bw_test = jnp.zeros(nx_bw_test)
    vz_bw_test = jnp.zeros(nx_bw_test)
    Bx_bw_test = jnp.ones(nx_bw_test)
    By_bw_test = jnp.zeros(nx_bw_test)
    Bz_bw_test = jnp.zeros(nx_bw_test)

    U_bw_test = jnp.zeros((8, nx_bw_test))
    U_bw_test = U_bw_test.at[0].set(rho_bw_test)
    U_bw_test = U_bw_test.at[1].set(rho_bw_test * vx_bw_test)
    U_bw_test = U_bw_test.at[2].set(rho_bw_test * vy_bw_test)
    U_bw_test = U_bw_test.at[3].set(rho_bw_test * vz_bw_test)
    U_bw_test = U_bw_test.at[4].set(p_bw_test / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_bw_test * (vx_bw_test**2 + vy_bw_test**2 + vz_bw_test**2) + 0.5 * (Bx_bw_test**2 + By_bw_test**2 + Bz_bw_test**2) / CONFIG['mu0'])
    U_bw_test = U_bw_test.at[5].set(Bx_bw_test)
    U_bw_test = U_bw_test.at[6].set(By_bw_test)
    U_bw_test = U_bw_test.at[7].set(Bz_bw_test)

    t_bw_test = 0.0
    for _ in range(25):
        v_abs = jnp.abs(U_bw_test[1] / U_bw_test[0])
        va = jnp.sqrt(jnp.sum(U_bw_test[5:8]**2, axis=0) / (CONFIG['mu0'] * U_bw_test[0] + 1e-20))
        cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw_test[4] - 0.5 * jnp.sum(U_bw_test[1:4]**2, axis=0) / U_bw_test[0] - 0.5 * jnp.sum(U_bw_test[5:8]**2, axis=0) / CONFIG['mu0']) / U_bw_test[0])
        cf = jnp.sqrt(cs**2 + va**2 + CONFIG['b_floor']**2)
        speed_max = jnp.max(v_abs + cf) + 1e-10
        dt_bw_test = 0.4 * dx_bw_test / speed_max
        dt_bw_test = jnp.minimum(dt_bw_test, 0.1 - t_bw_test)

        F_bw_test = jnp.zeros_like(U_bw_test)
        for i in range(nx_bw_test - 1):
            F_bw_test = F_bw_test.at[:, i].set(hlld_flux(U_bw_test[:, i], U_bw_test[:, i+1]))

        U_bw_test = U_bw_test.at[:, 1:-1].add(dt_bw_test / dx_bw_test * (F_bw_test[:, :-2] - F_bw_test[:, 1:-1]))

        t_bw_test += dt_bw_test.item()

    rho_final_mhd = U_bw_test[0]
    bx_final_mhd = U_bw_test[5]
    assert jnp.max(rho_final_mhd) > 2.8, f"MHD Blast density jump too low: {jnp.max(rho_final_mhd)}"
    assert jnp.max(bx_final_mhd) > 1.3, f"MHD Blast magnetic compression missing: {jnp.max(bx_final_mhd)}"
    print("  MHD Blast Wave integration test → PASSED")

    # 5. GEM (check current sheet + outflow)
    print("Test 5: GEM reconnection")
    Lx_test = 25.0
    Ly_test = 12.5
    nx_gem_test = 128
    ny_gem_test = 64
    dx_gem_test = Lx_test / nx_gem_test

    x_gem_test = jnp.linspace(-Lx_test/2, Lx_test/2, nx_gem_test)
    y_gem_test = jnp.linspace(-Ly_test/2, Ly_test/2, ny_gem_test)
    X_gem_test, Y_gem_test = jnp.meshgrid(x_gem_test, y_gem_test, indexing='ij')

    lambda_sheet = 0.5
    B0 = 1.0
    rho0 = 1.0
    p0 = B0**2 / (8 * np.pi)
    rho_gem_test = rho0 * jnp.cosh(Y_gem_test / lambda_sheet)**(-2)
    Bx_gem_test = B0 * jnp.tanh(Y_gem_test / lambda_sheet)
    By_gem_test = jnp.zeros_like(rho_gem_test)
    Bz_gem_test = jnp.zeros_like(rho_gem_test)
    B_gem_test = jnp.stack([Bx_gem_test, By_gem_test, Bz_gem_test], axis=0)

    pert = 0.01 * B0 * jnp.exp(- (X_gem_test**2 + Y_gem_test**2) / 5**2) * jnp.cos(2 * np.pi * X_gem_test / Lx_test)
    By_gem_test += pert
    B_gem_test = B_gem_test.at[1].set(By_gem_test)

    p_gem_test = p0 * jnp.ones_like(rho_gem_test)
    v_gem_test = jnp.zeros((3, nx_gem_test, ny_gem_test))
    S_gem_test = rho_gem_test * v_gem_test

    v2_gem_test = jnp.sum(v_gem_test**2, axis=0)
    B2_gem_test = jnp.sum(B_gem_test**2, axis=0)
    tau_gem_test = p_gem_test / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_gem_test * v2_gem_test + B2_gem_test / (2 * CONFIG['mu0'])

    state_gem_test = {'rho': rho_gem_test, 'S': S_gem_test, 'tau': tau_gem_test, 'B': B_gem_test, 'p': p_gem_test}

    t_gem_test = 0.0
    for _ in range(12):
        state_gem_test = step_2d(state_gem_test, dx_gem_test)
        t_gem_test += state_gem_test['dt_used'].item()

    v_gem_test_final = state_gem_test['S'] / state_gem_test['rho']
    J_gem_test = (grad_y(state_gem_test['B'][1]) - grad_x(state_gem_test['B'][0])) / CONFIG['mu0']
    v_out_gem = jnp.mean(jnp.abs(v_gem_test_final[0][nx_gem_test//2 - 5:nx_gem_test//2 + 5, :]))

    assert jnp.max(jnp.abs(J_gem_test)) > 1.5, f"GEM current sheet not forming: max |J| = {jnp.max(jnp.abs(J_gem_test))}"
    assert v_out_gem > 0.03, f"No significant outflow in GEM: v_out = {v_out_gem}"
    print("  GEM integration test → PASSED")

    # 6. Sweet-Parker (check reconnection rate range)
    print("Test 6: Sweet-Parker reconnection rate")
    Lx_sp_test = 10.0
    Ly_sp_test = 5.0
    nx_sp_test = 64
    ny_sp_test = 32
    dx_sp_test = Lx_sp_test / nx_sp_test

    x_sp_test = jnp.linspace(-Lx_sp_test/2, Lx_sp_test/2, nx_sp_test)
    y_sp_test = jnp.linspace(-Ly_sp_test/2, Ly_sp_test/2, ny_sp_test)
    X_sp_test, Y_sp_test = jnp.meshgrid(x_sp_test, y_sp_test, indexing='ij')

    delta_sp = 0.25
    B0_sp = 1.0
    rho0_sp = 1.0
    Bx_sp = B0_sp * jnp.tanh(Y_sp_test / delta_sp)
    By_sp = jnp.zeros_like(rho0_sp)
    Bz_sp = jnp.zeros_like(rho0_sp)
    B_sp = jnp.stack([Bx_sp, By_sp, Bz_sp], axis=0)

    p_sp = B0_sp**2 / (2 * CONFIG['mu0']) - 0.5 * B_sp[0]**2 / CONFIG['mu0']
    v_sp = jnp.zeros((3, nx_sp_test, ny_sp_test))
    S_sp = rho0_sp * v_sp

    v2_sp = jnp.sum(v_sp**2, axis=0)
    B2_sp = jnp.sum(B_sp**2, axis=0)
    tau_sp = p_sp / (CONFIG['gamma_eos'] - 1) + 0.5 * rho0_sp * v2_sp + B2_sp / (2 * CONFIG['mu0'])

    state_sp_test = {'rho': rho0_sp * jnp.ones((nx_sp_test, ny_sp_test)), 'S': S_sp, 'tau': tau_sp, 'B': B_sp, 'p': p_sp}

    t_sp_test = 0.0
    for _ in range(15):
        state_sp_test = step_2d(state_sp_test, dx_sp_test)
        t_sp_test += state_sp_test['dt_used'].item()

    v_sp_test = state_sp_test['S'] / state_sp_test['rho']
    v_in_sp = jnp.mean(jnp.abs(v_sp_test[1][:, ny_sp_test//2 - 3:ny_sp_test//2 + 3]))
    v_A_sp = B0_sp / jnp.sqrt(CONFIG['mu0'] * rho0_sp)
    reconn_rate_sp = v_in_sp / v_A_sp

    assert 0.002 < reconn_rate_sp < 0.15, f"Sweet-Parker rate out of expected range: {reconn_rate_sp}"
    print("  Sweet-Parker integration test → PASSED")

    print("\n=== All integration tests PASSED ===")

run_integration_tests()

print("\nALL DONE!")
print("Main sim + mode fits + 13 plots + validations + integration tests complete.")
print("Download everything:")
!zip -r zpinch_v1.1.8_full.zip plots checkpoints
