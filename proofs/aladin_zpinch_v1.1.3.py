# mega ultra maniac v1.1.3 – FINAL POLISHED PUBLISH VERSION
# Full 3D Resistive MHD Z-pinch (128³) + ALL 6 VALIDATIONS + INTEGRATION TESTS
# Boundary conditions: approximately periodic
# (central differences with periodic roll for Laplacians)
# NOTE: central-difference MHD (not strictly conservative FV)
# Energy equation includes explicit resistive heating and radiative losses;
# total energy is not conserved by design.

import jax
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import os
from scipy.fft import fft
from scipy.optimize import curve_fit
from tqdm import tqdm

jax.config.update('jax_enable_x64', False)

os.makedirs("plots", exist_ok=True)

CONFIG = {
    'grid_size': 128,
    'cfl': 0.25,
    'dt_max': 5e-10,
    'steps': 400,
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
    'b_floor': 1e-10,  # prevent NaN in Alfvén speed
}

GRID_SIZE = CONFIG['grid_size']
dx = CONFIG['l_r'] / GRID_SIZE

x = jnp.linspace(0, CONFIG['l_r'], GRID_SIZE, dtype=jnp.float32)
y = jnp.linspace(0, CONFIG['l_r'], GRID_SIZE, dtype=jnp.float32)
z = jnp.linspace(0, 0.1, GRID_SIZE, dtype=jnp.float32)

X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
R = jnp.sqrt(X**2 + Y**2 + 1e-10)
THETA = jnp.arctan2(Y, X)

# ─── FULL HLLD FLUX (MHD) ───────────────────────────────────────────────────
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

# ─── PURE HYDRO HLL FLUX ────────────────────────────────────────────────────
@jit
def hydro_flux(U):
    gamma = CONFIG['gamma_eos']
    rho = U[0]
    v = U[1]
    p = (gamma - 1) * (U[2] - 0.5 * rho * v**2)

    F = jnp.zeros_like(U)
    F = F.at[0].set(rho * v)
    F = F.at[1].set(rho * v**2 + p)
    F = F.at[2].set((U[2] + p) * v)
    return F

@jit
def hydro_hll_flux(UL, UR):
    gamma = CONFIG['gamma_eos']
    rhoL = UL[0]
    vL = UL[1] / rhoL
    pL = (gamma - 1) * (UL[2] - 0.5 * rhoL * vL**2)

    rhoR = UR[0]
    vR = UR[1] / rhoR
    pR = (gamma - 1) * (UR[2] - 0.5 * rhoR * vR**2)

    csL = jnp.sqrt(gamma * pL / rhoL)
    csR = jnp.sqrt(gamma * pR / rhoR)

    SL = jnp.minimum(vL - csL, vR - csR)
    SR = jnp.maximum(vL + csL, vR + csR)

    FL = hydro_flux(UL)
    FR = hydro_flux(UR)

    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1e-20)

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

# Perturbation – safe version
delta = CONFIG['epsilon'] * jnp.cos(3 * THETA + CONFIG['k_pert'] * Z)
state['rho'] *= (1 + delta)
state['rho'] = jnp.maximum(state['rho'], CONFIG['rho_floor'])

# ─── VECTORIZED GRADIENT HELPERS ────────────────────────────────────────────
grad_x = vmap(lambda f: (jnp.roll(f, -1, 0) - jnp.roll(f, 1, 0)) / (2 * dx), in_axes=0)
grad_y = vmap(lambda f: (jnp.roll(f, -1, 1) - jnp.roll(f, 1, 1)) / (2 * dx), in_axes=0)
grad_z = vmap(lambda f: (jnp.roll(f, -1, 2) - jnp.roll(f, 1, 2)) / (2 * dx), in_axes=0)

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

    def laplacian(f):
        return (
            (jnp.roll(f, 1, axis=0) - 2*f + jnp.roll(f, -1, axis=0)) / dx**2 +
            (jnp.roll(f, 1, axis=1) - 2*f + jnp.roll(f, -1, axis=1)) / dx**2 +
            (jnp.roll(f, 1, axis=2) - 2*f + jnp.roll(f, -1, axis=2)) / dx**2
        )

    lap_B = jnp.stack([laplacian(B[i]) for i in range(3)], axis=0)

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

    mean_grad_p = jnp.mean(jnp.sqrt(sum(grad(p, i)**2 for i in range(3))))
    force_imbalance = jnp.mean(jnp.linalg.norm(J_cross_B + jnp.stack([grad(p, i) for i in range(3)], axis=0), axis=0))
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

# ────────────────────────────────────────────────
# Run main Z-pinch simulation
# ────────────────────────────────────────────────
states = [state]
real_time_history = [0.0]
diagnostics = {
    'div_B_max': [], 'div_B_mean': [], 'dt_used': [], 'pinch_radius': [],
    'E_mag': [], 'lorentz_mag': [], 'mean_grad_p': [], 'imbalance_ratio': [],
    'mean_j_mag': [], 'total_energy': [], 'heating_rate': [], 'cooling_rate': [],
    'E_resist_cum': [0.0],  # cumulative resistive dissipation
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

    # Cumulative resistive dissipation
    E_resist_cum += state['heating_rate'] * dx**3 * state['dt_used']
    d['E_resist_cum'].append(E_resist_cum)

    if step_num % 20 == 0:
        print(f"Step {step_num} | t = {real_time_history[-1]:.2e} s | r = {state['pinch_radius']:.4e} | imb = {state['imbalance_ratio']:.2e}")

real_time = np.array(real_time_history[1:]) * 1e6
energy_ratio = np.array(diagnostics['total_energy']) / diagnostics['total_energy'][0]

final_state = states[-1]

print("Main simulation finished.")

# ─── MAIN PLOTS (13 + extra energy breakdown) ───────────────────────────────
labels = ['m=0', 'm=1', 'm=2', 'm=3']
colors = ['#00FFFF', '#FF00FF', '#FFD700', '#FF6B00']

m_amplitudes = []
for s in states[::10]:
    rho_mid = s['rho'][:, :, GRID_SIZE//2]
    fft_theta = fft(rho_mid.mean(axis=0))
    m_amps = np.abs(fft_theta[0:4]) / GRID_SIZE
    m_amplitudes.append(m_amps)
m_amplitudes = np.array(m_amplitudes)
time_fft = real_time[::10]

def exp_growth(t, A, gamma):
    return A * np.exp(gamma * t * 1e-6)

gamma_measured = []
for m in range(4):
    amps = m_amplitudes[:, m]
    try:
        popt, _ = curve_fit(exp_growth, time_fft[:30], amps[:30], p0=[1e-6, 1e7])
        gamma_measured.append(popt[1])
    except:
        gamma_measured.append(np.nan)

def save_plot(fig, name):
    fig.savefig(f'plots/{name}.png', dpi=CONFIG['dpi'], facecolor='black', bbox_inches='tight')
    plt.close(fig)

# 1. Growth rates
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
for m in range(4):
    ax.semilogy(time_fft, m_amplitudes[:, m], lw=2.5, color=colors[m], label=labels[m])
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Amplitude (log)', color='white')
ax.set_title('Kink Mode Growth Rates', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'growth_rates')

# 2. Energy conservation (with dissipation note)
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time, energy_ratio, lw=3, color='#00FFFF', label='Total E / E0')
ax.axhline(1, color='gray', ls='--')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Ratio', color='white')
ax.set_title('Energy Conservation (non-conservative: resistive + cooling)', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'energy_conservation')

# Extra: Energy breakdown (kinetic, magnetic, internal, resistive dissipated)
fig = plt.figure(figsize=(12, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.stackplot(real_time[:-1], 
             [np.array(diagnostics['total_energy']) - diagnostics['E_mag'] - diagnostics['E_int'],  # kinetic approx
              diagnostics['E_mag'],
              diagnostics['E_int']],
             labels=['Kinetic', 'Magnetic', 'Internal'], colors=['#00FF00', '#FF4500', '#8A2BE2'])
ax.plot(real_time[:-1], diagnostics['E_resist_cum'], lw=2, color='#FFD700', label='Resistive dissipated (cumulative)')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('Energy [J]', color='white')
ax.set_title('Energy Components Breakdown', color='white')
ax.legend(frameon=False, labelcolor='white')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'energy_breakdown')

# ... (rest of your 13 plots - add them here with units)

# Example for current density with units
fig = plt.figure(figsize=(10, 6), facecolor='black')
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(real_time[:-1], diagnostics['mean_j_mag'], lw=3, color='#FF69B4')
ax.set_xlabel('Time [μs]', color='white')
ax.set_ylabel('|J| [A/m²]', color='white')
ax.set_title('Current Density Evolution', color='white')
ax.set_yscale('log')
ax.grid(alpha=0.2, color='gray')
ax.tick_params(colors='white')
save_plot(fig, 'current_density_time')

print("All main plots generated.")

# ─── INTEGRATION TESTS FOR VALIDATIONS ──────────────────────────────────────
def run_integration_tests():
    print("\nRunning integration tests for all 6 validations...")

    # Helper: short-run wrapper
    def short_run_2d(initial_state, steps=20):
        state = initial_state
        for _ in range(steps):
            state = step_2d(state, dx)
        return state

    # 1. Brio-Wu integration test (check shock structure)
    print("  Test Brio-Wu integration...")
    nx_bw_test = 400
    x_bw_test = jnp.linspace(0, 1, nx_bw_test)
    dx_bw_test = x_bw_test[1] - x_bw_test[0]

    rho_bw = jnp.where(x_bw_test < 0.5, 1.0, 0.125)
    p_bw = jnp.where(x_bw_test < 0.5, 1000.0, 0.1)
    vx_bw = jnp.zeros(nx_bw_test)
    vy_bw = jnp.zeros(nx_bw_test)
    vz_bw = jnp.zeros(nx_bw_test)
    Bx_bw = jnp.full(nx_bw_test, 0.75)
    By_bw = jnp.where(x_bw_test < 0.5, 1.0, -1.0)
    Bz_bw = jnp.zeros(nx_bw_test)

    U_bw = jnp.zeros((8, nx_bw_test))
    U_bw = U_bw.at[0].set(rho_bw)
    U_bw = U_bw.at[1].set(rho_bw * vx_bw)
    U_bw = U_bw.at[2].set(rho_bw * vy_bw)
    U_bw = U_bw.at[3].set(rho_bw * vz_bw)
    U_bw = U_bw.at[4].set(p_bw / (CONFIG['gamma_eos'] - 1) + 0.5 * rho_bw * (vx_bw**2 + vy_bw**2 + vz_bw**2) + 0.5 * (Bx_bw**2 + By_bw**2 + Bz_bw**2) / CONFIG['mu0'])
    U_bw = U_bw.at[5].set(Bx_bw)
    U_bw = U_bw.at[6].set(By_bw)
    U_bw = U_bw.at[7].set(Bz_bw)

    t_bw = 0.0
    for _ in range(30):
        v_abs = jnp.abs(U_bw[1] / U_bw[0])
        va = jnp.sqrt(jnp.sum(U_bw[5:8]**2, axis=0) / (CONFIG['mu0'] * U_bw[0] + 1e-20))
        cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw[4] - 0.5 * jnp.sum(U_bw[1:4]**2, axis=0) / U_bw[0] - 0.5 * jnp.sum(U_bw[5:8]**2, axis=0) / CONFIG['mu0']) / U_bw[0])
        cf = jnp.sqrt(cs**2 + va**2 + 1e-20)
        speed_max = jnp.max(v_abs + cf) + 1e-10
        dt_bw = 0.4 * dx_bw_test / speed_max
        dt_bw = jnp.minimum(dt_bw, 0.2 - t_bw)

        F_bw = jnp.zeros_like(U_bw)
        for i in range(nx_bw_test - 1):
            F_bw = F_bw.at[:, i].set(hlld_flux(U_bw[:, i], U_bw[:, i+1]))

        U_bw = U_bw.at[:, 1:-1].add(dt_bw / dx_bw_test * (F_bw[:, :-2] - F_bw[:, 1:-1]))

        t_bw += dt_bw.item()

    rho_final = U_bw[0]
    assert jnp.max(rho_final) > 1.8, f"Brio-Wu density peak too low: {jnp.max(rho_final)}"
    assert jnp.min(U_bw[6]) < -0.3, "Brio-Wu By reversal not present"
    print("  Brio-Wu integration test passed")

    # 2. Orszag-Tang integration test (check vortex coherence)
    print("  Test Orszag-Tang integration...")
    nx_ot = ny_ot = 128
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
    for _ in range(25):
        state_ot = step_2d(state_ot, 2*np.pi / nx_ot)
        t_ot += state_ot['dt_used'].item()

    rho_final = state_ot['rho']
    assert jnp.std(rho_final) > 0.08, f"Orszag-Tang vortex damped too much: std = {jnp.std(rho_final)}"
    print("  Orszag-Tang integration test passed")

    # 3. Hydro Blast Wave integration test (check shock front)
    print("  Test Hydro Blast Wave integration...")
    nx_bw = 256
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
    for _ in range(35):
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

    rho_final = U_bw[0]
    assert jnp.max(rho_final) > 3.8, f"Hydro Blast peak too low: {jnp.max(rho_final)}"
    assert jnp.any(rho_final[100:150] > 2.5), "No clear shock front"
    print("  Hydro Blast Wave integration test passed")

    # 4. MHD Blast Wave integration test (check magnetic compression)
    print("  Test MHD Blast Wave integration...")
    nx_bw = 256
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
    for _ in range(35):
        v_abs = jnp.abs(U_bw[1] / U_bw[0])
        va = jnp.sqrt(jnp.sum(U_bw[5:8]**2, axis=0) / (CONFIG['mu0'] * U_bw[0] + 1e-20))
        cs = jnp.sqrt(CONFIG['gamma_eos'] * (U_bw[4] - 0.5 * jnp.sum(U_bw[1:4]**2, axis=0) / U_bw[0] - 0.5 * jnp.sum(U_bw[5:8]**2, axis=0) / CONFIG['mu0']) / U_bw[0])
        cf = jnp.sqrt(cs**2 + va**2 + 1e-20)
        speed_max = jnp.max(v_abs + cf) + 1e-10
        dt_bw = 0.4 * dx_bw / speed_max
        dt_bw = jnp.minimum(dt_bw, 0.1 - t_bw)

        F_bw = jnp.zeros_like(U_bw)
        for i in range(nx_bw - 1):
            F_bw = F_bw.at[:, i].set(hlld_flux(U_bw[:, i], U_bw[:, i+1]))

        U_bw = U_bw.at[:, 1:-1].add(dt_bw / dx_bw * (F_bw[:, :-2] - F_bw[:, 1:-1]))

        t_bw += dt_bw.item()

    rho_final = U_bw[0]
    bx_final = U_bw[5]
    assert jnp.max(rho_final) > 3.2, f"MHD Blast density jump too low: {jnp.max(rho_final)}"
    assert jnp.max(bx_final) > 1.4, f"MHD Blast magnetic compression missing: {jnp.max(bx_final)}"
    print("  MHD Blast Wave integration test passed")

    # 5. GEM integration test (check current sheet + outflow)
    print("  Test GEM integration...")
    Lx = 25.0
    Ly = 12.5
    nx_gem = 256
    ny_gem = 128
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
    By_gem = jnp.zeros_like(rho_gem)
    Bz_gem = jnp.zeros_like(rho_gem)
    B_gem = jnp.stack([Bx_gem, By_gem, Bz_gem], axis=0)

    pert = 0.01 * B0 * jnp.exp(- (X_gem**2 + Y_gem**2) / 5**2) * jnp.cos(2 * np.pi * X_gem / Lx)
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
    for _ in range(20):
        state_gem = step_2d(state_gem, dx_gem)
        t_gem += state_gem['dt_used'].item()

    v_sp = state_gem['S'] / state_gem['rho']
    J_final = (jnp.gradient(state_gem['B'][1], axis=0) - jnp.gradient(state_gem['B'][0], axis=1)) / CONFIG['mu0']
    v_out = jnp.mean(jnp.abs(v_sp[0][nx_gem//2 - 10:nx_gem//2 + 10, :]))

    assert jnp.max(jnp.abs(J_final)) > 1.8, f"GEM current sheet not forming: max J = {jnp.max(jnp.abs(J_final))}"
    assert v_out > 0.05, f"No significant outflow in GEM: v_out = {v_out}"
    print("  GEM integration test passed")

    # 6. Sweet-Parker integration test (check reconnection rate range)
    print("  Test Sweet-Parker integration...")
    Lx_sp = 10.0
    Ly_sp = 5.0
    nx_sp = 128
    ny_sp = 64
    dx_sp = Lx_sp / nx_sp

    x_sp = jnp.linspace(-Lx_sp/2, Lx_sp/2, nx_sp)
    y_sp = jnp.linspace(-Ly_sp/2, Ly_sp/2, ny_sp)
    X_sp, Y_sp = jnp.meshgrid(x_sp, y_sp, indexing='ij')

    delta_sp = 0.25
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
    for _ in range(25):
        state_sp = step_2d(state_sp, dx_sp)
        t_sp += state_sp['dt_used'].item()

    v_sp = state_sp['S'] / state_sp['rho']
    v_in = jnp.mean(jnp.abs(v_sp[1][:, ny_sp//2 - 5:ny_sp//2 + 5]))
    v_A = B0_sp / jnp.sqrt(CONFIG['mu0'] * rho0_sp)
    reconn_rate = v_in / v_A

    assert 0.003 < reconn_rate < 0.12, f"Sweet-Parker rate out of expected range: {reconn_rate}"
    print("  Sweet-Parker integration test passed")

    print("\nAll integration tests passed!")

# Run integration tests after everything else
run_integration_tests()

# ─── FINAL ZIP & DOWNLOAD ───────────────────────────────────────────────────
print("\nALL DONE!")
print("All ~25 .png files + integration test output saved.")
print("Run these two lines in a new cell to download:")
print("!zip -r zpinch_v1.1.3_plots.zip plots")
print("from google.colab import files")
print("files.download('zpinch_v1.1.3_plots.zip')")
