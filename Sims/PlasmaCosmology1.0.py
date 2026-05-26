import cupy as cp
import numpy as np

print("🌌 ALADIN Plasma Cosmology v1.0 — FULL SSP-RK3 + 3D HLLD (Fixed)")

# ====================== PARAMETERS ======================
N = 64
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
gamma = 5.0 / 3.0
CFL = 0.35
steps = 500
rho_floor = 1e-4
p_floor = 1e-6
internal_E_floor = 1e-6
v_max_cap = 8.0
v_phi_factor = 0.055
c_h = 8.0
kappa = 25.0

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== FIELDS ======================
rho = cp.maximum(cp.ones((N, N, N), dtype=cp.float32) * 1e-3, rho_floor)
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4

# Proper B field initialization
Bx = cp.ones((N, N, N), dtype=cp.float32) * 0.3
By = cp.ones((N, N, N), dtype=cp.float32) * 0.3
Bz = cp.ones((N, N, N), dtype=cp.float32) * 0.3
psi = cp.zeros((N, N, N), dtype=cp.float32)

r_cyl = cp.sqrt(X**2 + Y**2)
rho *= cp.exp(-r_cyl / 8.0) * cp.exp(-Z**2 / 2.25)
rho = cp.maximum(rho, rho_floor)

r3d = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
M_dm = nfw_enclosed_mass(r3d)
g_r = -G * M_dm / r3d**2
g_x = g_r * X / r3d
g_y = g_r * Y / r3d
g_z = g_r * Z / r3d

v_phi = v_phi_factor * cp.sqrt(cp.maximum(r_cyl * cp.abs(g_r), 0.0))
vx = -v_phi * (Y / (r_cyl + 1e-8))
vy =  v_phi * (X / (r_cyl + 1e-8))
vz = cp.zeros_like(vx)

mx = rho * vx
my = rho * vy
mz = rho * vz

mass0 = float(cp.sum(rho))
E0 = float(cp.sum(E_total))
Lz0 = float(cp.sum(rho * (X*vy - Y*vx)))

def compute_divB():
    div = cp.zeros_like(rho)
    div += (cp.roll(Bx, -1, axis=0) - cp.roll(Bx, 1, axis=0)) / (2 * dx)
    div += (cp.roll(By, -1, axis=1) - cp.roll(By, 1, axis=1)) / (2 * dx)
    div += (cp.roll(Bz, -1, axis=2) - cp.roll(Bz, 1, axis=2)) / (2 * dx)
    return div

# ====================== MUSCL ======================
def minmod(a, b):
    return cp.sign(a) * cp.minimum(cp.abs(a), cp.abs(b)) * (cp.sign(a) == cp.sign(b))

def reconstruct_plm(q, axis=0):
    dq_right = cp.roll(q, -1, axis=axis) - q
    dq_left = q - cp.roll(q, 1, axis=axis)
    slope = minmod(dq_left, dq_right)
    qL = q + 0.5 * slope
    qR = cp.roll(q, -1, axis=axis) - 0.5 * cp.roll(slope, -1, axis=axis)
    return qL, qR

# ====================== HLLD ======================
def hlld_flux_1d(rhoL, rhoR, uL, uR, vL, vR, wL, wR, EL, ER, pL, pR,
                 BnL, BnR, Bt1L, Bt1R, Bt2L, Bt2R):
    cs2L = gamma * pL / rhoL
    cs2R = gamma * pR / rhoR
    ca2L = (BnL**2 + Bt1L**2 + Bt2L**2) / rhoL
    ca2R = (BnR**2 + Bt1R**2 + Bt2R**2) / rhoR
    cfL = cp.sqrt(0.5 * (cs2L + ca2L + cp.sqrt((cs2L + ca2L)**2 - 4*cs2L*BnL**2/rhoL)))
    cfR = cp.sqrt(0.5 * (cs2R + ca2R + cp.sqrt((cs2R + ca2R)**2 - 4*cs2R*BnR**2/rhoR)))

    SL = cp.minimum(0.0, cp.minimum(uL - cfL, uR - cfR))
    SR = cp.maximum(0.0, cp.maximum(uL + cfL, uR + cfR))
    p_tot_L = pL + 0.5 * (BnL**2 + Bt1L**2 + Bt2L**2)
    p_tot_R = pR + 0.5 * (BnR**2 + Bt1R**2 + Bt2R**2)
    S_star = (rhoR * uR * (SR - uR) - rhoL * uL * (SL - uL) + (p_tot_L - p_tot_R)) / \
             (rhoR * (SR - uR) - rhoL * (SL - uL) + 1e-12)

    energy_flux_L = (EL + p_tot_L) * uL - BnL * (BnL * uL + Bt1L * vL + Bt2L * wL)
    energy_flux_R = (ER + p_tot_R) * uR - BnR * (BnR * uR + Bt1R * vR + Bt2R * wR)

    flux_mass = cp.where(SL >= 0, rhoL * uL,
                cp.where(SR <= 0, rhoR * uR,
                cp.where(S_star >= 0, rhoL * (SL - uL) * S_star / (SL - S_star),
                         rhoR * (SR - uR) * S_star / (SR - S_star))))

    flux_mu = cp.where(SL >= 0, rhoL * uL**2 + p_tot_L - BnL**2,
                cp.where(SR <= 0, rhoR * uR**2 + p_tot_R - BnR**2,
                cp.where(S_star >= 0, rhoL * (SL - uL) * S_star**2 / (SL - S_star) + p_tot_L - BnL**2,
                         rhoR * (SR - uR) * S_star**2 / (SR - S_star) + p_tot_R - BnR**2)))

    flux_energy = cp.where(SL >= 0, energy_flux_L,
                    cp.where(SR <= 0, energy_flux_R,
                    cp.where(S_star >= 0, (EL + p_tot_L) * S_star - BnL * (BnL * uL + Bt1L * vL + Bt2L * wL),
                             (ER + p_tot_R) * S_star - BnR * (BnR * uR + Bt1R * vR + Bt2R * wR))))

    return flux_mass, flux_mu, cp.zeros_like(vL), cp.zeros_like(wL), flux_energy

# ====================== FULL RHS ======================
def rhs(rho, mx, my, mz, E_total):
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx**2 + By**2 + Bz**2)
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - E_kin - E_mag), p_floor)

    drho = cp.zeros_like(rho)
    dmx = rho * g_x
    dmy = rho * g_y
    dmz = rho * g_z
    dE = cp.zeros_like(E_total)

    # x-sweep
    rhoL, rhoR = reconstruct_plm(rho, 0)
    mxL, mxR = reconstruct_plm(mx, 0)
    myL, myR = reconstruct_plm(my, 0)
    mzL, mzR = reconstruct_plm(mz, 0)
    EL, ER = reconstruct_plm(E_total, 0)
    pL, pR = reconstruct_plm(p_thermal, 0)
    uL = mxL / rhoL; uR = mxR / rhoR
    vL = myL / rhoL; vR = myR / rhoR
    wL = mzL / rhoL; wR = mzR / rhoR
    fm, fmx, _, _, fE = hlld_flux_1d(rhoL, rhoR, uL, uR, vL, vR, wL, wR, EL, ER, pL, pR, Bx, Bx, By, By, Bz, Bz)
    drho -= (fm - cp.roll(fm, 1, axis=0)) / dx
    dmx -= (fmx - cp.roll(fmx, 1, axis=0)) / dx
    dE -= (fE - cp.roll(fE, 1, axis=0)) / dx

    # y-sweep
    rhoL, rhoR = reconstruct_plm(rho, 1)
    mxL, mxR = reconstruct_plm(mx, 1)
    myL, myR = reconstruct_plm(my, 1)
    mzL, mzR = reconstruct_plm(mz, 1)
    EL, ER = reconstruct_plm(E_total, 1)
    pL, pR = reconstruct_plm(p_thermal, 1)
    uL = myL / rhoL; uR = myR / rhoR
    vL = mzL / rhoL; vR = mzR / rhoR
    wL = mxL / rhoL; wR = mxR / rhoR
    fm, fmy, _, _, fE = hlld_flux_1d(rhoL, rhoR, uL, uR, vL, vR, wL, wR, EL, ER, pL, pR, By, By, Bz, Bz, Bx, Bx)
    drho -= (fm - cp.roll(fm, 1, axis=1)) / dx
    dmy -= (fmy - cp.roll(fmy, 1, axis=1)) / dx
    dE -= (fE - cp.roll(fE, 1, axis=1)) / dx

    # z-sweep
    rhoL, rhoR = reconstruct_plm(rho, 2)
    mxL, mxR = reconstruct_plm(mx, 2)
    myL, myR = reconstruct_plm(my, 2)
    mzL, mzR = reconstruct_plm(mz, 2)
    EL, ER = reconstruct_plm(E_total, 2)
    pL, pR = reconstruct_plm(p_thermal, 2)
    uL = mzL / rhoL; uR = mzR / rhoR
    vL = mxL / rhoL; vR = mxR / rhoR
    wL = myL / rhoL; wR = myR / rhoR
    fm, fmz, _, _, fE = hlld_flux_1d(rhoL, rhoR, uL, uR, vL, vR, wL, wR, EL, ER, pL, pR, Bz, Bz, Bx, Bx, By, By)
    drho -= (fm - cp.roll(fm, 1, axis=2)) / dx
    dmz -= (fmz - cp.roll(fmz, 1, axis=2)) / dx
    dE -= (fE - cp.roll(fE, 1, axis=2)) / dx

    return drho, dmx, dmy, dmz, dE

print("Starting simulation...")

for step in range(steps):
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx**2 + By**2 + Bz**2)
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - E_kin - E_mag), p_floor)

    cf = cp.sqrt(gamma * p_thermal / rho + (Bx**2 + By**2 + Bz**2) / rho)
    v_total = cp.sqrt(vx**2 + vy**2 + vz**2)
    local_dt = CFL * dx / (v_total + cf + 1e-8)
    dt = float(cp.min(local_dt))

    # SSP-RK3 with renamed variables to avoid overwriting E0
    rho_rk = rho.copy()
    mx_rk = mx.copy()
    my_rk = my.copy()
    mz_rk = mz.copy()
    E_rk = E_total.copy()

    drho, dmx, dmy, dmz, dE = rhs(rho_rk, mx_rk, my_rk, mz_rk, E_rk)
    rho1 = rho_rk + dt * drho
    mx1 = mx_rk + dt * dmx
    my1 = my_rk + dt * dmy
    mz1 = mz_rk + dt * dmz
    E1 = E_rk + dt * dE

    drho, dmx, dmy, dmz, dE = rhs(rho1, mx1, my1, mz1, E1)
    rho2 = (3*rho_rk + rho1 + dt * drho) / 4
    mx2 = (3*mx_rk + mx1 + dt * dmx) / 4
    my2 = (3*my_rk + my1 + dt * dmy) / 4
    mz2 = (3*mz_rk + mz1 + dt * dmz) / 4
    E2 = (3*E_rk + E1 + dt * dE) / 4

    drho, dmx, dmy, dmz, dE = rhs(rho2, mx2, my2, mz2, E2)
    rho = (rho_rk + 2*rho2 + 2*dt * drho) / 3
    mx = (mx_rk + 2*mx2 + 2*dt * dmx) / 3
    my = (my_rk + 2*my2 + 2*dt * dmy) / 3
    mz = (mz_rk + 2*mz2 + 2*dt * dmz) / 3
    E_total = (E_rk + 2*E2 + 2*dt * dE) / 3

    # Floors and velocity cap
    rho = cp.maximum(rho, rho_floor)
    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx**2 + By**2 + Bz**2)
    internal_E = cp.maximum(E_total - E_kin - E_mag, internal_E_floor)
    E_total = E_kin + E_mag + internal_E

    vx = mx / rho
    vy = my / rho
    vz = mz / rho

    v = cp.sqrt(vx**2 + vy**2 + vz**2)
    scale = cp.minimum(1.0, v_max_cap / (v + 1e-8))
    vx *= scale
    vy *= scale
    vz *= scale
    mx = rho * vx
    my = rho * vy
    mz = rho * vz

    # Induction
    vx_avg = 0.5 * (vx + cp.roll(vx, -1, axis=0))
    vy_avg = 0.5 * (vy + cp.roll(vy, -1, axis=1))
    vz_avg = 0.5 * (vz + cp.roll(vz, -1, axis=2))
    Bx_avg = 0.5 * (Bx + cp.roll(Bx, -1, axis=0))
    By_avg = 0.5 * (By + cp.roll(By, -1, axis=1))
    Bz_avg = 0.5 * (Bz + cp.roll(Bz, -1, axis=2))

    Bx += dt / dx * (vy_avg * Bz_avg - vz_avg * By_avg)
    By += dt / dx * (vz_avg * Bx_avg - vx_avg * Bz_avg)
    Bz += dt / dx * (vx_avg * By_avg - vy_avg * Bx_avg)

    # Dedner
    divB = compute_divB()
    psi -= dt * c_h**2 * divB - dt * kappa * psi
    Bx -= dt * (cp.roll(psi, -1, axis=0) - psi) / dx
    By -= dt * (cp.roll(psi, -1, axis=1) - psi) / dx
    Bz -= dt * (cp.roll(psi, -1, axis=2) - psi) / dx

    if step % 50 == 0:
        mass_now = float(cp.sum(rho))
        E_now = float(cp.sum(E_total))
        Lz_now = float(cp.sum(rho * (X*vy - Y*vx)))
        div_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.nanmax(v))
        Bmax = float(cp.nanmax(cp.sqrt(Bx**2 + By**2 + Bz**2)))

        mass_drift = float(100.0 * (mass_now - mass0) / (mass0 + 1e-12))
        energy_drift = float(100.0 * (E_now - E0) / (E0 + 1e-12))
        lz_drift = float(100.0 * (Lz_now - Lz0) / (abs(Lz0) + 1e-12))

        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB = {div_max:.2e}")
        print(f"  Mass drift: {mass_drift:.4f}% | Energy drift: {energy_drift:.4f}% | Lz drift: {lz_drift:.4f}%")

print("\n✅ v1.0 Complete!")
