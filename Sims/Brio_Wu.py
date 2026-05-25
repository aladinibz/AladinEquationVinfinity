import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v61.0 — Brio-Wu MHD Shock Tube Test")

# ====================== PARAMETERS ======================
N = 512
L = 1.0
dx = L / N
x = cp.linspace(0, L, N, dtype=cp.float32)

gamma = 2.0   # Brio-Wu uses gamma=2
CFL = 0.4
steps = 200
t_final = 0.2

# ====================== BRIO-WU INITIAL CONDITIONS ======================
rho = cp.ones(N, dtype=cp.float32)
mx = cp.zeros(N, dtype=cp.float32)
my = cp.zeros(N, dtype=cp.float32)
mz = cp.zeros(N, dtype=cp.float32)
E_total = cp.ones(N, dtype=cp.float32) * 0.5

Bx = cp.ones(N+1, dtype=cp.float32) * 0.75   # constant normal field
By = cp.zeros(N, dtype=cp.float32)
Bz = cp.zeros(N, dtype=cp.float32)

# Left state (x < 0.5)
left = x < 0.5
rho[left] = 1.0
E_total[left] = 1.0 / (gamma - 1) + 0.5 * (1.0**2 + 0.0**2) + 0.5 * (0.75**2 + 1.0**2)
By[left] = 1.0

# Right state (x > 0.5)
right = x >= 0.5
rho[right] = 0.125
E_total[right] = 0.1 / (gamma - 1) + 0.5 * (0.125 * 0.0**2) + 0.5 * (0.75**2 + (-1.0)**2)
By[right] = -1.0

# ====================== HLLD SOLVER ======================
def hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx_normal):
    vxL = mxL / rhoL
    vxR = mxR / rhoR
    vyL = myL / rhoL
    vyR = myR / rhoR
    vzL = mzL / rhoL
    vzR = mzR / rhoR

    cs2L = gamma * pL / rhoL
    cs2R = gamma * pR / rhoR
    ca2L = (Bx_normal**2 + ByL**2 + BzL**2) / rhoL
    ca2R = (Bx_normal**2 + ByR**2 + BzR**2) / rhoR
    cfL = cp.sqrt(0.5 * (cs2L + ca2L + cp.sqrt((cs2L + ca2L)**2 - 4*cs2L*Bx_normal**2/rhoL)))
    cfR = cp.sqrt(0.5 * (cs2R + ca2R + cp.sqrt((cs2R + ca2R)**2 - 4*cs2R*Bx_normal**2/rhoR)))

    SL = cp.minimum(0.0, cp.minimum(vxL - cfL, vxR - cfR))
    SR = cp.maximum(0.0, cp.maximum(vxL + cfL, vxR + cfR))

    p_mag_L = 0.5 * (ByL**2 + BzL**2)
    p_mag_R = 0.5 * (ByR**2 + BzR**2)
    p_tot_L = pL + p_mag_L
    p_tot_R = pR + p_mag_R

    S_star = (rhoR * vxR * (SR - vxR) - rhoL * vxL * (SL - vxL) +
              (pL - pR) + 0.5*(p_mag_L - p_mag_R)) / \
             (rhoR * (SR - vxR) - rhoL * (SL - vxL) + 1e-12)

    # Left star
    rho_star_L = rhoL * (SL - vxL) / (SL - S_star + 1e-12)
    mx_star_L = rho_star_L * S_star
    By_star_L = ByL * (SL - vxL) / (SL - S_star + 1e-12)
    Bz_star_L = BzL * (SL - vxL) / (SL - S_star + 1e-12)
    my_star_L = myL - Bx_normal * (By_star_L - ByL) / cp.sqrt(rhoL + 1e-12)
    mz_star_L = mzL - Bx_normal * (Bz_star_L - BzL) / cp.sqrt(rhoL + 1e-12)

    # Right star
    rho_star_R = rhoR * (SR - vxR) / (SR - S_star + 1e-12)
    mx_star_R = rho_star_R * S_star
    By_star_R = ByR * (SR - vxR) / (SR - S_star + 1e-12)
    Bz_star_R = BzR * (SR - vxR) / (SR - S_star + 1e-12)
    my_star_R = myR - Bx_normal * (By_star_R - ByR) / cp.sqrt(rhoR + 1e-12)
    mz_star_R = mzR - Bx_normal * (Bz_star_R - BzR) / cp.sqrt(rhoR + 1e-12)

    # Flux selection
    flux_mass = cp.where(SL >= 0, rhoL * vxL,
                cp.where(SR <= 0, rhoR * vxR,
                cp.where(S_star >= 0, rho_star_L * S_star, rho_star_R * S_star)))

    flux_mx = cp.where(SL >= 0, mxL * vxL + p_tot_L - Bx_normal**2,
                cp.where(SR <= 0, mxR * vxR + p_tot_R - Bx_normal**2,
                cp.where(S_star >= 0, mx_star_L * S_star + p_tot_L - Bx_normal**2,
                mx_star_R * S_star + p_tot_R - Bx_normal**2)))

    flux_my = cp.where(SL >= 0, myL * vxL - Bx_normal * ByL,
                cp.where(SR <= 0, myR * vxR - Bx_normal * ByR,
                cp.where(S_star >= 0, my_star_L * S_star - Bx_normal * By_star_L,
                my_star_R * S_star - Bx_normal * By_star_R)))

    flux_mz = cp.where(SL >= 0, mzL * vxL - Bx_normal * BzL,
                cp.where(SR <= 0, mzR * vxR - Bx_normal * BzR,
                cp.where(S_star >= 0, mz_star_L * S_star - Bx_normal * Bz_star_L,
                mz_star_R * S_star - Bx_normal * Bz_star_R)))

    flux_energy = cp.where(SL >= 0, (EL + p_tot_L) * vxL - Bx_normal * (Bx_normal * vxL + ByL * vyL + BzL * vzL),
                    cp.where(SR <= 0, (ER + p_tot_R) * vxR - Bx_normal * (Bx_normal * vxR + ByR * vyR + BzR * vzR),
                    cp.where(S_star >= 0, (EL + p_tot_L) * S_star - Bx_normal * (Bx_normal * vxL + ByL * vyL + BzL * vzL),
                    (ER + p_tot_R) * S_star - Bx_normal * (Bx_normal * vxR + ByR * vyR + BzR * vzR))))

    return flux_mass, flux_mx, flux_my, flux_mz, flux_energy

# ====================== 1D SHOCK TUBE EVOLUTION ======================
print("Running Brio-Wu MHD Shock Tube Test...")

for step in range(steps):
    dt = CFL * dx / 2.0   # conservative CFL for test

    # MUSCL reconstruction
    rhoL, rhoR = reconstruct_plm(rho)
    mxL, mxR = reconstruct_plm(mx)
    myL, myR = reconstruct_plm(my)
    mzL, mzR = reconstruct_plm(mz)
    EL, ER = reconstruct_plm(E_total)
    pL = cp.maximum((gamma - 1) * (EL - 0.5 * (mxL**2 + myL**2 + mzL**2) / rhoL), p_floor)
    pR = cp.maximum((gamma - 1) * (ER - 0.5 * (mxR**2 + myR**2 + mzR**2) / rhoR), p_floor)

    f_mass, f_mx, f_my, f_mz, f_E = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx[0])

    rho -= dt * (f_mass - cp.roll(f_mass, 1)) / dx
    mx -= dt * (f_mx - cp.roll(f_mx, 1)) / dx
    my -= dt * (f_my - cp.roll(f_my, 1)) / dx
    mz -= dt * (f_mz - cp.roll(f_mz, 1)) / dx
    E_total -= dt * (f_E - cp.roll(f_E, 1)) / dx

    # Primitive update
    rho = cp.maximum(rho, rho_floor)
    vx = mx / rho
    vy = my / rho
    vz = mz / rho

print("Simulation finished. Plotting results...")

# Plot
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(x.get(), rho.get(), 'b-', label='Density')
axs[0].set_title('Density')
axs[1].plot(x.get(), (gamma-1)*(E_total.get() - 0.5*rho.get()*(vx.get()**2 + vy.get()**2 + vz.get()**2) - 0.5*(Bx.get()[:-1]**2 + By.get()**2 + Bz.get()**2)), 'r-', label='Pressure')
axs[1].set_title('Pressure')
axs[2].plot(x.get(), vy.get(), 'g-', label='v_y')
axs[2].set_title('Transverse Velocity v_y')
axs[3].plot(x.get(), By.get(), 'm-', label='B_y')
axs[3].set_title('Transverse Magnetic Field B_y')

for ax in axs:
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

print("\n✅ Brio-Wu Shock Tube Test Completed!")
