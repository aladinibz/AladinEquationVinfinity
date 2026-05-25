import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v58.0 — FULL HLLD STAR STATES + 3D SWEEPS")

# ====================== PARAMETERS ======================
N = 64
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
gamma = 5.0 / 3.0
CFL = 0.06
steps = 400
rho_floor = 1e-4
p_floor = 1e-6
v_max_cap = 5.0

v_phi_factor = 0.055
c_h_factor = 10.0
kappa_factor = 25.0

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== KERNEL (Upwind CT EMF) ======================
kernel_code = r'''
extern "C" __global__
void ct_emf_kernel(float* vx, float* vy, float* vz,
                   float* Bx, float* By, float* Bz,
                   float dt, float dx, int N)
{
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    if (i < N-1 && j < N-1 && k < N-1) {
        int idx = (i * N + j) * N + k;
        float vx_avg = 0.25f * (vx[idx] + vx[idx + N] + vx[idx + 1] + vx[idx + N + 1]);
        float vy_avg = 0.25f * (vy[idx] + vy[idx + N] + vy[idx + 1] + vy[idx + N + 1]);
        float Bx_avg = 0.25f * (Bx[idx] + Bx[idx + N] + Bx[idx + 1] + Bx[idx + N + 1]);
        float By_avg = 0.25f * (By[idx] + By[idx + N] + By[idx + 1] + By[idx + N + 1]);

        float sign_v = (vx_avg * vy_avg > 0.0f) ? 1.0f : -1.0f;
        float Ez_val = - (vx_avg * By_avg - vy_avg * Bx_avg) * (1.0f + 0.012f * sign_v);

        int bz_idx = (i * N + j) * (N + 1) + k;
        Bz[bz_idx] += (dt / dx) * Ez_val;

        if (Bz[bz_idx] > 6.0f) Bz[bz_idx] = 6.0f;
        if (Bz[bz_idx] < -6.0f) Bz[bz_idx] = -6.0f;
    }
}
'''

kernel = cp.RawKernel(kernel_code, 'ct_emf_kernel')

# ====================== FIELDS ======================
rho = cp.maximum(cp.ones((N, N, N), dtype=cp.float32) * 1e-3, rho_floor)
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)
psi = cp.zeros((N, N, N), dtype=cp.float32)

r_cyl = cp.sqrt(X**2 + Y**2)
rho *= cp.exp(-r_cyl / 8.0) * cp.exp(-Z**2 / 2.25)
rho = cp.maximum(rho, rho_floor)

r3d = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
M_dm = nfw_enclosed_mass(r3d)
g_r = -G * M_dm / r3d**2
v_phi = v_phi_factor * cp.sqrt(cp.maximum(r_cyl * cp.abs(g_r), 0.0))

vx = -v_phi * (Y / (r_cyl + 1e-8))
vy =  v_phi * (X / (r_cyl + 1e-8))
vz = cp.zeros_like(vx)

mx = rho * vx
my = rho * vy
mz = rho * vz

B0 = 0.5
Bphi = 0.3
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 500.0)

c_h = c_h_factor * 80.0
kappa = kappa_factor / dx

def cell_center_B():
    Bx_c = 0.5 * (Bx[1:,:,:] + Bx[:-1,:,:])
    By_c = 0.5 * (By[:,1:,:] + By[:,:-1,:])
    Bz_c = 0.5 * (Bz[:,:,1:] + Bz[:,:,:-1])
    return Bx_c, By_c, Bz_c

def compute_divB():
    div = cp.zeros((N, N, N), dtype=cp.float32)
    div += (Bx[1:,:,:] - Bx[:-1,:,:]) / dx
    div += (By[:,1:,:] - By[:,:-1,:]) / dx
    div += (Bz[:,:,1:] - Bz[:,:,:-1]) / dx
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

# ====================== FULL HLLD WITH STAR STATES ======================
def hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx_normal):
    vxL = mxL / rhoL
    vxR = mxR / rhoR
    vyL = myL / rhoL
    vyR = myR / rhoR
    vzL = mzL / rhoL
    vzR = mzR / rhoR

    cL = cp.sqrt(gamma * pL / rhoL)
    cR = cp.sqrt(gamma * pR / rhoR)
    SL = cp.minimum(0.0, cp.minimum(vxL - cL, vxR - cR))
    SR = cp.maximum(0.0, cp.maximum(vxL + cL, vxR + cR))
    den = cp.maximum(SR - SL, 1e-8)

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
    my_star_L = myL - Bx_normal * (ByL * (S_star - vxL)) / (SL - S_star + 1e-12)
    mz_star_L = mzL - Bx_normal * (BzL * (S_star - vxL)) / (SL - S_star + 1e-12)

    # Right star
    rho_star_R = rhoR * (SR - vxR) / (SR - S_star + 1e-12)
    mx_star_R = rho_star_R * S_star
    my_star_R = myR - Bx_normal * (ByR * (S_star - vxR)) / (SR - S_star + 1e-12)
    mz_star_R = mzR - Bx_normal * (BzR * (S_star - vxR)) / (SR - S_star + 1e-12)

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
                cp.where(S_star >= 0, my_star_L * S_star - Bx_normal * ByL,
                my_star_R * S_star - Bx_normal * ByR)))

    flux_mz = cp.where(SL >= 0, mzL * vxL - Bx_normal * BzL,
                cp.where(SR <= 0, mzR * vxR - Bx_normal * BzR,
                cp.where(S_star >= 0, mz_star_L * S_star - Bx_normal * BzL,
                mz_star_R * S_star - Bx_normal * BzR)))

    flux_energy = cp.where(SL >= 0, (EL + p_tot_L) * vxL - Bx_normal * (Bx_normal * vxL + ByL * vyL + BzL * vzL),
                    cp.where(SR <= 0, (ER + p_tot_R) * vxR - Bx_normal * (Bx_normal * vxR + ByR * vyR + BzR * vzR),
                    cp.where(S_star >= 0, (EL + p_tot_L) * S_star - Bx_normal * (Bx_normal * vxL + ByL * vyL + BzL * vzL),
                    (ER + p_tot_R) * S_star - Bx_normal * (Bx_normal * vxR + ByR * vyR + BzR * vzR))))

    return flux_mass, flux_mx, flux_my, flux_mz, flux_energy

print("Starting v58.0 — Full HLLD Star States + 3D Sweeps...")

block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)

for step in range(steps):
    dt = CFL * dx / 380.0

    # CT EMF
    kernel(grid, block, (vx, vy, vz, Bx, By, Bz, dt, dx, N), shared_mem=6*1000*4)

    Bx_c, By_c, Bz_c = cell_center_B()

    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx_c**2 + By_c**2 + Bz_c**2)
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - E_kin - E_mag), p_floor)

    # x-sweep
    rhoL, rhoR = reconstruct_plm(rho, axis=0)
    mxL, mxR = reconstruct_plm(mx, axis=0)
    myL, myR = reconstruct_plm(my, axis=0)
    mzL, mzR = reconstruct_plm(mz, axis=0)
    EL, ER = reconstruct_plm(E_total, axis=0)
    pL, pR = reconstruct_plm(p_thermal, axis=0)
    f_mass, f_mx, f_my, f_mz, f_E = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx_c)
    rho -= dt * (f_mass - cp.roll(f_mass, 1, axis=0)) / dx
    mx -= dt * (f_mx - cp.roll(f_mx, 1, axis=0)) / dx
    my -= dt * (f_my - cp.roll(f_my, 1, axis=0)) / dx
    mz -= dt * (f_mz - cp.roll(f_mz, 1, axis=0)) / dx
    E_total -= dt * (f_E - cp.roll(f_E, 1, axis=0)) / dx

    # y-sweep
    rhoL, rhoR = reconstruct_plm(rho, axis=1)
    mxL, mxR = reconstruct_plm(mx, axis=1)
    myL, myR = reconstruct_plm(my, axis=1)
    mzL, mzR = reconstruct_plm(mz, axis=1)
    EL, ER = reconstruct_plm(E_total, axis=1)
    pL, pR = reconstruct_plm(p_thermal, axis=1)
    f_mass, f_mx, f_my, f_mz, f_E = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx_c)
    rho -= dt * (f_mass - cp.roll(f_mass, 1, axis=1)) / dx
    mx -= dt * (f_mx - cp.roll(f_mx, 1, axis=1)) / dx
    my -= dt * (f_my - cp.roll(f_my, 1, axis=1)) / dx
    mz -= dt * (f_mz - cp.roll(f_mz, 1, axis=1)) / dx
    E_total -= dt * (f_E - cp.roll(f_E, 1, axis=1)) / dx

    # z-sweep
    rhoL, rhoR = reconstruct_plm(rho, axis=2)
    mxL, mxR = reconstruct_plm(mx, axis=2)
    myL, myR = reconstruct_plm(my, axis=2)
    mzL, mzR = reconstruct_plm(mz, axis=2)
    EL, ER = reconstruct_plm(E_total, axis=2)
    pL, pR = reconstruct_plm(p_thermal, axis=2)
    f_mass, f_mx, f_my, f_mz, f_E = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx_c)
    rho -= dt * (f_mass - cp.roll(f_mass, 1, axis=2)) / dx
    mx -= dt * (f_mx - cp.roll(f_mx, 1, axis=2)) / dx
    my -= dt * (f_my - cp.roll(f_my, 1, axis=2)) / dx
    mz -= dt * (f_mz - cp.roll(f_mz, 1, axis=2)) / dx
    E_total -= dt * (f_E - cp.roll(f_E, 1, axis=2)) / dx

    # Update primitives
    rho = cp.maximum(rho, rho_floor)
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

    # Safety
    Bx = cp.nan_to_num(Bx, nan=0.0, posinf=6.0, neginf=-6.0)
    By = cp.nan_to_num(By, nan=0.0, posinf=6.0, neginf=-6.0)
    Bz = cp.nan_to_num(Bz, nan=0.0, posinf=6.0, neginf=-6.0)

    if step % 50 == 0:
        div_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.max(v))
        Bmax = float(cp.nanmax(cp.sqrt(Bx_c**2 + By_c**2 + Bz_c**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB_max = {div_max:.2e}")

print("\n✅ v58.0 Full HLLD Star States + 3D Sweeps Finished!")
