import cupy as cp
import numpy as np

print("🌌 ALADIN Plasma Cosmology v65.2 — Fixed & Complete Version")

# ====================== PARAMETERS ======================
N = 64
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
gamma = 5.0 / 3.0
CFL = 0.4
steps = 500
rho_floor = 1e-4
p_floor = 1e-6

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

# ====================== OPTIMIZED CT KERNEL ======================
kernel_code = r'''
extern "C" __global__
void ct_emf_kernel(const float* __restrict__ vx,
                   const float* __restrict__ vy,
                   const float* __restrict__ vz,
                   float* __restrict__ Bz,
                   float dt, float dx, int N)
{
    extern __shared__ float s_data[];
    float* s_vx = s_data;
    float* s_vy = s_vx + (blockDim.x + 2)*(blockDim.y + 2)*(blockDim.z + 2);
    float* s_Bx = s_vy + (blockDim.x + 2)*(blockDim.y + 2)*(blockDim.z + 2);
    float* s_By = s_Bx + (blockDim.x + 2)*(blockDim.y + 2)*(blockDim.z + 2);

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    int s_idx = (tx + 1) + (ty + 1)*(blockDim.x + 2) + (tz + 1)*(blockDim.x + 2)*(blockDim.y + 2);
    int g_idx = (i * N + j) * N + k;

    if (i < N && j < N && k < N) {
        s_vx[s_idx] = vx[g_idx];
        s_vy[s_idx] = vy[g_idx];
        s_Bx[s_idx] = Bx[g_idx];
        s_By[s_idx] = By[g_idx];
    }
    __syncthreads();

    if (i < N-1 && j < N-1 && k < N-1) {
        float vx_avg = 0.25f * (s_vx[s_idx] + s_vx[s_idx+(blockDim.x+2)] + s_vx[s_idx+1] + s_vx[s_idx+(blockDim.x+2)+1]);
        float vy_avg = 0.25f * (s_vy[s_idx] + s_vy[s_idx+(blockDim.x+2)] + s_vy[s_idx+1] + s_vy[s_idx+(blockDim.x+2)+1]);
        float Bx_avg = 0.25f * (s_Bx[s_idx] + s_Bx[s_idx+(blockDim.x+2)] + s_Bx[s_idx+1] + s_Bx[s_idx+(blockDim.x+2)+1]);
        float By_avg = 0.25f * (s_By[s_idx] + s_By[s_idx+(blockDim.x+2)] + s_By[s_idx+1] + s_By[s_idx+(blockDim.x+2)+1]);

        float sign_v = (vx_avg * vy_avg > 0.0f) ? 1.0f : -1.0f;
        float Ez_val = - (vx_avg * By_avg - vy_avg * Bx_avg) * (1.0f + 0.012f * sign_v);

        int bz_idx = (i * N + j) * (N + 1) + k;
        atomicAdd(&Bz[bz_idx], (dt / dx) * Ez_val);

        if (Bz[bz_idx] > 6.0f) atomicExch(&Bz[bz_idx], 6.0f);
        if (Bz[bz_idx] < -6.0f) atomicExch(&Bz[bz_idx], -6.0f);
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

B0 = 0.5
Bphi = 0.3
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 500.0)

mass0 = float(cp.sum(rho))
E0 = float(cp.sum(E_total))
Lz0 = float(cp.sum(rho * (X*vy - Y*vx)))

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

# ====================== TRUE HLLD ======================
def hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, B_normal, By_normal, Bz_normal):
    vxL = mxL / rhoL
    vxR = mxR / rhoR
    vyL = myL / rhoL
    vyR = myR / rhoR
    vzL = mzL / rhoL
    vzR = mzR / rhoR

    cs2L = gamma * pL / rhoL
    cs2R = gamma * pR / rhoR
    ca2L = (B_normal**2 + By_normal**2 + Bz_normal**2) / rhoL
    ca2R = (B_normal**2 + By_normal**2 + Bz_normal**2) / rhoR
    cfL = cp.sqrt(0.5 * (cs2L + ca2L + cp.sqrt((cs2L + ca2L)**2 - 4*cs2L*B_normal**2/rhoL)))
    cfR = cp.sqrt(0.5 * (cs2R + ca2R + cp.sqrt((cs2R + ca2R)**2 - 4*cs2R*B_normal**2/rhoR)))

    SL = cp.minimum(0.0, cp.minimum(vxL - cfL, vxR - cfR))
    SR = cp.maximum(0.0, cp.maximum(vxL + cfL, vxR + cfR))

    p_mag_L = 0.5 * (By_normal**2 + Bz_normal**2)
    p_mag_R = 0.5 * (By_normal**2 + Bz_normal**2)
    p_tot_L = pL + p_mag_L
    p_tot_R = pR + p_mag_R

    S_star = (rhoR * vxR * (SR - vxR) - rhoL * vxL * (SL - vxL) +
              (pL - pR) + 0.5*(p_mag_L - p_mag_R)) / \
             (rhoR * (SR - vxR) - rhoL * (SL - vxL) + 1e-12)

    rho_star_L = rhoL * (SL - vxL) / (SL - S_star + 1e-12)
    mx_star_L = rho_star_L * S_star
    By_star_L = By_normal * (SL - vxL) / (SL - S_star + 1e-12)
    Bz_star_L = Bz_normal * (SL - vxL) / (SL - S_star + 1e-12)
    my_star_L = myL - B_normal * (By_star_L - By_normal) / cp.sqrt(rhoL + 1e-12)
    mz_star_L = mzL - B_normal * (Bz_star_L - Bz_normal) / cp.sqrt(rhoL + 1e-12)

    rho_star_R = rhoR * (SR - vxR) / (SR - S_star + 1e-12)
    mx_star_R = rho_star_R * S_star
    By_star_R = By_normal * (SR - vxR) / (SR - S_star + 1e-12)
    Bz_star_R = Bz_normal * (SR - vxR) / (SR - S_star + 1e-12)
    my_star_R = myR - B_normal * (By_star_R - By_normal) / cp.sqrt(rhoR + 1e-12)
    mz_star_R = mzR - B_normal * (Bz_star_R - Bz_normal) / cp.sqrt(rhoR + 1e-12)

    flux_mass = cp.where(SL >= 0, rhoL * vxL, cp.where(SR <= 0, rhoR * vxR, cp.where(S_star >= 0, rho_star_L * S_star, rho_star_R * S_star)))
    flux_mx = cp.where(SL >= 0, mxL * vxL + p_tot_L - B_normal**2, cp.where(SR <= 0, mxR * vxR + p_tot_R - B_normal**2, cp.where(S_star >= 0, mx_star_L * S_star + p_tot_L - B_normal**2, mx_star_R * S_star + p_tot_R - B_normal**2)))
    flux_my = cp.where(SL >= 0, myL * vxL - B_normal * By_normal, cp.where(SR <= 0, myR * vxR - B_normal * By_normal, cp.where(S_star >= 0, my_star_L * S_star - B_normal * By_star_L, my_star_R * S_star - B_normal * By_star_R)))
    flux_mz = cp.where(SL >= 0, mzL * vxL - B_normal * Bz_normal, cp.where(SR <= 0, mzR * vxR - B_normal * Bz_normal, cp.where(S_star >= 0, mz_star_L * S_star - B_normal * Bz_star_L, mz_star_R * S_star - B_normal * Bz_star_R)))
    flux_energy = cp.where(SL >= 0, (EL + p_tot_L) * vxL - B_normal * (B_normal * vxL + By_normal * vyL + Bz_normal * vzL),
                    cp.where(SR <= 0, (ER + p_tot_R) * vxR - B_normal * (B_normal * vxR + By_normal * vyR + Bz_normal * vzR),
                    cp.where(S_star >= 0, (EL + p_tot_L) * S_star - B_normal * (B_normal * vxL + By_normal * vyL + Bz_normal * vzL),
                    (ER + p_tot_R) * S_star - B_normal * (B_normal * vxR + By_normal * vyR + Bz_normal * vzR))))

    return flux_mass, flux_mx, flux_my, flux_mz, flux_energy

# ====================== SSP-RK3 RHS ======================
def rhs(rho, mx, my, mz, E_total):
    Bx_c, By_c, Bz_c = cell_center_B()
    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx_c**2 + By_c**2 + Bz_c**2)
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - E_kin - E_mag), p_floor)

    dmx = rho * g_x
    dmy = rho * g_y
    dmz = rho * g_z
    dE = rho * (vx * g_x + vy * g_y + vz * g_z)

    # x-sweep
    rhoL, rhoR = reconstruct_plm(rho, 0)
    mxL, mxR = reconstruct_plm(mx, 0)
    myL, myR = reconstruct_plm(my, 0)
    mzL, mzR = reconstruct_plm(mz, 0)
    EL, ER = reconstruct_plm(E_total, 0)
    pL, pR = reconstruct_plm(p_thermal, 0)
    fm, fmx, fmy, fmz, fE = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bx_c, By_c, Bz_c)
    drho = - (fm - cp.roll(fm, 1, axis=0)) / dx
    dmx += - (fmx - cp.roll(fmx, 1, axis=0)) / dx
    dmy += - (fmy - cp.roll(fmy, 1, axis=0)) / dx
    dmz += - (fmz - cp.roll(fmz, 1, axis=0)) / dx
    dE += - (fE - cp.roll(fE, 1, axis=0)) / dx

    # y-sweep
    rhoL, rhoR = reconstruct_plm(rho, 1)
    mxL, mxR = reconstruct_plm(mx, 1)
    myL, myR = reconstruct_plm(my, 1)
    mzL, mzR = reconstruct_plm(mz, 1)
    EL, ER = reconstruct_plm(E_total, 1)
    pL, pR = reconstruct_plm(p_thermal, 1)
    fm, fmx, fmy, fmz, fE = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, By_c, Bx_c, Bz_c)
    drho += - (fm - cp.roll(fm, 1, axis=1)) / dx
    dmx += - (fmx - cp.roll(fmx, 1, axis=1)) / dx
    dmy += - (fmy - cp.roll(fmy, 1, axis=1)) / dx
    dmz += - (fmz - cp.roll(fmz, 1, axis=1)) / dx
    dE += - (fE - cp.roll(fE, 1, axis=1)) / dx

    # z-sweep
    rhoL, rhoR = reconstruct_plm(rho, 2)
    mxL, mxR = reconstruct_plm(mx, 2)
    myL, myR = reconstruct_plm(my, 2)
    mzL, mzR = reconstruct_plm(mz, 2)
    EL, ER = reconstruct_plm(E_total, 2)
    pL, pR = reconstruct_plm(p_thermal, 2)
    fm, fmx, fmy, fmz, fE = hlld_flux_1d(rhoL, rhoR, mxL, mxR, myL, myR, mzL, mzR, EL, ER, pL, pR, Bz_c, Bx_c, By_c)
    drho += - (fm - cp.roll(fm, 1, axis=2)) / dx
    dmx += - (fmx - cp.roll(fmx, 1, axis=2)) / dx
    dmy += - (fmy - cp.roll(fmy, 1, axis=2)) / dx
    dmz += - (fmz - cp.roll(fmz, 1, axis=2)) / dx
    dE += - (fE - cp.roll(fE, 1, axis=2)) / dx

    return drho, dmx, dmy, dmz, dE

print("Starting simulation...")

block = (16, 16, 4)
grid = ((N + 15)//16, (N + 15)//16, (N + 3)//4)
shared_mem_bytes = 4 * 4 * (18 * 18 * 6)

for step in range(steps):
    Bx_c, By_c, Bz_c = cell_center_B()
    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx_c**2 + By_c**2 + Bz_c**2)
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - E_kin - E_mag), p_floor)

    cf = cp.sqrt(gamma * p_thermal / rho + (Bx_c**2 + By_c**2 + Bz_c**2) / rho)
    max_speed = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2) + cf + 1e-8))
    dt = CFL * dx / max_speed

    # SSP-RK3
    rho0 = rho.copy()
    mx0 = mx.copy()
    my0 = my.copy()
    mz0 = mz.copy()
    E0 = E_total.copy()

    drho, dmx, dmy, dmz, dE = rhs(rho0, mx0, my0, mz0, E0)
    rho1 = rho0 + dt * drho
    mx1 = mx0 + dt * dmx
    my1 = my0 + dt * dmy
    mz1 = mz0 + dt * dmz
    E1 = E0 + dt * dE

    drho, dmx, dmy, dmz, dE = rhs(rho1, mx1, my1, mz1, E1)
    rho2 = (3*rho0 + rho1 + dt * drho) / 4
    mx2 = (3*mx0 + mx1 + dt * dmx) / 4
    my2 = (3*my0 + my1 + dt * dmy) / 4
    mz2 = (3*mz0 + mz1 + dt * dmz) / 4
    E2 = (3*E0 + E1 + dt * dE) / 4

    drho, dmx, dmy, dmz, dE = rhs(rho2, mx2, my2, mz2, E2)
    rho = (rho0 + 2*rho2 + 2*dt * drho) / 3
    mx = (mx0 + 2*mx2 + 2*dt * dmx) / 3
    my = (my0 + 2*my2 + 2*dt * dmy) / 3
    mz = (mz0 + 2*mz2 + 2*dt * dmz) / 3
    E_total = (E0 + 2*E2 + 2*dt * dE) / 3

    # Update primitives
    rho = cp.maximum(rho, rho_floor)
    vx = mx / rho
    vy = my / rho
    vz = mz / rho

    if step % 50 == 0:
        mass_now = float(cp.sum(rho))
        E_now = float(cp.sum(E_total))
        Lz_now = float(cp.sum(rho * (X*vy - Y*vx)))
        div_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.nanmax(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.nanmax(cp.sqrt(Bx_c**2 + By_c**2 + Bz_c**2)))

        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB = {div_max:.2e}")
        print(f"  Mass drift: {100*(mass_now-mass0)/mass0:.4f}% | Energy drift: {100*(E_now-E0)/E0:.4f}% | Lz drift: {100*(Lz_now-Lz0)/Lz0:.4f}%")

print("\n✅ v65.1 Simulation Finished!")
