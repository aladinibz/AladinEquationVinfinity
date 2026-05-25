import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v40.9 — FIXED DEDNER + STAGGERED J×B")

# ====================== PARAMETERS ======================
N = 64
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
gamma = 5.0 / 3.0
CFL = 0.15
steps = 400
mu0 = 1.0

v_phi_factor = 0.12
c_h_factor = 12.0
kappa_factor = 15.0

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== FUSED KERNEL ======================
kernel_code = r'''
extern "C" __global__
void ct_yee_kernel(float* vx, float* vy, float* vz,
                   float* Bx, float* By, float* Bz,
                   float dt, float dx, int N)
{
    extern __shared__ float sdata[];
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    int txs = blockDim.x + 2;
    int tys = blockDim.y + 2;
    int tzs = blockDim.z + 2;

    float* s_vx = sdata;
    float* s_vy = s_vx + txs*tys*tzs;
    float* s_vz = s_vy + txs*tys*tzs;
    float* s_Bx = s_vz + txs*tys*tzs;
    float* s_By = s_Bx + txs*tys*tzs;
    float* s_Bz = s_By + txs*tys*tzs;

    if (i < N && j < N && k < N) {
        int g = (i*N + j)*N + k;
        int s = (tx+1)*tys*tzs + (ty+1)*tzs + (tz+1);
        s_vx[s] = vx[g];
        s_vy[s] = vy[g];
        s_vz[s] = vz[g];
        s_Bx[s] = Bx[g];
        s_By[s] = By[g];
        s_Bz[s] = Bz[g];
    }
    __syncthreads();

    if (i < N-1 && j < N-1 && k < N-1) {
        int s = (tx+1)*tys*tzs + (ty+1)*tzs + (tz+1);
        float vx_avg = 0.25f * (s_vx[s] + s_vx[s+tys*tzs] + s_vx[s+tzs] + s_vx[s+tys*tzs+tzs]);
        float vy_avg = 0.25f * (s_vy[s] + s_vy[s+tys*tzs] + s_vy[s+tzs] + s_vy[s+tys*tzs+tzs]);
        float Bx_avg = 0.25f * (s_Bx[s] + s_Bx[s+tys*tzs] + s_Bx[s+tzs] + s_Bx[s+tys*tzs+tzs]);
        float By_avg = 0.25f * (s_By[s] + s_By[s+tys*tzs] + s_By[s+tzs] + s_By[s+tys*tzs+tzs]);

        float Ez_val = 0.04f * (-(vx_avg * By_avg - vy_avg * Bx_avg));

        int bz_idx = (i*N + j)*(N+1) + k;
        Bz[bz_idx] += (dt / dx) * Ez_val;
    }
}
'''

kernel = cp.RawKernel(kernel_code, 'ct_yee_kernel')

# ====================== FIELDS ======================
rho = cp.ones((N, N, N), dtype=cp.float32) * 1e-3
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

B0 = 1.5
Bphi = 0.8
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 250.0)

c_h = c_h_factor * 120.0
kappa = kappa_factor / dx

def compute_divB():
    div = cp.zeros((N, N, N), dtype=cp.float32)
    div += (Bx[1:,:,:] - Bx[:-1,:,:]) / dx
    div += (By[:,1:,:] - By[:,:-1,:]) / dx
    div += (Bz[:,:,1:] - Bz[:,:,:-1]) / dx
    return div

def compute_staggered_JxB():
    Jx = (By[:,1:,:] - By[:,:-1,:]) / dx - (Bz[1:,:,:] - Bz[:-1,:,:]) / dx
    Jy = (Bz[:,:,1:] - Bz[:,:,:-1]) / dx - (Bx[1:,:,:] - Bx[:-1,:,:]) / dx
    Jz = (Bx[:,1:,:] - Bx[:,:-1,:]) / dx - (By[1:,:,:] - By[:-1,:,:]) / dx

    # Safe cell-centered
    Jx_c = 0.5 * (Jx[:,:-1,:-1] + Jx[:,1:,:-1])
    Jy_c = 0.5 * (Jy[:-1,:,:-1] + Jy[1:,:,:-1])
    Jz_c = 0.5 * (Jz[:-1,:-1,:] + Jz[:-1,1:,:])

    Bx_c = 0.5 * (Bx[1:,:,:] + Bx[:-1,:,:])
    By_c = 0.5 * (By[:,1:,:] + By[:,:-1,:])
    Bz_c = 0.5 * (Bz[:,:,1:] + Bz[:,:,:-1])

    fx = Jy_c * Bz_c - Jz_c * By_c
    fy = Jz_c * Bx_c - Jx_c * Bz_c
    fz = Jx_c * By_c - Jy_c * Bx_c
    return fx, fy, fz

print("Starting v40.9 with Fixed Dedner + Staggered J×B...")

block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)
shared_bytes = 6 * (10**3) * 4

for step in range(steps):
    dt = CFL * dx / 280.0
    
    kernel(grid, block, (vx, vy, vz, Bx, By, Bz, dt, dx, N), shared_mem=shared_bytes)

    # === FIXED DEDNER (direct gradient) ===
    divB = compute_divB()
    psi -= dt * c_h**2 * divB - dt * kappa * psi

    psi_x = (psi[1:,:,:] - psi[:-1,:,:]) / dx
    psi_y = (psi[:,1:,:] - psi[:,:-1,:]) / dx
    psi_z = (psi[:,:,1:] - psi[:,:,:-1]) / dx

    Bx[1:-1,:,:] -= dt * psi_x
    By[:,1:-1,:] -= dt * psi_y
    Bz[:,:,1:-1] -= dt * psi_z

    # === STAGGERED J×B ===
    Jx, Jy, Jz = compute_staggered_JxB()
    mx += dt * Jx
    my += dt * Jy
    mz += dt * Jz

    if step % 50 == 0:
        div_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        
        Bx_c = 0.5 * (Bx[1:,:,:] + Bx[:-1,:,:])
        By_c = 0.5 * (By[:,1:,:] + By[:,:-1,:])
        Bz_c = 0.5 * (Bz[:,:,1:] + Bz[:,:,:-1])
        Bmag = cp.sqrt(Bx_c**2 + By_c**2 + Bz_c**2)
        Bmax = float(cp.nanmax(Bmag))
        
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB_max = {div_max:.2e} | JxB active")

print("\n✅ v40.9 Finished! J×B is now in action.")

# Quick JxB magnitude print at end
Jx_final, Jy_final, Jz_final = compute_staggered_JxB()
print(f"Final avg |J×B| ≈ {float(cp.mean(cp.sqrt(Jx_final**2 + Jy_final**2 + Jz_final**2))):.2e}")
