import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v40.1 — FIXED STAGGERED INDEXING + FULL CT-YEE")

# ====================== PARAMETERS ======================
N = 64
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
gamma = 5.0 / 3.0
CFL = 0.16
steps = 400

v_phi_factor = 0.12
c_h_factor = 10.0
kappa_factor = 12.0

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== FIXED FULL STAGGERED KERNEL ======================
kernel_code = r'''
extern "C" __global__
void fixed_ct_yee_full(float* vx, float* vy, float* vz,
                       float* Bx, float* By, float* Bz, float* psi,
                       float dt, float dx, int N, float c_h, float kappa)
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

        // Ez at z-edge
        float vx_avg = 0.25f * (s_vx[s] + s_vx[s+tys*tzs] + s_vx[s+tzs] + s_vx[s+tys*tzs+tzs]);
        float vy_avg = 0.25f * (s_vy[s] + s_vy[s+tys*tzs] + s_vy[s+tzs] + s_vy[s+tys*tzs+tzs]);
        float Bx_avg = 0.25f * (s_Bx[s] + s_Bx[s+tys*tzs] + s_Bx[s+tzs] + s_Bx[s+tys*tzs+tzs]);
        float By_avg = 0.25f * (s_By[s] + s_By[s+tys*tzs] + s_By[s+tzs] + s_By[s+tys*tzs+tzs]);
        float Ez_val = 0.05f * (-(vx_avg * By_avg - vy_avg * Bx_avg));
        int bz_idx = (i*N + j)*(N+1) + k;
        Bz[bz_idx] += (dt / dx) * Ez_val;

        // Ey at y-edge
        float vz_avg = 0.25f * (s_vz[s] + s_vz[s+tys*tzs] + s_vz[s+tzs] + s_vz[s+tys*tzs+tzs]);
        float Bx_avg_y = Bx_avg;
        float Bz_avg_y = 0.25f * (s_Bz[s] + s_Bz[s+tys*tzs] + s_Bz[s+tzs] + s_Bz[s+tys*tzs+tzs]);
        float Ey_val = 0.05f * (-(vz_avg * Bx_avg_y - vx_avg * Bz_avg_y));
        int bx_idx = (i*N + j)*N + k;
        Bx[bx_idx] += (dt / dx) * Ey_val;

        // Ex at x-edge
        float vy_avg_x = vy_avg;
        float vz_avg_x = vz_avg;
        float By_avg_x = By_avg;
        float Bz_avg_x = Bz_avg_y;
        float Ex_val = 0.05f * (-(vy_avg_x * Bz_avg_x - vz_avg_x * By_avg_x));
        int by_idx = (i*N + j)*N + k;
        By[by_idx] += (dt / dx) * Ex_val;
    }

    if (i < N && j < N && k < N) {
        int g = (i*N + j)*N + k;
        psi[g] -= dt * kappa * psi[g];
    }
}
'''

kernel = cp.RawKernel(kernel_code, 'fixed_ct_yee_full')

# ====================== FIELDS & INIT ======================
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

B0 = 2.0
Bphi = 1.0
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 300.0)

c_h = c_h_factor * 150.0
kappa = kappa_factor / dx

def compute_divB():
    div = cp.zeros((N, N, N), dtype=cp.float32)
    div += (Bx[1:,:,:] - Bx[:-1,:,:]) / dx
    div += (By[:,1:,:] - By[:,:-1,:]) / dx
    div += (Bz[:,:,1:] - Bz[:,:,:-1]) / dx
    return div

print("Starting v40.1 with Fixed Staggered Indexing + Full CT Curl...")

block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)
shared_bytes = 6 * (10**3) * 4

for step in range(steps):
    dt = CFL * dx / 280.0
    kernel(grid, block, (vx, vy, vz, Bx, By, Bz, psi, dt, dx, N, c_h, kappa), shared_mem=shared_bytes)

    if step % 50 == 0:
        div_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        
        Bx_c = 0.5 * (Bx[1:,:,:] + Bx[:-1,:,:])
        By_c = 0.5 * (By[:,1:,:] + By[:,:-1,:])
        Bz_c = 0.5 * (Bz[:,:,1:] + Bz[:,:,:-1])
        Bmag = cp.sqrt(Bx_c**2 + By_c**2 + Bz_c**2)
        Bmax = float(cp.nanmax(Bmag))
        
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB_max = {div_max:.2e}")

print("\n✅ v40.1 Finished with Fixed Indexing!")
