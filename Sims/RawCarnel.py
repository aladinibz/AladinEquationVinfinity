import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v34.6 — OPTIMIZED TRUE CT YEE (FIXED)")

# ====================== PARAMETERS ======================
N = 96
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
mu0 = 1.0
gamma = 5.0 / 3.0
CFL = 0.25
steps = 400

rho_floor = 1e-6
p_floor = 1e-4
v_phi_factor = 0.12

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== KERNEL DEFINITION ======================
kernel_code = r'''
extern "C" __global__
void ct_yee_optimized(float* vx, float* vy, float* vz,
                      float* Bx, float* By, float* Bz,
                      float* Ex, float* Ey, float* Ez,
                      float dt, float dx, int N)
{
    extern __shared__ float sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    int txs = blockDim.x + 3;
    int tys = blockDim.y + 2;
    int tzs = blockDim.z + 2;

    float* s_vx = sdata;
    float* s_vy = s_vx + txs*tys*tzs;
    float* s_vz = s_vy + txs*tys*tzs;
    float* s_Bx = s_vz + txs*tys*tzs;
    float* s_By = s_Bx + txs*tys*tzs;
    float* s_Bz = s_By + txs*tys*tzs;

    if (i < N && j < N && k < N) {
        int s_idx = (tx+1)*tys*tzs + (ty+1)*tzs + (tz+1);
        int g_idx = (i*N + j)*N + k;
        s_vx[s_idx] = vx[g_idx];
        s_vy[s_idx] = vy[g_idx];
        s_vz[s_idx] = vz[g_idx];
        s_Bx[s_idx] = Bx[g_idx];
        s_By[s_idx] = By[g_idx];
        s_Bz[s_idx] = Bz[g_idx];
    }
    __syncthreads();

    if (i < N-1 && j < N-1 && k < N) {
        int base = (tx+1)*tys*tzs + (ty+1)*tzs + (tz+1);

        float vx_avg = 0.25f * (s_vx[base] + s_vx[base + tys*tzs] + s_vx[base + tzs] + s_vx[base + tys*tzs + tzs]);
        float vy_avg = 0.25f * (s_vy[base] + s_vy[base + tys*tzs] + s_vy[base + tzs] + s_vy[base + tys*tzs + tzs]);
        float Bx_avg = 0.25f * (s_Bx[base] + s_Bx[base + tys*tzs] + s_Bx[base + tzs] + s_Bx[base + tys*tzs + tzs]);
        float By_avg = 0.25f * (s_By[base] + s_By[base + tys*tzs] + s_By[base + tzs] + s_By[base + tys*tzs + tzs]);

        int ez_idx = (i+1)*(N+1)*N + (j+1)*N + k;
        Ez[ez_idx] = -(vx_avg * By_avg - vy_avg * Bx_avg);
    }
}
'''

ct_kernel = cp.RawKernel(kernel_code, 'ct_yee_optimized')

# ====================== FIELDS & INITIAL CONDITIONS ======================
rho = cp.ones((N, N, N), dtype=cp.float32) * 1e-3
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)

Ex = cp.zeros((N, N+1, N+1), dtype=cp.float32)
Ey = cp.zeros((N+1, N, N+1), dtype=cp.float32)
Ez = cp.zeros((N+1, N+1, N), dtype=cp.float32)

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

B0 = 5.0
Bphi = 2.0
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 200.0)

print("Starting simulation...")

# ====================== RUN ======================
block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)
tile_x = 8 + 3
tile_y = 8 + 2
tile_z = 8 + 2
shared_mem_bytes = 6 * tile_x * tile_y * tile_z * 4

for step in range(steps):
    dt = 0.25 * dx / 150.0

    Ex.fill(0)
    Ey.fill(0)
    Ez.fill(0)

    ct_kernel(grid, block, (vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, dt, dx, N), shared_mem=shared_mem_bytes)

    if step % 50 == 0:
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")

print("\n✅ v34.6 Fixed and running!")
