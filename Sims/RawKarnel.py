import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v35.0 — TRUE CT YEE + FULL Ex/Ey/Ez EMFs")

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

v_phi_factor = 0.12

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== KERNEL WITH FULL Ex/Ey/Ez ======================
kernel_code = r'''
extern "C" __global__
void ct_yee_emf_full(float* vx, float* vy, float* vz,
                     float* Bx, float* By, float* Bz,
                     float* Ex, float* Ey, float* Ez, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N-1 || j >= N-1 || k >= N) return;

    // Cell-centered velocities (common)
    int c000 = (i*N + j)*N + k;
    int c100 = ((i+1)*N + j)*N + k;
    int c010 = (i*N + (j+1))*N + k;
    int c110 = ((i+1)*N + (j+1))*N + k;

    // ====================== Ez (z-edge) ======================
    if (i < N-1 && j < N-1) {
        int bx000 = (i*N + j)*N + k;
        int bx100 = ((i+1)*N + j)*N + k;
        int bx010 = (i*N + (j+1))*N + k;
        int bx110 = ((i+1)*N + (j+1))*N + k;

        int by000 = (i*(N+1) + j)*N + k;
        int by100 = ((i+1)*(N+1) + j)*N + k;
        int by010 = (i*(N+1) + (j+1))*N + k;
        int by110 = ((i+1)*(N+1) + (j+1))*N + k;

        float vx_avg = 0.25f * (vx[c000] + vx[c100] + vx[c010] + vx[c110]);
        float vy_avg = 0.25f * (vy[c000] + vy[c100] + vy[c010] + vy[c110]);

        float Bx_avg = 0.25f * (Bx[bx000] + Bx[bx100] + Bx[bx010] + Bx[bx110]);
        float By_avg = 0.25f * (By[by000] + By[by100] + By[by010] + By[by110]);

        int ez_idx = (i+1)*(N+1)*N + (j+1)*N + k;
        Ez[ez_idx] = -(vx_avg * By_avg - vy_avg * Bx_avg);
    }

    // ====================== Ey (y-edge) ======================
    if (i < N-1 && k < N-1) {
        int c0k0 = (i*N + j)*N + k;
        int c1k0 = ((i+1)*N + j)*N + k;
        int c0k1 = (i*N + j)*N + (k+1);
        int c1k1 = ((i+1)*N + j)*N + (k+1);

        int bx0k0 = (i*N + j)*N + k;
        int bx1k0 = ((i+1)*N + j)*N + k;
        int bx0k1 = (i*N + j)*N + (k+1);
        int bx1k1 = ((i+1)*N + j)*N + (k+1);

        int bz0k0 = (i*N + j)*(N+1) + k;
        int bz1k0 = ((i+1)*N + j)*(N+1) + k;
        int bz0k1 = (i*N + j)*(N+1) + (k+1);
        int bz1k1 = ((i+1)*N + j)*(N+1) + (k+1);

        float vx_avg = 0.25f * (vx[c0k0] + vx[c1k0] + vx[c0k1] + vx[c1k1]);
        float vz_avg = 0.25f * (vz[c0k0] + vz[c1k0] + vz[c0k1] + vz[c1k1]);

        float Bx_avg = 0.25f * (Bx[bx0k0] + Bx[bx1k0] + Bx[bx0k1] + Bx[bx1k1]);
        float Bz_avg = 0.25f * (Bz[bz0k0] + Bz[bz1k0] + Bz[bz0k1] + Bz[bz1k1]);

        int ey_idx = (i+1)*N*(N+1) + j*(N+1) + (k+1);
        Ey[ey_idx] = -(vz_avg * Bx_avg - vx_avg * Bz_avg);
    }

    // ====================== Ex (x-edge) - FULL ======================
    if (j < N-1 && k < N-1) {
        int c0j0 = (i*N + j)*N + k;
        int c0j1 = (i*N + j)*N + (k+1);
        int c1j0 = (i*N + (j+1))*N + k;   // note: for Ex we use y-neighbors
        int c1j1 = (i*N + (j+1))*N + (k+1);

        int by0j0 = (i*(N+1) + j)*N + k;
        int by0j1 = (i*(N+1) + j)*N + (k+1);
        int by1j0 = (i*(N+1) + (j+1))*N + k;
        int by1j1 = (i*(N+1) + (j+1))*N + (k+1);

        int bz0j0 = (i*N + j)*(N+1) + k;
        int bz0j1 = (i*N + j)*(N+1) + (k+1);
        int bz1j0 = (i*N + (j+1))*(N+1) + k;
        int bz1j1 = (i*N + (j+1))*(N+1) + (k+1);

        float vy_avg = 0.25f * (vy[c0j0] + vy[c0j1] + vy[c1j0] + vy[c1j1]);
        float vz_avg = 0.25f * (vz[c0j0] + vz[c0j1] + vz[c1j0] + vz[c1j1]);

        float By_avg = 0.25f * (By[by0j0] + By[by0j1] + By[by1j0] + By[by1j1]);
        float Bz_avg = 0.25f * (Bz[bz0j0] + Bz[bz0j1] + Bz[bz1j0] + Bz[bz1j1]);

        int ex_idx = i*(N+1)*(N+1) + (j+1)*(N+1) + (k+1);
        Ex[ex_idx] = -(vy_avg * Bz_avg - vz_avg * By_avg);
    }
}
'''

ct_kernel = cp.RawKernel(kernel_code, 'ct_yee_emf_full')

# ====================== FIELDS & INIT ======================
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

print("Starting simulation with full Ex/Ey/Ez...")

# ====================== RUN ======================
block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)

for step in range(steps):
    dt = 0.25 * dx / 150.0

    Ex.fill(0)
    Ey.fill(0)
    Ez.fill(0)

    ct_kernel(grid, block, (vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, N))

    if step % 50 == 0:
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")

print("\n✅ v35.0 Full Ex/Ey/Ez EMFs with correct staggered indexing finished!")
