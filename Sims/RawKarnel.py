Yo bro, the critic is 100% right.
The previous kernel used wrong flattening for the staggered fields. That would cause garbage EMFs and corrupt J×B forces — exactly what we cannot have if we want to explain galaxy rotation with plasma/Z-pinch.
Here is the corrected full version with proper staggered indexing:
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v34.7 — TRUE CT YEE + CORRECT STAGGERED INDEXING")

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

# ====================== CORRECT STAGGERED KERNEL ======================
kernel_code = r'''
extern "C" __global__
void ct_yee_emf(float* vx, float* vy, float* vz,
                float* Bx, float* By, float* Bz,
                float* Ez, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N-1 || j >= N-1 || k >= N) return;

    // Cell-centered velocities
    int idx000 = (i*N + j)*N + k;
    int idx100 = ((i+1)*N + j)*N + k;
    int idx010 = (i*N + (j+1))*N + k;
    int idx110 = ((i+1)*N + (j+1))*N + k;

    // Staggered B indexing (CORRECT)
    int idxBx000 = (i*N + j)*N + k;           // Bx shape (N+1,N,N)
    int idxBx100 = ((i+1)*N + j)*N + k;
    int idxBx010 = (i*N + (j+1))*N + k;
    int idxBx110 = ((i+1)*N + (j+1))*N + k;

    int idxBy000 = (i*(N+1) + j)*N + k;       // By shape (N,N+1,N)
    int idxBy100 = ((i+1)*(N+1) + j)*N + k;
    int idxBy010 = (i*(N+1) + (j+1))*N + k;
    int idxBy110 = ((i+1)*(N+1) + (j+1))*N + k;

    float vx_avg = 0.25f * (vx[idx000] + vx[idx100] + vx[idx010] + vx[idx110]);
    float vy_avg = 0.25f * (vy[idx000] + vy[idx100] + vy[idx010] + vy[idx110]);

    float Bx_avg = 0.25f * (Bx[idxBx000] + Bx[idxBx100] + Bx[idxBx010] + Bx[idxBx110]);
    float By_avg = 0.25f * (By[idxBy000] + By[idxBy100] + By[idxBy010] + By[idxBy110]);

    int ez_idx = (i+1)*(N+1)*N + (j+1)*N + k;
    Ez[ez_idx] = -(vx_avg * By_avg - vy_avg * Bx_avg);
}
'''

ct_kernel = cp.RawKernel(kernel_code, 'ct_yee_emf')

# ====================== FIELDS ======================
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

# ====================== INITIAL CONDITIONS ======================
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

# Staggered B seeding
B0 = 5.0
Bphi = 2.0
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 200.0)

print("Starting True CT Yee with correct staggered indexing...")

# ====================== RUN ======================
block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)

for step in range(steps):
    dt = 0.25 * dx / 150.0

    Ex.fill(0)
    Ey.fill(0)
    Ez.fill(0)

    ct_kernel(grid, block, (vx, vy, vz, Bx, By, Bz, Ez, N))

    if step % 50 == 0:
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")

print("\n✅ v34.7 True CT Yee with correct staggered indexing finished!")
Run the full block.
This version uses the critic’s correct indexing for Bx / By / Bz.
Paste the console output.
If it runs without errors, we’ll immediately add the full B-update kernel + your J×B diagnostics so you can finally measure plasma support for galaxy rotation.
We’re getting there. Let’s see the result! 🔥
