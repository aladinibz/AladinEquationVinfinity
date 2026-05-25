import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v34.0 — Fixed CUDA Kernel for Staggered CT Yee")

# ====================== FIXED CUDA KERNEL ======================
kernel_code = r'''
extern "C" __global__
void ct_yee_update(float* Bx, float* By, float* Bz,
                   float* Ex, float* Ey, float* Ez,
                   float dt, float dx, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i < N && j >= 1 && j < N && k >= 1 && k < N) {
        // Update Bx (x-faces)
        float dEz_dy = (Ez[i * (N+1)*(N+1) + (j+1)*(N+1) + k] - Ez[i * (N+1)*(N+1) + j*(N+1) + k]) / dx;
        float dEy_dz = (Ey[i * (N+1)*(N+1) + j*(N+1) + (k+1)] - Ey[i * (N+1)*(N+1) + j*(N+1) + k]) / dx;
        Bx[i * N * N + j * N + k] += dt * (dEz_dy - dEy_dz);

        // Update By (y-faces)
        float dEx_dz = (Ex[(i) * (N+1)*(N+1) + j*(N+1) + (k+1)] - Ex[(i) * (N+1)*(N+1) + j*(N+1) + k]) / dx;
        float dEz_dx = (Ez[(i+1) * (N+1)*(N+1) + j*(N+1) + k] - Ez[i * (N+1)*(N+1) + j*(N+1) + k]) / dx;
        By[i * N * N + j * N + k] += dt * (dEx_dz - dEz_dx);

        // Update Bz (z-faces)
        float dEy_dx = (Ey[(i+1) * (N+1)*(N+1) + j*(N+1) + k] - Ey[i * (N+1)*(N+1) + j*(N+1) + k]) / dx;
        float dEx_dy = (Ex[i * (N+1)*(N+1) + (j+1)*(N+1) + k] - Ex[i * (N+1)*(N+1) + j*(N+1) + k]) / dx;
        Bz[i * N * N + j * N + k] += dt * (dEy_dx - dEx_dy);
    }
}
'''

ct_kernel = cp.RawKernel(kernel_code, 'ct_yee_update')

# ====================== FIELDS ======================
N = 128
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

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

# Initial B seeding
r_cyl = cp.sqrt(X**2 + Y**2)
Bx[1:,:,:] = -2.0 * (Y[1:,:,:] / (r_cyl[1:,:,:] + 1e-8))
By[:,1:,:] =  2.0 * (X[:,1:,:] / (r_cyl[:,1:,:] + 1e-8))
Bz[:,:,1:] = 5.0 * cp.exp(-(X[:,:,1:]**2 + Y[:,:,1:]**2 + Z[:,:,1:]**2) / 200.0)

# ====================== MAIN LOOP ======================
for step in range(300):
    dt = 0.25 * dx / 150.0

    # Fill EMFs (simplified - replace with full v x B later)
    Ex.fill(0.01)
    Ey.fill(0.01)
    Ez.fill(0.01)

    # Launch CUDA kernel
    blocks = (N//8, N//8, N//8)
    threads = (8, 8, 8)
    ct_kernel(blocks, threads, (Bx, By, Bz, Ex, Ey, Ez, dt, dx, N))

    if step % 50 == 0:
        Bmax = float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG")

print("\n✅ CUDA Staggered CT Yee kernel finished.")
