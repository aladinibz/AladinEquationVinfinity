import numpy as np
import cupy as cp

# ====================== PARAMETERS ======================
N = 256
L = 1.0
dx = L / N

# ====================== DATA ======================
rho = cp.ones((N, N, N), dtype=cp.float32)
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)

np.random.seed(42)
pert = 0.08
Bx[1:-1] = cp.asarray(np.random.randn(N-1, N, N) * pert, dtype=cp.float32)
By[:,1:-1] = cp.asarray(np.random.randn(N, N-1, N) * pert, dtype=cp.float32)
Bz[:-1,:-1] = cp.asarray(np.random.randn(N-1, N-1, N) * pert * 0.5 + 0.5, dtype=cp.float32)  # Fixed shape

print("✅ Data initialized")

# ====================== MINIMAL DEBUG KERNEL ======================
debug_kernel = cp.RawKernel(r'''
extern "C" __global__ void debug_emf_kernel(
    const float* vx, const float* vy,
    const float* Bx, const float* By,
    float* Emfz, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    // Very strict bounds
    if (i >= N-1 || j >= N-1 || k >= N) return;

    // Simple 4-neighbor without heavy halo for debugging
    float vx1 = vx[i*N*N + j*N + k];
    float vx2 = (i+1 < N) ? vx[(i+1)*N*N + j*N + k] : vx1;
    float vx3 = (j+1 < N) ? vx[i*N*N + (j+1)*N + k] : vx1;
    float vx4 = (i+1 < N && j+1 < N) ? vx[(i+1)*N*N + (j+1)*N + k] : vx1;

    float vy1 = vy[i*N*N + j*N + k];
    float vy2 = (i+1 < N) ? vy[(i+1)*N*N + j*N + k] : vy1;
    float vy3 = (j+1 < N) ? vy[i*N*N + (j+1)*N + k] : vy1;
    float vy4 = (i+1 < N && j+1 < N) ? vy[(i+1)*N*N + (j+1)*N + k] : vy1;

    float vx_edge = 0.25f * (vx1 + vx2 + vx3 + vx4);
    float vy_edge = 0.25f * (vy1 + vy2 + vy3 + vy4);

    float Bx_e = 0.5f * (Bx[(i+1)*N*N + j*N + k] + Bx[(i+1)*N*N + (j+1)*N + k]);
    float By_e = 0.5f * (By[i*N*N + (j+1)*N + k] + By[(i+1)*N*N + (j+1)*N + k]);

    Emfz[i*N*N + j*N + k] = -(vx_edge * By_e - vy_edge * Bx_e);
}
''', 'debug_emf_kernel')

print("✅ Minimal debug kernel ready")

# ====================== LAUNCH (Conservative) ======================
block = (8, 8, 8)                     # Small and safe
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)

Emfz = cp.zeros((N+1, N+1, N), dtype=cp.float32)

debug_emf_kernel(grid, block, (mx/rho, my/rho, Bx, By, Emfz, N))

print("✅ Kernel launched!")
print(f"Emfz max abs = {float(cp.max(cp.abs(Emfz))):.6f}")
