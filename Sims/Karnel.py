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
E_total = cp.ones((N, N, N), dtype=cp.float32) * 3.0

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)

np.random.seed(42)
pert = 0.08
Bx[1:-1] = cp.asarray(np.random.randn(N-1, N, N) * pert, dtype=cp.float32)
By[:,1:-1] = cp.asarray(np.random.randn(N, N-1, N) * pert, dtype=cp.float32)
Bz[:-1,:-1] = cp.asarray(np.random.randn(N-1, N-1, N+1) * pert * 0.5 + 0.5, dtype=cp.float32)

print("✅ Data initialized")

# ====================== SAFER RAW KERNEL ======================
raw_emf_kernel = cp.RawKernel(r'''
extern "C" __global__ void raw_emf_kernel(
    const float* vx, const float* vy, const float* vz,
    const float* Bx, const float* By, const float* Bz,
    float* Emfx, float* Emfy, float* Emfz,
    int N, float dx)
{
    extern __shared__ float sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    if (i >= N-1 || j >= N-1 || k >= N-1) return;

    int txs = blockDim.x + 2;
    int tys = blockDim.y + 2;

    float* s_vx = sdata;
    float* s_vy = sdata + txs * tys * (blockDim.z + 2);
    float* s_vz = sdata + 2 * txs * tys * (blockDim.z + 2);

    int sidx = (tz * tys + ty) * txs + tx;

    // Safe main load
    s_vx[sidx] = vx[i*N*N + j*N + k];
    s_vy[sidx] = vy[i*N*N + j*N + k];
    s_vz[sidx] = vz[i*N*N + j*N + k];

    // Safe halo
    if (tx == 0 && i > 0) {
        s_vx[sidx-1] = vx[(i-1)*N*N + j*N + k];
        s_vy[sidx-1] = vy[(i-1)*N*N + j*N + k];
        s_vz[sidx-1] = vz[(i-1)*N*N + j*N + k];
    }
    if (ty == 0 && j > 0) {
        s_vx[sidx - txs] = vx[i*N*N + (j-1)*N + k];
        s_vy[sidx - txs] = vy[i*N*N + (j-1)*N + k];
        s_vz[sidx - txs] = vz[i*N*N + (j-1)*N + k];
    }

    __syncthreads();

    // ==================== EMF_z ====================
    if (i < N-1 && j < N-1 && k < N) {
        float vx_edge = 0.25f * (s_vx[sidx] + s_vx[sidx+1] + s_vx[sidx+txs] + s_vx[sidx+txs+1]);
        float vy_edge = 0.25f * (s_vy[sidx] + s_vy[sidx+1] + s_vy[sidx+txs] + s_vy[sidx+txs+1]);

        float Bx_e = 0.5f * (Bx[(i+1)*N*N + j*N + k] + Bx[(i+1)*N*N + (j+1)*N + k]);
        float By_e = 0.5f * (By[i*N*N + (j+1)*N + k] + By[(i+1)*N*N + (j+1)*N + k]);

        Emfz[i*N*N + j*N + k] = -(vx_edge * By_e - vy_edge * Bx_e);
    }

    // ==================== EMF_y ====================
    if (i < N-1 && k < N-1) {
        float vx_edge = 0.25f * (s_vx[sidx] + s_vx[sidx+1] + s_vx[sidx + txs*tys] + s_vx[sidx + txs*tys + 1]);
        float vz_edge = 0.25f * (s_vz[sidx] + s_vz[sidx+1] + s_vz[sidx + txs*tys] + s_vz[sidx + txs*tys + 1]);

        float Bx_e = 0.5f * (Bx[(i+1)*N*N + j*N + k] + Bx[(i+1)*N*N + j*N + (k+1)]);
        float Bz_e = 0.5f * (Bz[i*N*N + j*N + (k+1)] + Bz[(i+1)*N*N + j*N + (k+1)]);

        Emfy[i*N*N + j*N + k] = -(vz_edge * Bx_e - vx_edge * Bz_e);
    }

    // ==================== EMF_x ====================
    if (j < N-1 && k < N-1) {
        float vy_edge = 0.25f * (s_vy[sidx] + s_vy[sidx + txs] + s_vy[sidx + txs*tys] + s_vy[sidx + txs*tys + txs]);
        float vz_edge = 0.25f * (s_vz[sidx] + s_vz[sidx + txs] + s_vz[sidx + txs*tys] + s_vz[sidx + txs*tys + txs]);

        float By_e = 0.5f * (By[i*N*N + (j+1)*N + k] + By[i*N*N + (j+1)*N + (k+1)]);
        float Bz_e = 0.5f * (Bz[i*N*N + j*N + (k+1)] + Bz[i*N*N + (j+1)*N + (k+1)]);

        Emfx[i*N*N + j*N + k] = -(vy_edge * Bz_e - vz_edge * By_e);
    }
}
''', 'raw_emf_kernel')

print("✅ Full 3-edge Raw Kernel updated")

# ====================== LAUNCH ======================
block = (16, 16, 4)
grid = ((N + 15)//16, (N + 15)//16, (N + 3)//4)

shared_bytes = 3 * (block[0]+2) * (block[1]+2) * (block[2]+2) * 4

Emfx = cp.zeros((N, N+1, N+1), dtype=cp.float32)
Emfy = cp.zeros((N+1, N, N+1), dtype=cp.float32)
Emfz = cp.zeros((N+1, N+1, N), dtype=cp.float32)

raw_emf_kernel(grid, block, 
               (mx/rho, my/rho, mz/rho, Bx, By, Bz, Emfx, Emfy, Emfz, N, dx),
               shared_mem=shared_bytes)

print("✅ Kernel launched successfully!")
print(f"Emfz max = {float(cp.max(cp.abs(Emfz))):.6f}")
print(f"Emfy max = {float(cp.max(cp.abs(Emfy))):.6f}")
print(f"Emfx max = {float(cp.max(cp.abs(Emfx))):.6f}")
