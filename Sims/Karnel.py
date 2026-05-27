import numpy as np
import cupy as cp

# ====================== PARAMETERS ======================
N = 128
L = 1.0
dx = L / N
dt = 0.00015
max_steps = 1000
print_interval = 50
NG = 3
Ni = N + 2 * NG
gamma = 5.0 / 3.0

# ====================== FIELDS ======================
rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
my = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
mz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
E_total = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 3.0

Bx = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
By = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
Bz = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)

Bx_new = cp.zeros_like(Bx)
By_new = cp.zeros_like(By)
Bz_new = cp.zeros_like(Bz)

Emfx = cp.zeros((Ni, Ni+1, Ni+1), dtype=cp.float32)
Emfy = cp.zeros((Ni+1, Ni, Ni+1), dtype=cp.float32)
Emfz = cp.zeros((Ni+1, Ni+1, Ni), dtype=cp.float32)

np.random.seed(42)
pert = 0.1
Bx[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
By[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
Bz[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert * 0.5 + 0.5, dtype=cp.float32)

def update_ghosts():
    cons = [rho, mx, my, mz, E_total]
    for f in cons:
        f[0:NG,:,:] = f[Ni-NG:Ni,:,:]
        f[Ni:,:,:] = f[NG:2*NG,:,:]
    for f in [Bx, By, Bz]:
        f[0:NG,:,:] = f[Ni-NG:Ni,:,:]
        f[Ni:,:,:] = f[NG:2*NG,:,:]

# ====================== PURE TRUE CT YEE + JxB KERNEL ======================
kernel = cp.RawKernel(r'''
extern "C" __global__ void jxb_ct_yee(
    const float* rho, const float* mx, const float* my, const float* mz, const float* E_total,
    const float* Bx_old, const float* By_old, const float* Bz_old,
    float* mx_new, float* my_new, float* mz_new, float* E_new,
    float* Bx_new, float* By_new, float* Bz_new,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, int NG, float dt_over_dx)
{
    extern __shared__ float sdata[];
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx + NG;
    int j = blockIdx.y * blockDim.y + ty + NG;
    int k = blockIdx.z * blockDim.z + tz + NG;

    if (i >= Ni-NG || j >= Ni-NG || k >= Ni-NG) return;

    int txs = blockDim.x + 2;
    int tys = blockDim.y + 2;
    int tzs = blockDim.z + 2;
    int sidx = (tz + 1) * tys * txs + (ty + 1) * txs + (tx + 1);

    float* s_vx = sdata;
    float* s_vy = sdata + txs*tys*tzs;
    float* s_vz = sdata + 2*txs*tys*tzs;

    s_vx[sidx] = mx[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];
    s_vy[sidx] = my[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];
    s_vz[sidx] = mz[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];

    __syncthreads();

    // Pure True CT Yee EMFs
    if (i < Ni-1 && j < Ni-1 && k < Ni) {
        float vx_e = 0.25f * (s_vx[sidx] + s_vx[sidx+1] + s_vx[sidx+txs] + s_vx[sidx+txs+1]);
        float vy_e = 0.25f * (s_vy[sidx] + s_vy[sidx+1] + s_vy[sidx+txs] + s_vy[sidx+txs+1]);
        float Bx_e = 0.5f * (Bx_old[(i+1)*Ni*Ni + j*Ni + k] + Bx_old[(i+1)*Ni*Ni + (j+1)*Ni + k]);
        float By_e = 0.5f * (By_old[i*Ni*Ni + (j+1)*Ni + k] + By_old[(i+1)*Ni*Ni + (j+1)*Ni + k]);
        Emfz[i*Ni*Ni + j*Ni + k] = -(vx_e * By_e - vy_e * Bx_e);
    }

    if (i < Ni-1 && k < Ni-1) {
        float vx_e = 0.25f * (s_vx[sidx] + s_vx[sidx+1] + s_vx[sidx+txs*tys] + s_vx[sidx+txs*tys+1]);
        float vz_e = 0.25f * (s_vz[sidx] + s_vz[sidx+1] + s_vz[sidx+txs*tys] + s_vz[sidx+txs*tys+1]);
        float Bx_e = 0.5f * (Bx_old[(i+1)*Ni*Ni + j*Ni + k] + Bx_old[(i+1)*Ni*Ni + j*Ni + (k+1)]);
        float Bz_e = 0.5f * (Bz_old[i*Ni*Ni + j*Ni + (k+1)] + Bz_old[(i+1)*Ni*Ni + j*Ni + (k+1)]);
        Emfy[i*Ni*Ni + j*Ni + k] = -(vz_e * Bx_e - vx_e * Bz_e);
    }

    if (j < Ni-1 && k < Ni-1) {
        float vy_e = 0.25f * (s_vy[sidx] + s_vy[sidx+txs] + s_vy[sidx+txs*tys] + s_vy[sidx+txs*tys+txs]);
        float vz_e = 0.25f * (s_vz[sidx] + s_vz[sidx+txs] + s_vz[sidx+txs*tys] + s_vz[sidx+txs*tys+txs]);
        float By_e = 0.5f * (By_old[i*Ni*Ni + (j+1)*Ni + k] + By_old[i*Ni*Ni + (j+1)*Ni + (k+1)]);
        float Bz_e = 0.5f * (Bz_old[i*Ni*Ni + j*Ni + (k+1)] + Bz_old[i*Ni*Ni + (j+1)*Ni + (k+1)]);
        Emfx[i*Ni*Ni + j*Ni + k] = -(vy_e * Bz_e - vz_e * By_e);
    }

    __syncthreads();

    // Pure True CT Yee curl
    if (i < Ni && j < Ni && k < Ni) {
        float dEz_dy = Emfz[i*Ni*Ni + (j+1)*Ni + k] - Emfz[i*Ni*Ni + j*Ni + k];
        float dEy_dz = Emfy[i*Ni*Ni + j*Ni + (k+1)] - Emfy[i*Ni*Ni + j*Ni + k];
        Bx_new[i*Ni*Ni + j*Ni + k] = Bx_old[i*Ni*Ni + j*Ni + k] - dt_over_dx * (dEz_dy - dEy_dz);

        float dEx_dz = Emfx[i*Ni*Ni + j*Ni + (k+1)] - Emfx[i*Ni*Ni + j*Ni + k];
        float dEz_dx = Emfz[(i+1)*Ni*Ni + j*Ni + k] - Emfz[i*Ni*Ni + j*Ni + k];
        By_new[i*Ni*Ni + j*Ni + k] = By_old[i*Ni*Ni + j*Ni + k] - dt_over_dx * (dEx_dz - dEz_dx);

        float dEy_dx = Emfy[(i+1)*Ni*Ni + j*Ni + k] - Emfy[i*Ni*Ni + j*Ni + k];
        float dEx_dy = Emfx[i*Ni*Ni + (j+1)*Ni + k] - Emfx[i*Ni*Ni + j*Ni + k];
        Bz_new[i*Ni*Ni + j*Ni + k] = Bz_old[i*Ni*Ni + j*Ni + k] - dt_over_dx * (dEy_dx - dEx_dy);
    }

    // J x B Lorentz force (your engine!)
    if (i < Ni && j < Ni && k < Ni) {
        float Jx = (Bz_old[i*Ni*Ni + j*Ni + (k+1)] - Bz_old[i*Ni*Ni + j*Ni + k]) / dx 
                 - (By_old[i*Ni*Ni + (j+1)*Ni + k] - By_old[i*Ni*Ni + j*Ni + k]) / dx;
        float Jy = (Bx_old[(i+1)*Ni*Ni + j*Ni + k] - Bx_old[i*Ni*Ni + j*Ni + k]) / dx 
                 - (Bz_old[i*Ni*Ni + j*Ni + (k+1)] - Bz_old[i*Ni*Ni + j*Ni + k]) / dx;
        float Jz = (By_old[i*Ni*Ni + (j+1)*Ni + k] - By_old[i*Ni*Ni + j*Ni + k]) / dx 
                 - (Bx_old[(i+1)*Ni*Ni + j*Ni + k] - Bx_old[i*Ni*Ni + j*Ni + k]) / dx;

        float vx = mx[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];
        float vy = my[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];
        float vz = mz[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];

        float jxb_x = Jy * Bz_old[i*Ni*Ni + j*Ni + k] - Jz * By_old[i*Ni*Ni + j*Ni + k];
        float jxb_y = Jz * Bx_old[i*Ni*Ni + j*Ni + k] - Jx * Bz_old[i*Ni*Ni + j*Ni + k];
        float jxb_z = Jx * By_old[i*Ni*Ni + j*Ni + k] - Jy * Bx_old[i*Ni*Ni + j*Ni + k];

        mx_new[i*Ni*Ni + j*Ni + k] = mx[i*Ni*Ni + j*Ni + k] + dt_over_dx * jxb_x;
        my_new[i*Ni*Ni + j*Ni + k] = my[i*Ni*Ni + j*Ni + k] + dt_over_dx * jxb_y;
        mz_new[i*Ni*Ni + j*Ni + k] = mz[i*Ni*Ni + j*Ni + k] + dt_over_dx * jxb_z;

        // Energy work term v · (J × B)
        float work = vx * jxb_x + vy * jxb_y + vz * jxb_z;
        E_new[i*Ni*Ni + j*Ni + k] = E_total[i*Ni*Ni + j*Ni + k] + dt_over_dx * work;
    }
}
''', 'jxb_ct_yee')

print("✅ Pure True CT Yee + Full J×B Lorentz Force + Energy Work Term ready!")

# ====================== LAUNCH LOOP ======================
block = (16, 16, 4)
grid = ((N + 15)//16, (N + 15)//16, (N + 3)//4)
shared_bytes = 3 * (block[0]+2) * (block[1]+2) * (block[2]+2) * 4

steps = 0
while steps < max_steps:
    update_ghosts()

    kernel(grid, block, (rho, mx, my, mz, E_total, Bx, By, Bz,
                         mx_new, my_new, mz_new, E_new,
                         Bx_new, By_new, Bz_new, Emfx, Emfy, Emfz,
                         Ni, NG, dt/dx),
           shared_mem=shared_bytes)

    # Swap
    mx, mx_new = mx_new, mx
    my, my_new = my_new, my
    mz, mz_new = mz_new, mz
    E_total, E_new = E_new, E_total
    Bx, Bx_new = Bx_new, Bx
    By, By_new = By_new, By
    Bz, Bz_new = Bz_new, Bz

    steps += 1
    if steps % print_interval == 0:
        vmax = float(cp.max(cp.sqrt((mx/rho)**2 + (my/rho)**2 + (mz/rho)**2)))
        print(f"Step {steps:4d} | Max|v| = {vmax:.4f}")

print("\n✅ We are here bro! Pure True CT Yee + JxB engine ready for galaxy rotation tests!")
