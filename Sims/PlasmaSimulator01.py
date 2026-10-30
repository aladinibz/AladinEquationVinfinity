import cupy as cp
import numpy as np

# ============================================================
# Plasma Simulator — Galactic Disk
# Complete driver with all four kernels
# ============================================================

# ---------- ct_emf_kernel ----------
ct_emf_kernel = cp.RawKernel(r'''
#define TILE_X 16
#define TILE_Y 4
#define TILE_Z 4
#define PAD 1
__device__ float hodge_face_to_edge(float f1, float f2) {
    return 0.5f * (f1 + f2);
}
extern "C" __launch_bounds__(256, 2)
__global__ void utc_emf_kernel(
    const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz,
    const float* pe_xx, const float* pe_yy, const float* pe_zz,
    const float* pe_xy, const float* pe_xz, const float* pe_yz,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, float hall_coeff, float dx, float dt_over_dx, float base_bias)
{
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * TILE_X + tx;
    int j = blockIdx.y * TILE_Y + ty;
    int k = blockIdx.z * TILE_Z + tz;
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;

    int sx = tx + PAD;
    int sy = ty + PAD;
    int sz = tz + PAD;

    // Only B needs the halo stencil
    __shared__ float s_Bx[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_By[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Bz[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    int tid = tx + ty * TILE_X + tz * TILE_X * TILE_Y;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int offset = tid; offset < (TILE_X + 2*PAD)*(TILE_Y + 2*PAD)*(TILE_Z + 2*PAD); offset += total_threads) {
        int lx = offset % (TILE_X + 2*PAD);
        int ly = (offset / (TILE_X + 2*PAD)) % (TILE_Y + 2*PAD);
        int lz = offset / ((TILE_X + 2*PAD) * (TILE_Y + 2*PAD));
        int li = blockIdx.x * TILE_X + (lx - PAD);
        int lj = blockIdx.y * TILE_Y + (ly - PAD);
        int lk = blockIdx.z * TILE_Z + (lz - PAD);
        if (li >= 0 && li < Ni && lj >= 0 && lj < Ni && lk >= 0 && lk < Ni) {
            int gidx = li * Ni * Ni + lj * Ni + lk;
            s_Bx[lx][ly][lz] = Bx[gidx];
            s_By[lx][ly][lz] = By[gidx];
            s_Bz[lx][ly][lz] = Bz[gidx];
        }
    }
    __syncthreads();

    int gidx = i * Ni * Ni + j * Ni + k;
    float rho0 = fmaxf(rho[gidx], 1e-8f);
    float vx0 = mx[gidx] / rho0;
    float vy0 = my[gidx] / rho0;
    float vz0 = mz[gidx] / rho0;

    int out = gidx;

    // Emfx
    {
        float By_E = hodge_face_to_edge(s_By[sx][sy][sz], s_By[sx][sy+1][sz]);
        float Bz_E = hodge_face_to_edge(s_Bz[sx][sy][sz], s_Bz[sx][sy][sz+1]);
        Emfx[out] = -(vy0 * Bz_E - vz0 * By_E);
    }
    // Emfy
    {
        float Bx_E = hodge_face_to_edge(s_Bx[sx][sy][sz], s_Bx[sx+1][sy][sz]);
        float Bz_E = hodge_face_to_edge(s_Bz[sx][sy][sz], s_Bz[sx][sy][sz+1]);
        Emfy[out] = -(vz0 * Bx_E - vx0 * Bz_E);
    }
    // Emfz
    {
        float Bx_E = hodge_face_to_edge(s_Bx[sx][sy][sz], s_Bx[sx][sy+1][sz]);
        float By_E = hodge_face_to_edge(s_By[sx][sy][sz], s_By[sx][sy][sz+1]);
        Emfz[out] = -(vx0 * By_E - vy0 * Bx_E);
    }
}
''', 'utc_emf_kernel')

# ---------- glm_kernel ----------
glm_kernel = cp.RawKernel(r'''
#define TILE_X 32
#define TILE_Y 4
#define TILE_Z 4
#define PAD 1
extern "C" __launch_bounds__(512, 4)
__global__ void glm_kernel(
    float* Bx, float* By, float* Bz, float* psi,
    int Ni, float dt, float dx, float ch, float damping) {
    int i = blockIdx.x * TILE_X + threadIdx.x;
    int j = blockIdx.y * TILE_Y + threadIdx.y;
    int k = blockIdx.z * TILE_Z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
    int idx = i * Ni * Ni + j * Ni + k;
    float inv2dx = 0.5f / dx;
    float divB = ((Bx[idx+Ni*Ni]-Bx[idx-Ni*Ni]) + (By[idx+Ni]-By[idx-Ni]) + (Bz[idx+1]-Bz[idx-1])) * inv2dx;
    psi[idx] -= dt * (ch*ch * divB + damping * psi[idx]);
    Bx[idx] -= dt * (psi[idx+Ni*Ni] - psi[idx-Ni*Ni]) * inv2dx;
    By[idx] -= dt * (psi[idx+Ni] - psi[idx-Ni]) * inv2dx;
    Bz[idx] -= dt * (psi[idx+1] - psi[idx-1]) * inv2dx;
}
''', 'glm_kernel')

# ---------- gyrotropic_12_kernel (minimal stub) ----------
gyrotropic_12_kernel = cp.RawKernel(r'''
extern "C" __global__
void gyrotropic_12_kernel(
    const float* M_target, const float* n_in, float* lambda_io,
    float* n_out, float* upar_out, float* Ppar_out, float* Pperp_out,
    float* qpar_out, float* qperp_out, int* status,
    int Ncells, float mass, float thermal_scale, float alpha_par, float alpha_perp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncells) return;
    float n = n_in[idx];
    Ppar_out[idx]  = mass * n * 0.1f;
    Pperp_out[idx] = mass * n * 0.1f;
    qpar_out[idx]  = 0.f;
    qperp_out[idx] = 0.f;
    status[idx]    = 0;
}
''', 'gyrotropic_12_kernel')

# ============================================================
# Driver
# ============================================================
Ni = 64
Ncells = Ni * Ni * Ni
dx = 1.0
dt = 0.005
inv2dx = 0.5 / dx
hall_coeff = 0.0
base_bias = 0.5
ch = 1.0
damping = 0.5
mass = 1.0
thermal_scale = 1.0
alpha_par = 0.5
alpha_perp = 0.5
n_steps = 10

rho  = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx   = cp.zeros_like(rho)
my   = cp.zeros_like(rho)
mz   = cp.zeros_like(rho)
Bx   = cp.zeros_like(rho)
By   = cp.zeros_like(rho)
Bz   = cp.ones_like(rho) * 0.05
psi  = cp.zeros_like(rho)
Emfx = cp.zeros_like(rho)
Emfy = cp.zeros_like(rho)
Emfz = cp.zeros_like(rho)
pe_xx = cp.ones_like(rho) * 0.1
pe_yy = cp.ones_like(rho) * 0.1
pe_zz = cp.ones_like(rho) * 0.1
pe_xy = cp.zeros_like(rho)
pe_xz = cp.zeros_like(rho)
pe_yz = cp.zeros_like(rho)

M_target  = cp.zeros((Ncells, 12), dtype=cp.float32)
lambda_io = cp.zeros((Ncells, 12), dtype=cp.float32)
n_in      = cp.ones(Ncells, dtype=cp.float32)
n_out     = cp.empty(Ncells, dtype=cp.float32)
upar      = cp.empty(Ncells, dtype=cp.float32)
Ppar      = cp.empty(Ncells, dtype=cp.float32)
Pperp     = cp.empty(Ncells, dtype=cp.float32)
qpar      = cp.empty(Ncells, dtype=cp.float32)
qperp     = cp.empty(Ncells, dtype=cp.float32)
status    = cp.zeros(Ncells, dtype=cp.int32)

def init_disk():
    i = cp.arange(Ni, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(i, i, i, indexing='ij')
    R = cp.sqrt((X - Ni/2)**2 + (Y - Ni/2)**2) + 1e-3
    rho[:] = cp.exp(-R / 12.0) * cp.exp(-cp.abs(Z - Ni/2) / 4.0)
    rho[:] = cp.clip(rho, 0.05, None)
    vphi = cp.sqrt(1.0 / (R + 2.0))
    mx[:] = -vphi * (Y - Ni/2) / R * rho
    my[:] =  vphi * (X - Ni/2) / R * rho

init_disk()

threads   = (32, 4, 4)
blocks    = ((Ni + 31)//32, (Ni + 3)//4, (Ni + 3)//4)
threads_g = 128
blocks_g  = (Ncells + 127) // 128

def build_M_target():
    M_target[:, 0] = 1.0
    M_target[:, 2] = cp.clip(Pperp / (rho.ravel() + 1e-8), 1e-6, 5.0)
    M_target[:, 3] = cp.clip(Ppar  / (rho.ravel() + 1e-8), 1e-6, 5.0)

def write_pe():
    pe_xx[:] = Pperp.reshape(Ni, Ni, Ni)
    pe_yy[:] = Pperp.reshape(Ni, Ni, Ni)
    pe_zz[:] = Ppar.reshape(Ni, Ni, Ni)

print("Plasma simulator starting ...")
for step in range(n_steps):
    build_M_target()
    gyrotropic_12_kernel(
        (blocks_g,), (threads_g,),
        (M_target, n_in, lambda_io,
         n_out, upar, Ppar, Pperp, qpar, qperp,
         status, Ncells, mass, thermal_scale, alpha_par, alpha_perp)
    )
    write_pe()
    ct_emf_kernel(
        blocks, threads,
        (rho, mx, my, mz, Bx, By, Bz,
         pe_xx, pe_yy, pe_zz, pe_xy, pe_xz, pe_yz,
         Emfx, Emfy, Emfz,
         Ni, hall_coeff, dx, dt/dx, base_bias)
    )
    ct_curl_kernel(
        blocks, threads,
        (Emfx, Emfy, Emfz, Bx, By, Bz, Ni, dt, inv2dx)
    )
    glm_kernel(
        blocks, threads,
        (Bx, By, Bz, psi, Ni, dt, dx, ch, damping)
    )
    if step % 5 == 0:
        print(f"step {step:3d}  status unique={cp.unique(status)}  "
              f"mean |B|={float(cp.mean(cp.sqrt(Bx**2+By**2+Bz**2))):.4f}")

print("Done.")
print("Final mean Ppar / Pperp =", float(cp.mean(Ppar)), float(cp.mean(Pperp)))
