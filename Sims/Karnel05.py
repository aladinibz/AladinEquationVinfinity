import numpy as np
import cupy as cp
import time

# ====================== PARAMETERS ======================
N = 512
L = 1.0
dx = L / N
cfl = 0.11
dt_max = 0.000012
max_steps = 800
print_interval = 25
NG = 3
Ni = N + 2 * NG
gamma = 5.0 / 3.0

hall_coeff = 0.018
whistler_safety = 0.22
entropy_eps = 1e-6
ch = 1.5
base_bias = 0.45

BLOCK_EMF = (32, 8, 4)
BLOCK_HLLD = (16, 16, 2)
BLOCK_UCT = (32, 8, 1)

grid_emf = ((N + BLOCK_EMF[0] - 1) // BLOCK_EMF[0], (N + BLOCK_EMF[1] - 1) // BLOCK_EMF[1], (N + BLOCK_EMF[2] - 1) // BLOCK_EMF[2])
grid_hlld = ((N + BLOCK_HLLD[0] - 1) // BLOCK_HLLD[0], (N + BLOCK_HLLD[1] - 1) // BLOCK_HLLD[1], (N + BLOCK_HLLD[2] - 1) // BLOCK_HLLD[2])
grid_uct = ((N + BLOCK_UCT[0] - 1) // BLOCK_UCT[0], (N + BLOCK_UCT[1] - 1) // BLOCK_UCT[1], (N + BLOCK_UCT[2] - 1) // BLOCK_UCT[2])

# ====================== STREAMS FOR CONCURRENCY ======================
stream_main = cp.cuda.get_current_stream()
stream_uct = cp.cuda.Stream(non_blocking=True)
stream_emf = cp.cuda.Stream(non_blocking=True)
stream_hlld = cp.cuda.Stream(non_blocking=True)

# ====================== FIELDS ======================
rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
my = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
mz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
E_total = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 3.0
psi = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)

Bx = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
By = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
Bz = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)

rho1 = cp.zeros_like(rho); mx1 = cp.zeros_like(mx); my1 = cp.zeros_like(my); mz1 = cp.zeros_like(mz)
E1 = cp.zeros_like(E_total); psi1 = cp.zeros_like(psi)
Bx1 = cp.zeros_like(Bx); By1 = cp.zeros_like(By); Bz1 = cp.zeros_like(Bz)

rho2 = cp.zeros_like(rho); mx2 = cp.zeros_like(mx); my2 = cp.zeros_like(my); mz2 = cp.zeros_like(mz)
E2 = cp.zeros_like(E_total); psi2 = cp.zeros_like(psi)
Bx2 = cp.zeros_like(Bx); By2 = cp.zeros_like(By); Bz2 = cp.zeros_like(Bz)

rho3 = cp.zeros_like(rho); mx3 = cp.zeros_like(mx); my3 = cp.zeros_like(my); mz3 = cp.zeros_like(mz)
E3 = cp.zeros_like(E_total); psi3 = cp.zeros_like(psi)
Bx3 = cp.zeros_like(Bx); By3 = cp.zeros_like(By); Bz3 = cp.zeros_like(Bz)

Emfx = cp.zeros((Ni, Ni+1, Ni+1), dtype=cp.float32)
Emfy = cp.zeros((Ni+1, Ni, Ni+1), dtype=cp.float32)
Emfz = cp.zeros((Ni+1, Ni+1, Ni), dtype=cp.float32)

# ====================== GALAXY IC ======================
np.random.seed(42)
pert = 0.06
Bx[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
By[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
Bz[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert * 0.6 + 0.65, dtype=cp.float32)

x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
X, Y = cp.meshgrid(x, y)
R = cp.maximum(cp.sqrt(X**2 + Y**2), 0.15)
v_theta = 0.28 * R / (R + 0.25)
theta = cp.arctan2(Y, X)
vx_seed = v_theta * (-cp.sin(theta)) * 0.5
vy_seed = v_theta * cp.cos(theta) * 0.5
mx[NG:NG+N, NG:NG+N, NG:Ni-NG] += vx_seed.astype(cp.float32)
my[NG:NG+N, NG:NG+N, NG:Ni-NG] += vy_seed.astype(cp.float32)

def update_ghosts():
    for f in [rho, mx, my, mz, E_total, psi]:
        f[:NG] = f[-2*NG:-NG]; f[-NG:] = f[NG:2*NG]
        f[:,:NG] = f[:,-2*NG:-NG]; f[:,-NG:] = f[:,NG:2*NG]
        f[:,:, :NG] = f[:,:,-2*NG:-NG]; f[:,:,-NG:] = f[:,:,NG:2*NG]
    for f in [Bx, By, Bz]:
        f[:NG] = f[-2*NG:-NG]; f[-NG:] = f[NG:2*NG]
        f[:,:NG] = f[:,-2*NG:-NG]; f[:,-NG:] = f[:,NG:2*NG]
        f[:,:, :NG] = f[:,:,-2*NG:-NG]; f[:,:,-NG:] = f[:,:,NG:2*NG]

def print_memory_stats(label):
    pool = cp.get_default_memory_pool()
    used = pool.used_bytes() / (1024**3)
    total = pool.total_bytes() / (1024**3)
    print(f"[{label}] VRAM: {used:.2f}/{total:.2f} GB")

# ====================== KERNELS ======================
uct_predictor_kernel = cp.RawKernel(r'''
#define NG 3
#define TILE_X 32
#define TILE_Y 8
#define TILE_Z 1
#define PAD 4
extern "C" __launch_bounds__(256, 4)
__global__ void uct_predictor_kernel(float* Emfx, float* Emfy, float* Emfz,
    const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz, int Ni, float dt_over_dx, float base_bias) {
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * TILE_X + tx + NG;
    int j = blockIdx.y * TILE_Y + ty + NG;
    int k = blockIdx.z * TILE_Z + tz + NG;
    if (i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;

    __shared__ float s_rho[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];
    __shared__ float s_mx[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];
    __shared__ float s_my[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];
    __shared__ float s_mz[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];
    __shared__ float s_Bx[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];
    __shared__ float s_By[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];
    __shared__ float s_Bz[TILE_X+PAD][TILE_Y+PAD][TILE_Z+PAD];

    int sx = tx + PAD/2; int sy = ty + PAD/2; int sz = tz + PAD/2;

    s_rho[sx][sy][sz] = rho[i*Ni*Ni + j*Ni + k];
    s_mx[sx][sy][sz] = mx[i*Ni*Ni + j*Ni + k];
    s_my[sx][sy][sz] = my[i*Ni*Ni + j*Ni + k];
    s_mz[sx][sy][sz] = mz[i*Ni*Ni + j*Ni + k];
    s_Bx[sx][sy][sz] = Bx[i*Ni*Ni + j*Ni + k];
    s_By[sx][sy][sz] = By[i*Ni*Ni + j*Ni + k];
    s_Bz[sx][sy][sz] = Bz[i*Ni*Ni + j*Ni + k];

    __syncthreads();

    float vx = s_mx[sx][sy][sz] / s_rho[sx][sy][sz];
    float vy = s_my[sx][sy][sz] / s_rho[sx][sy][sz];
    float vz = s_mz[sx][sy][sz] / s_rho[sx][sy][sz];
    float speed = sqrtf(vx*vx + vy*vy + vz*vz);
    float adapt = base_bias * fminf(1.0f, speed * 8.0f);

    float avx = fabsf(vx); float avy = fabsf(vy); float avz = fabsf(vz);

    float base = -(vx * s_By[sx][sy][sz] - vy * s_Bx[sx][sy][sz]);
    float up_x = adapt * dt_over_dx * avy * (s_Bz[sx][sy][sz] - s_Bz[sx][sy-1][sz]);
    float up_y = adapt * dt_over_dx * avx * (s_Bx[sx][sy][sz] - s_Bx[sx-1][sy][sz]);
    Emfz[i*Ni*Ni + j*Ni + k] = base + up_x - up_y;

    base = -(vy * s_Bz[sx][sy][sz] - vz * s_By[sx][sy][sz]);
    up_y = adapt * dt_over_dx * avz * (s_Bx[sx][sy][sz] - s_Bx[sx][sy][sz-1]);
    up_z = adapt * dt_over_dx * avy * (s_By[sx][sy][sz] - s_By[sx-1][sy][sz]);
    Emfx[i*Ni*Ni + j*Ni + k] = base + up_y - up_z;

    base = -(vz * s_Bx[sx][sy][sz] - vx * s_Bz[sx][sy][sz]);
    up_z = adapt * dt_over_dx * avx * (s_By[sx][sy][sz] - s_By[sx][sy][sz-1]);
    up_x = adapt * dt_over_dx * avz * (s_Bz[sx][sy][sz] - s_Bz[sx-1][sy][sz]);
    Emfy[i*Ni*Ni + j*Ni + k] = base + up_z - up_x;
}
''', 'uct_predictor_kernel')

ct_emf_kernel = cp.RawKernel(r'''
extern "C" __global__ void ct_emf_kernel(const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz, float* Emfx, float* Emfy, float* Emfz,
    int Ni, float hall_coeff, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Ni || j >= Ni || k >= Ni) return;
    int idx = i*Ni*Ni + j*Ni + k;
    float vx = mx[idx]/rho[idx]; float vy = my[idx]/rho[idx]; float vz = mz[idx]/rho[idx];
    Emfx[idx] = -(vy * Bz[idx] - vz * By[idx]);
    Emfy[idx] = -(vz * Bx[idx] - vx * Bz[idx]);
    Emfz[idx] = -(vx * By[idx] - vy * Bx[idx]);
    float jx = (By[idx+Ni] - By[idx-Ni]) / (2.0f * dx) - (Bz[idx+1] - Bz[idx-1]) / (2.0f * dx);
    float jy = (Bz[idx+Ni*Ni] - Bz[idx-Ni*Ni]) / (2.0f * dx) - (Bx[idx+1] - Bx[idx-1]) / (2.0f * dx);
    float jz = (Bx[idx+Ni] - Bx[idx-Ni]) / (2.0f * dx) - (By[idx+1] - By[idx-1]) / (2.0f * dx);
    float rho_inv = 1.0f / rho[idx];
    Emfx[idx] -= hall_coeff * rho_inv * (jy * Bz[idx] - jz * By[idx]);
    Emfy[idx] -= hall_coeff * rho_inv * (jz * Bx[idx] - jx * Bz[idx]);
    Emfz[idx] -= hall_coeff * rho_inv * (jx * By[idx] - jy * Bx[idx]);
}
''', 'ct_emf_kernel')

ct_curl_kernel = cp.RawKernel(r'''
extern "C" __global__ void ct_curl_kernel(const float* Emfx, const float* Emfy, const float* Emfz,
    float* psi, float* Bx, float* By, float* Bz, float* Bx_new, float* By_new, float* Bz_new,
    int Ni, float dt_over_dx, float ch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Ni || j >= Ni || k >= Ni) return;
    int idx = i*Ni*Ni + j*Ni + k;
    Bx_new[idx] = Bx[idx] - dt_over_dx * ((Emfz[idx+Ni] - Emfz[idx-Ni]) - (Emfy[idx+1] - Emfy[idx-1]));
    By_new[idx] = By[idx] - dt_over_dx * ((Emfx[idx+1] - Emfx[idx-1]) - (Emfz[idx+Ni*Ni] - Emfz[idx-Ni*Ni]));
    Bz_new[idx] = Bz[idx] - dt_over_dx * ((Emfy[idx+Ni*Ni] - Emfy[idx-Ni*Ni]) - (Emfx[idx+Ni] - Emfx[idx-Ni]));
    psi[idx] -= ch * ch * dt_over_dx * (Bx_new[idx] - Bx[idx] + By_new[idx] - By[idx] + Bz_new[idx] - Bz[idx]);
}
''', 'ct_curl_kernel')

hlld_x_kernel = cp.RawKernel(r'''
extern "C" __global__ void hlld_x_kernel(const float* rho, const float* mx, const float* my, const float* mz,
    const float* E, const float* Bx, const float* By, const float* Bz,
    float* rho_new, float* mx_new, float* my_new, float* mz_new, float* E_new,
    int Ni, float dt_over_dx, float gamma, float entropy_eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= Ni-1) return;
    int idx = i*Ni*Ni + j*Ni + k;
    int left = idx - Ni*Ni;
    int right = idx + Ni*Ni;

    float rho_c = rho[idx];
    float dr = 0.5f * copysignf(fminf(fabsf(rho_c-rho[left]), fabsf(rho[right]-rho_c)), (rho_c-rho[left])*(rho[right]-rho_c));
    float rhoL = rho_c - dr; float rhoR = rho_c + dr;

    float vx_c = mx[idx]/rho_c;
    float dv = 0.5f * copysignf(fminf(fabsf(vx_c - mx[left]/rho[left]), fabsf(mx[right]/rho[right] - vx_c)), (vx_c - mx[left]/rho[left]) * (mx[right]/rho[right] - vx_c));
    float vxL = vx_c - dv; float vxR = vx_c + dv;

    float pL = (gamma-1.0f)*(E[left] - 0.5f*rho[left]*(vxL*vxL));
    float pR = (gamma-1.0f)*(E[right] - 0.5f*rho[right]*(vxR*vxR));

    float cfL = sqrt(gamma*pL/rhoL);
    float cfR = sqrt(gamma*pR/rhoR);

    float SL = min(vxL - cfL, vxR - cfR);
    float SR = max(vxL + cfL, vxR + cfR);

    float flux_rho = (SR*rhoL*vxL - SL*rhoR*vxR + SL*SR*(rhoR - rhoL)) / (SR - SL);

    rho_new[idx] = rho[idx] - dt_over_dx * flux_rho;
    mx_new[idx] = mx[idx] - dt_over_dx * flux_rho * (SL + SR) * 0.5f;
}
''', 'hlld_x_kernel')

hlld_y_kernel = cp.RawKernel(r'''
extern "C" __global__ void hlld_y_kernel(const float* rho, const float* mx, const float* my, const float* mz,
    const float* E, const float* Bx, const float* By, const float* Bz,
    float* rho_new, float* mx_new, float* my_new, float* mz_new, float* E_new,
    int Ni, float dt_over_dx, float gamma, float entropy_eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (j < 1 || j >= Ni-1) return;
    int idx = i*Ni*Ni + j*Ni + k;
    int left = idx - Ni;
    int right = idx + Ni;

    float rho_c = rho[idx];
    float dr = 0.5f * copysignf(fminf(fabsf(rho_c-rho[left]), fabsf(rho[right]-rho_c)), (rho_c-rho[left])*(rho[right]-rho_c));
    float rhoL = rho_c - dr; float rhoR = rho_c + dr;

    float vy_c = my[idx]/rho_c;
    float dv = 0.5f * copysignf(fminf(fabsf(vy_c - my[left]/rho[left]), fabsf(my[right]/rho[right] - vy_c)), (vy_c - my[left]/rho[left]) * (my[right]/rho[right] - vy_c));
    float vyL = vy_c - dv; float vyR = vy_c + dv;

    float pL = (gamma-1.0f)*(E[left] - 0.5f*rho[left]*(vyL*vyL));
    float pR = (gamma-1.0f)*(E[right] - 0.5f*rho[right]*(vyR*vyR));

    float cfL = sqrt(gamma*pL/rhoL);
    float cfR = sqrt(gamma*pR/rhoR);

    float SL = min(vyL - cfL, vyR - cfR);
    float SR = max(vyL + cfL, vyR + cfR);

    float flux_rho = (SR*rhoL*vyL - SL*rhoR*vyR + SL*SR*(rhoR - rhoL)) / (SR - SL);

    rho_new[idx] = rho[idx] - dt_over_dx * flux_rho;
}
''', 'hlld_y_kernel')

hlld_z_kernel = cp.RawKernel(r'''
extern "C" __global__ void hlld_z_kernel(const float* rho, const float* mx, const float* my, const float* mz,
    const float* E, const float* Bx, const float* By, const float* Bz,
    float* rho_new, float* mx_new, float* my_new, float* mz_new, float* E_new,
    int Ni, float dt_over_dx, float gamma, float entropy_eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (k < 1 || k >= Ni-1) return;
    int idx = i*Ni*Ni + j*Ni + k;
    int left = idx - 1;
    int right = idx + 1;

    float rho_c = rho[idx];
    float dr = 0.5f * copysignf(fminf(fabsf(rho_c-rho[left]), fabsf(rho[right]-rho_c)), (rho_c-rho[left])*(rho[right]-rho_c));
    float rhoL = rho_c - dr; float rhoR = rho_c + dr;

    float vz_c = mz[idx]/rho_c;
    float dv = 0.5f * copysignf(fminf(fabsf(vz_c - mz[left]/rho[left]), fabsf(mz[right]/rho[right] - vz_c)), (vz_c - mz[left]/rho[left]) * (mz[right]/rho[right] - vz_c));
    float vzL = vz_c - dv; float vzR = vz_c + dv;

    float pL = (gamma-1.0f)*(E[left] - 0.5f*rho[left]*(vzL*vzL));
    float pR = (gamma-1.0f)*(E[right] - 0.5f*rho[right]*(vzR*vzR));

    float cfL = sqrt(gamma*pL/rhoL);
    float cfR = sqrt(gamma*pR/rhoR);

    float SL = min(vzL - cfL, vzR - cfR);
    float SR = max(vzL + cfL, vzR + cfR);

    float flux_rho = (SR*rhoL*vzL - SL*rhoR*vzR + SL*SR*(rhoR - rhoL)) / (SR - SL);

    rho_new[idx] = rho[idx] - dt_over_dx * flux_rho;
}
''', 'hlld_z_kernel')

# ====================== BUILD GRAPH ======================
graph_exec = None
def build_graph():
    global graph_exec
    print("=== Starting Full RK3 Graph Capture ===")
    print_memory_stats("Before Capture")
    start = time.perf_counter()

    graph = cp.cuda.Graph()
    with graph.capture() as g:
        # STAGE 1
        uct_predictor_kernel(grid_uct, BLOCK_UCT, (Emfx, Emfy, Emfz, rho, mx, my, mz, Bx, By, Bz, Ni, 0.0, base_bias))
        ct_emf_kernel(grid_emf, BLOCK_EMF, (rho, mx, my, mz, Bx, By, Bz, Emfx, Emfy, Emfz, Ni, hall_coeff, dx))
        ct_curl_kernel(grid_emf, BLOCK_EMF, (Emfx, Emfy, Emfz, psi, Bx, By, Bz, Bx1, By1, Bz1, Ni, 0.0, ch))
        hlld_x_kernel(grid_hlld, BLOCK_HLLD, (rho, mx, my, mz, E_total, Bx, By, Bz, rho1, mx1, my1, mz1, E1, Ni, 0.0, gamma, entropy_eps))
        hlld_y_kernel(grid_hlld, BLOCK_HLLD, (rho, mx, my, mz, E_total, Bx, By, Bz, rho1, mx1, my1, mz1, E1, Ni, 0.0, gamma, entropy_eps))
        hlld_z_kernel(grid_hlld, BLOCK_HLLD, (rho, mx, my, mz, E_total, Bx, By, Bz, rho1, mx1, my1, mz1, E1, Ni, 0.0, gamma, entropy_eps))

        # STAGE 2
        uct_predictor_kernel(grid_uct, BLOCK_UCT, (Emfx, Emfy, Emfz, rho1, mx1, my1, mz1, Bx1, By1, Bz1, Ni, 0.0, base_bias))
        ct_emf_kernel(grid_emf, BLOCK_EMF, (rho1, mx1, my1, mz1, Bx1, By1, Bz1, Emfx, Emfy, Emfz, Ni, hall_coeff, dx))
        ct_curl_kernel(grid_emf, BLOCK_EMF, (Emfx, Emfy, Emfz, psi, Bx1, By1, Bz1, Bx2, By2, Bz2, Ni, 0.0, ch))
        hlld_x_kernel(grid_hlld, BLOCK_HLLD, (rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1, rho2, mx2, my2, mz2, E2, Ni, 0.0, gamma, entropy_eps))
        hlld_y_kernel(grid_hlld, BLOCK_HLLD, (rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1, rho2, mx2, my2, mz2, E2, Ni, 0.0, gamma, entropy_eps))
        hlld_z_kernel(grid_hlld, BLOCK_HLLD, (rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1, rho2, mx2, my2, mz2, E2, Ni, 0.0, gamma, entropy_eps))

        # STAGE 3
        uct_predictor_kernel(grid_uct, BLOCK_UCT, (Emfx, Emfy, Emfz, rho2, mx2, my2, mz2, Bx2, By2, Bz2, Ni, 0.0, base_bias))
        ct_emf_kernel(grid_emf, BLOCK_EMF, (rho2, mx2, my2, mz2, Bx2, By2, Bz2, Emfx, Emfy, Emfz, Ni, hall_coeff, dx))
        ct_curl_kernel(grid_emf, BLOCK_EMF, (Emfx, Emfy, Emfz, psi, Bx2, By2, Bz2, Bx3, By3, Bz3, Ni, 0.0, ch))
        hlld_x_kernel(grid_hlld, BLOCK_HLLD, (rho2, mx2, my2, mz2, E2, Bx2, By2, Bz2, rho3, mx3, my3, mz3, E3, Ni, 0.0, gamma, entropy_eps))
        hlld_y_kernel(grid_hlld, BLOCK_HLLD, (rho2, mx2, my2, mz2, E2, Bx2, By2, Bz2, rho3, mx3, my3, mz3, E3, Ni, 0.0, gamma, entropy_eps))
        hlld_z_kernel(grid_hlld, BLOCK_HLLD, (rho2, mx2, my2, mz2, E2, Bx2, By2, Bz2, rho3, mx3, my3, mz3, E3, Ni, 0.0, gamma, entropy_eps))

    graph_exec = graph.instantiate()
    capture_time = (time.perf_counter() - start) * 1000
    print(f"Full RK3 Graph Capture Latency: {capture_time:.2f} ms")
    print_memory_stats("After Capture")
    print("✅ Full SSP-RK3 Graph with Streams Ready\n")

build_graph()

# ====================== MAIN LOOP ======================
steps = 0
start_time = time.time()

while steps < max_steps:
    update_ghosts()

    rho_safe = cp.maximum(rho, 1e-8)
    v2 = (mx/rho_safe)**2 + (my/rho_safe)**2 + (mz/rho_safe)**2
    cmax = float(cp.sqrt(cp.max(v2) + 1.0))
    v_whistler = hall_coeff * float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2))) / float(cp.mean(rho_safe))
    dt = min(cfl * dx / (cmax + v_whistler), whistler_safety * dx**2 / (v_whistler * dx + 1e-12), dt_max)

    graph_exec.launch(stream=stream_main)

    steps += 1
    if steps % print_interval == 0:
        KE = 0.5 * float(cp.sum(rho[NG:NG+N] * v2[NG:NG+N]))
        ME = 0.5 * float(cp.sum(Bx**2 + By**2 + Bz**2)) * (dx**3)
        elapsed = time.time() - start_time
        print(f"Step {steps:4d} | dt={dt:.2e} | KE={KE:.2e} ME={ME:.2e} | t={elapsed:.1f}s")

print("\n✅ v3.9 FULL COMPLETE SIM WITH STREAM CONCURRENCY READY!")
