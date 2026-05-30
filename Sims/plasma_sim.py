import numpy as np
import cupy as cp
import time
import sys

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
base_bias = 0.45

TILE_X, TILE_Y, TILE_Z = 32, 8, 4
PAD = 2

grid_emf = (
    (N + TILE_X - 1) // TILE_X,
    (N + TILE_Y - 1) // TILE_Y,
    (N + TILE_Z - 1) // TILE_Z
)

# ====================== FIELDS ======================
rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
my = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
mz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
E_total = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 3.0

Bx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
By = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
Bz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
psi = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)

pe_xx = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 1.0
pe_yy = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 1.0
pe_zz = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 1.0
pe_xy = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
pe_xz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
pe_yz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)

Emfx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
Emfy = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
Emfz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)

# RK3 temporaries
rho3 = rho.copy()
mx3 = mx.copy()
my3 = my.copy()
mz3 = mz.copy()
E3 = E_total.copy()
Bx3 = Bx.copy()
By3 = By.copy()
Bz3 = Bz.copy()
psi3 = psi.copy()

# ====================== UTC_EMF_KERNEL (Fixed) ======================
utc_emf_kernel = cp.RawKernel(r'''
#define TILE_X 32
#define TILE_Y 8
#define TILE_Z 4
#define PAD 2

__device__ float mc_limiter(float a, float b) {
    if (a * b <= 0.0f) return 0.0f;
    float min1 = 2.0f * fminf(fabsf(a), fabsf(b));
    float min2 = 0.5f * fabsf(a + b);
    return copysignf(fminf(min1, min2), a);
}

__device__ float hodge_face_to_edge(float f1, float f2) {
    return 0.5f * (f1 + f2);
}

extern "C" __launch_bounds__(256, 4)
__global__ void utc_emf_kernel(const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz,
    const float* pe_xx, const float* pe_yy, const float* pe_zz,
    const float* pe_xy, const float* pe_xz, const float* pe_yz,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, float hall_coeff, float dx, float dt_over_dx, float base_bias) {

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * TILE_X + tx;
    int j = blockIdx.y * TILE_Y + ty;
    int k = blockIdx.z * TILE_Z + tz;

    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;

    int sx = tx + PAD;
    int sy = ty + PAD;
    int sz = tz + PAD;

    __shared__ float s_rho[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_mx [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_my [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_mz [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Bx [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_By [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Bz [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    __shared__ float s_pexy[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_pexz[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_peyz[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    int tid = tx + ty * TILE_X + tz * TILE_X * TILE_Y;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    for (int offset = tid * 4; offset < (TILE_X + 2*PAD)*(TILE_Y + 2*PAD)*(TILE_Z + 2*PAD); offset += total_threads * 4) {
        int lx = offset % (TILE_X + 2*PAD);
        int ly = (offset / (TILE_X + 2*PAD)) % (TILE_Y + 2*PAD);
        int lz = offset / ((TILE_X + 2*PAD) * (TILE_Y + 2*PAD));

        int li = i + (lx - PAD);
        int lj = j + (ly - PAD);
        int lk = k + (lz - PAD);

        if (li >= 0 && li < Ni && lj >= 0 && lj < Ni && lk >= 0 && lk < Ni) {
            int gidx = li * Ni * Ni + lj * Ni + lk;

            if (lx + 3 < TILE_X + 2*PAD) {
                float4 v4 = ((const float4*)(rho + gidx))[0]; ((float4*)(&s_rho[lx][ly][lz]))[0] = v4;
                v4 = ((const float4*)(mx + gidx))[0];   ((float4*)(&s_mx[lx][ly][lz]))[0] = v4;
                v4 = ((const float4*)(my + gidx))[0];   ((float4*)(&s_my[lx][ly][lz]))[0] = v4;
                v4 = ((const float4*)(mz + gidx))[0];   ((float4*)(&s_mz[lx][ly][lz]))[0] = v4;
                v4 = ((const float4*)(Bx + gidx))[0];   ((float4*)(&s_Bx[lx][ly][lz]))[0] = v4;
                v4 = ((const float4*)(By + gidx))[0];   ((float4*)(&s_By[lx][ly][lz]))[0] = v4;
                v4 = ((const float4*)(Bz + gidx))[0];   ((float4*)(&s_Bz[lx][ly][lz]))[0] = v4;
            } else {
                s_rho[lx][ly][lz] = rho[gidx];
                s_mx[lx][ly][lz]  = mx[gidx];
                s_my[lx][ly][lz]  = my[gidx];
                s_mz[lx][ly][lz]  = mz[gidx];
                s_Bx[lx][ly][lz]  = Bx[gidx];
                s_By[lx][ly][lz]  = By[gidx];
                s_Bz[lx][ly][lz]  = Bz[gidx];
            }

            s_pexy[lx][ly][lz] = pe_xy[gidx];
            s_pexz[lx][ly][lz] = pe_xz[gidx];
            s_peyz[lx][ly][lz] = pe_yz[gidx];
        }
    }
    __syncthreads();

    float rho0 = fmaxf(s_rho[sx][sy][sz], 1e-8f);
    float vx0 = s_mx[sx][sy][sz] / rho0;
    float vy0 = s_my[sx][sy][sz] / rho0;
    float vz0 = s_mz[sx][sy][sz] / rho0;

    float adapt = base_bias * (1.0f + 6.0f * (fabsf(vx0) + fabsf(vy0) + fabsf(vz0)) * dt_over_dx);

    float jx = (s_By[sx][sy][sz+1] - s_By[sx][sy][sz-1] - (s_Bz[sx][sy+1][sz] - s_Bz[sx][sy-1][sz])) * 0.5f / dx;
    float jy = (s_Bz[sx][sy][sz+1] - s_Bz[sx][sy][sz-1] - (s_Bx[sx][sy+1][sz] - s_Bx[sx][sy-1][sz])) * 0.5f / dx;
    float jz = (s_Bx[sx][sy+1][sz] - s_Bx[sx][sy-1][sz] - (s_By[sx][sy][sz+1] - s_By[sx][sy][sz-1])) * 0.5f / dx;

    // Emfx (y-z edge)
    {
        float rho_E = 0.25f * (s_rho[sx][sy][sz] + s_rho[sx][sy+1][sz] + s_rho[sx][sy][sz+1] + s_rho[sx][sy+1][sz+1]);
        float vy_E = vy0 + mc_limiter(vy0 - s_my[sx][sy-1][sz]/fmaxf(s_rho[sx][sy-1][sz],1e-8f), s_my[sx][sy+1][sz]/fmaxf(s_rho[sx][sy+1][sz],1e-8f) - vy0);
        float vz_E = vz0 + mc_limiter(vz0 - s_mz[sx][sy-1][sz]/fmaxf(s_rho[sx][sy-1][sz],1e-8f), s_mz[sx][sy+1][sz]/fmaxf(s_rho[sx][sy+1][sz],1e-8f) - vz0);

        float By_E = hodge_face_to_edge(s_By[sx][sy][sz], s_By[sx][sy+1][sz]);
        float Bz_E = hodge_face_to_edge(s_Bz[sx][sy][sz], s_Bz[sx][sy][sz+1]);

        float inv_ne = 1.0f / rho_E;
        float vey_E = vy_E - jy * hall_coeff * inv_ne;
        float vez_E = vz_E - jz * hall_coeff * inv_ne;

        float E_ve = -(vey_E * Bz_E - vez_E * By_E);

        float corner = adapt * dt_over_dx * 0.25f * (
            (s_my[sx][sy+1][sz] * s_Bz[sx][sy+1][sz] - s_my[sx][sy-1][sz] * s_Bz[sx][sy-1][sz]) -
            (s_mz[sx][sy][sz+1] * s_By[sx][sy][sz+1] - s_mz[sx][sy][sz-1] * s_By[sx][sy][sz-1])
        );

        float dPxy_dy = (s_pexy[sx][sy+1][sz] - s_pexy[sx][sy-1][sz]) * 0.5f / dx;
        float dPxz_dz = (s_pexz[sx][sy][sz+1] - s_pexz[sx][sy][sz-1]) * 0.5f / dx;
        float Pe_term = - (dPxy_dy + dPxz_dz) * inv_ne;

        Emfx[i*Ni*Ni + j*Ni + k] = E_ve - corner + Pe_term;
    }

    // Emfy (x-z edge)
    {
        float rho_E = 0.25f * (s_rho[sx][sy][sz] + s_rho[sx+1][sy][sz] + s_rho[sx][sy][sz+1] + s_rho[sx+1][sy][sz+1]);
        float vz_E = vz0 + mc_limiter(vz0 - s_mz[sx-1][sy][sz]/fmaxf(s_rho[sx-1][sy][sz],1e-8f), s_mz[sx+1][sy][sz]/fmaxf(s_rho[sx+1][sy][sz],1e-8f) - vz0);
        float vx_E = vx0 + mc_limiter(vx0 - s_mx[sx-1][sy][sz]/fmaxf(s_rho[sx-1][sy][sz],1e-8f), s_mx[sx+1][sy][sz]/fmaxf(s_rho[sx+1][sy][sz],1e-8f) - vx0);

        float Bx_E = hodge_face_to_edge(s_Bx[sx][sy][sz], s_Bx[sx+1][sy][sz]);
        float Bz_E = hodge_face_to_edge(s_Bz[sx][sy][sz], s_Bz[sx][sy][sz+1]);

        float inv_ne = 1.0f / rho_E;
        float vez_E = vz_E - jz * hall_coeff * inv_ne;
        float vex_E = vx_E - jx * hall_coeff * inv_ne;

        float E_ve = -(vez_E * Bx_E - vex_E * Bz_E);

        float corner = adapt * dt_over_dx * 0.25f * (
            (s_mz[sx+1][sy][sz] * s_Bx[sx+1][sy][sz] - s_mz[sx-1][sy][sz] * s_Bx[sx-1][sy][sz]) -
            (s_mx[sx][sy][sz+1] * s_Bz[sx][sy][sz+1] - s_mx[sx][sy][sz-1] * s_Bz[sx][sy][sz-1])
        );

        float dPxy_dx = (s_pexy[sx+1][sy][sz] - s_pexy[sx-1][sy][sz]) * 0.5f / dx;
        float dPyz_dz = (s_peyz[sx][sy][sz+1] - s_peyz[sx][sy][sz-1]) * 0.5f / dx;
        float Pe_term = - (dPxy_dx + dPyz_dz) * inv_ne;

        Emfy[i*Ni*Ni + j*Ni + k] = E_ve - corner + Pe_term;
    }

    // Emfz (x-y edge)
    {
        float rho_E = 0.25f * (s_rho[sx][sy][sz] + s_rho[sx+1][sy][sz] + s_rho[sx][sy+1][sz] + s_rho[sx+1][sy+1][sz]);
        float vx_E = vx0 + mc_limiter(vx0 - s_mx[sx-1][sy][sz]/fmaxf(s_rho[sx-1][sy][sz],1e-8f), s_mx[sx+1][sy][sz]/fmaxf(s_rho[sx+1][sy][sz],1e-8f) - vx0);
        float vy_E = vy0 + mc_limiter(vy0 - s_my[sx][sy-1][sz]/fmaxf(s_rho[sx][sy-1][sz],1e-8f), s_my[sx][sy+1][sz]/fmaxf(s_rho[sx][sy+1][sz],1e-8f) - vy0);

        float Bx_E = hodge_face_to_edge(s_Bx[sx][sy][sz], s_Bx[sx][sy+1][sz]);
        float By_E = hodge_face_to_edge(s_By[sx][sy][sz], s_By[sx][sy][sz+1]);

        float inv_ne = 1.0f / rho_E;
        float vex_E = vx_E - jx * hall_coeff * inv_ne;
        float vey_E = vy_E - jy * hall_coeff * inv_ne;

        float E_ve = -(vex_E * By_E - vey_E * Bx_E);

        float corner = adapt * dt_over_dx * 0.25f * (
            (s_mx[sx+1][sy][sz] * s_By[sx+1][sy][sz] - s_mx[sx-1][sy][sz] * s_By[sx-1][sy][sz]) -
            (s_my[sx][sy+1][sz] * s_Bx[sx][sy+1][sz] - s_my[sx][sy-1][sz] * s_Bx[sx][sy-1][sz])
        );

        float dPxz_dx = (s_pexz[sx+1][sy][sz] - s_pexz[sx-1][sy][sz]) * 0.5f / dx;
        float dPyz_dy = (s_peyz[sx][sy+1][sz] - s_peyz[sx][sy-1][sz]) * 0.5f / dx;
        float Pe_term = - (dPxz_dx + dPyz_dy) * inv_ne;

        Emfz[i*Ni*Ni + j*Ni + k] = E_ve - corner + Pe_term;
    }
}
''', 'utc_emf_kernel')

# ====================== GHOST UPDATES ======================
def update_ghosts():
    for field in (rho, mx, my, mz, E_total, psi):
        field[:, :, :NG] = field[:, :, -NG:]
        field[:, :, -NG:] = field[:, :, NG:2*NG]
        field[:, :NG, :] = field[:, -NG:, :]
        field[:, -NG:, :] = field[:, NG:2*NG, :]
        field[:NG, :, :] = field[-NG:, :, :]
        field[-NG:, :, :] = field[NG:2*NG, :, :]

    for field in (Bx, By, Bz):
        field[:, :, :NG] = field[:, :, -NG:]
        field[:, :, -NG:] = field[:, :, NG:2*NG]
        field[:, :NG, :] = field[:, -NG:, :]
        field[:, -NG:, :] = field[:, NG:2*NG, :]
        field[:NG, :, :] = field[-NG:, :, :]
        field[-NG:, :, :] = field[NG:2*NG, :, :]

    for p in (pe_xx, pe_yy, pe_zz, pe_xy, pe_xz, pe_yz):
        p[:, :, :NG] = p[:, :, -NG:]
        p[:, :, -NG:] = p[:, :, NG:2*NG]
        p[:, :NG, :] = p[:, -NG:, :]
        p[:, -NG:, :] = p[:, NG:2*NG, :]
        p[:NG, :, :] = p[-NG:, :, :]
        p[-NG:, :, :] = p[NG:2*NG, :, :]

# ====================== MAIN LOOP ======================
steps = 0
start_time = time.time()

print("🚀 Starting Plasma Universe Simulation...\n")

while steps < max_steps:
    update_ghosts()

    rho_safe = cp.maximum(rho, 1e-8)
    v2 = (mx/rho_safe)**2 + (my/rho_safe)**2 + (mz/rho_safe)**2
    cmax = float(cp.sqrt(cp.max(v2) + 1.0))

    Bmag = cp.sqrt(Bx**2 + By**2 + Bz**2)
    B_max = float(cp.max(Bmag))
    rho_min = float(cp.min(rho_safe))

    dt_mhd = cfl * dx / (cmax + 1e-8)
    dt_hall = 0.20 * (dx**2 * rho_min) / (hall_coeff * (B_max + 1e-8))
    dt = min(dt_mhd, dt_hall, dt_max)
    dt_over_dx = dt / dx

    utc_emf_kernel(
        grid_emf, (TILE_X, TILE_Y, TILE_Z), stream=cp.cuda.get_current_stream(),
        args=(rho, mx, my, mz, Bx, By, Bz,
              pe_xx, pe_yy, pe_zz, pe_xy, pe_xz, pe_yz,
              Emfx, Emfy, Emfz,
              Ni, hall_coeff, dx, dt_over_dx, base_bias)
    )

    steps += 1
    if steps % print_interval == 0:
        KE = 0.5 * float(cp.sum(rho[NG:NG+N] * v2[NG:NG+N]))
        ME = 0.5 * float(cp.sum(Bx**2 + By**2 + Bz**2)) * (dx**3)
        TE = float(cp.sum(E_total[NG:NG+N])) * (dx**3)
        print(f"Step {steps:4d} | dt={dt:.2e} | KE={KE:.2e} ME={ME:.2e} TE={TE:.2e}")

print("\n✅ Simulation finished!")
