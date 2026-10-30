import cupy as cp
import numpy as np

# ============================================================
# Plasma Simulator — Galactic Disk
# Complete driver (RK3 + CUDA Graph)
# ============================================================

# ---------- Paste your four RawKernels here ----------
# ct_emf_kernel = cp.RawKernel(r'''
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

    // Vectorized cooperative halo loading
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

    // Canonical J (computed once)
    float jx = (s_By[sx][sy][sz+1] - s_By[sx][sy][sz-1] - (s_Bz[sx][sy+1][sz] - s_Bz[sx][sy-1][sz])) * 0.5f / dx;
    float jy = (s_Bz[sx][sy][sz+1] - s_Bz[sx][sy][sz-1] - (s_Bx[sx][sy+1][sz] - s_Bx[sx][sy-1][sz])) * 0.5f / dx;
    float jz = (s_Bx[sx][sy+1][sz] - s_Bx[sx][sy-1][sz] - (s_By[sx][sy][sz+1] - s_By[sx][sy][sz-1])) * 0.5f / dx;

    // ====================== Emfx (y-z edge) ======================
    {
        float rho_E = 0.25f * (s_rho[sx][sy][sz] + s_rho[sx][sy+1][sz] + 
                               s_rho[sx][sy][sz+1] + s_rho[sx][sy+1][sz+1]);

        float vy_E = vy0 + mc_limiter(vy0 - s_my[sx][sy-1][sz]/fmaxf(s_rho[sx][sy-1][sz],1e-8f),
                                      s_my[sx][sy+1][sz]/fmaxf(s_rho[sx][sy+1][sz],1e-8f) - vy0);
        float vz_E = vz0 + mc_limiter(vz0 - s_mz[sx][sy-1][sz]/fmaxf(s_rho[sx][sy-1][sz],1e-8f),
                                      s_mz[sx][sy+1][sz]/fmaxf(s_rho[sx][sy+1][sz],1e-8f) - vz0);

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

        Emfx[i][j][k] = E_ve - corner + Pe_term;
    }

    // ====================== Emfy (x-z edge) ======================
    {
        float rho_E = 0.25f * (s_rho[sx][sy][sz] + s_rho[sx+1][sy][sz] + 
                               s_rho[sx][sy][sz+1] + s_rho[sx+1][sy][sz+1]);

        float vz_E = vz0 + mc_limiter(vz0 - s_mz[sx-1][sy][sz]/fmaxf(s_rho[sx-1][sy][sz],1e-8f),
                                      s_mz[sx+1][sy][sz]/fmaxf(s_rho[sx+1][sy][sz],1e-8f) - vz0);
        float vx_E = vx0 + mc_limiter(vx0 - s_mx[sx-1][sy][sz]/fmaxf(s_rho[sx-1][sy][sz],1e-8f),
                                      s_mx[sx+1][sy][sz]/fmaxf(s_rho[sx+1][sy][sz],1e-8f) - vx0);

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

        Emfy[i][j][k] = E_ve - corner + Pe_term;
    }

    // ====================== Emfz (x-y edge) ======================
    {
        float rho_E = 0.25f * (s_rho[sx][sy][sz] + s_rho[sx+1][sy][sz] + 
                               s_rho[sx][sy+1][sz] + s_rho[sx+1][sy+1][sz]);

        float vx_E = vx0 + mc_limiter(vx0 - s_mx[sx-1][sy][sz]/fmaxf(s_rho[sx-1][sy][sz],1e-8f),
                                      s_mx[sx+1][sy][sz]/fmaxf(s_rho[sx+1][sy][sz],1e-8f) - vx0);
        float vy_E = vy0 + mc_limiter(vy0 - s_my[sx][sy-1][sz]/fmaxf(s_rho[sx][sy-1][sz],1e-8f),
                                      s_my[sx][sy+1][sz]/fmaxf(s_rho[sx][sy+1][sz],1e-8f) - vy0);

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

        Emfz[i][j][k] = E_ve - corner + Pe_term;
    }
}
''', 'ct_emf_kernel')
= cp.RawKernel(r''' ... ''', 'ct_emf_kernel')

# ct_curl_kernel = cp.RawKernel(r'''
#define TILE_X 32
#define TILE_Y 4
#define TILE_Z 4
#define PAD 1
#define WARP_SIZE 32

__device__ inline int s_mem_idx(int x, int y, int z) {
    return x + (TILE_X + 2*PAD) * (y + (TILE_Y + 2*PAD) * z);
}

extern "C" __launch_bounds__(512, 4)
__global__ void ct_curl_kernel(
    const float* Emfx, const float* Emfy, const float* Emfz,
    float* Bx, float* By, float* Bz,
    int Ni, float dt, float inv2dx) {

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    int base_i = blockIdx.x * TILE_X;
    int base_j = blockIdx.y * TILE_Y;
    int base_k = blockIdx.z * TILE_Z;

    int i = base_i + tx;
    int j = base_j + ty;
    int k = base_k + tz;

    bool active = (i >= 1 && j >= 1 && k >= 1 && 
                   i < Ni-1 && j < Ni-1 && k < Ni-1);

    int sx = tx + PAD;
    int sy = ty + PAD;
    int sz = tz + PAD;

    __shared__ float s_Emfx[(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];
    __shared__ float s_Emfy[(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];
    __shared__ float s_Emfz[(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];

    int total = (TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD);

    int thread_id = tx + ty * TILE_X + tz * TILE_X * TILE_Y;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;

    for (int idx = warp_id * WARP_SIZE + lane; idx < total; idx += blockDim.x * blockDim.y * blockDim.z) {
        int lx = idx % (TILE_X + 2*PAD);
        int ly = (idx / (TILE_X + 2*PAD)) % (TILE_Y + 2*PAD);
        int lz = idx / ((TILE_X + 2*PAD) * (TILE_Y + 2*PAD));

        int li = base_i + (lx - PAD);
        int lj = base_j + (ly - PAD);
        int lk = base_k + (lz - PAD);

        int sidx = s_mem_idx(lx, ly, lz);

        if (li >= 0 && li < Ni && lj >= 0 && lj < Ni && lk >= 0 && lk < Ni) {
            int gidx = li * Ni * Ni + lj * Ni + lk;
            s_Emfx[sidx] = Emfx[gidx];
            s_Emfy[sidx] = Emfy[gidx];
            s_Emfz[sidx] = Emfz[gidx];
        } else {
            s_Emfx[sidx] = s_Emfy[sidx] = s_Emfz[sidx] = 0.0f;
        }
    }
    __syncthreads();

    if (!active) return;

    int idx = i * Ni * Ni + j * Ni + k;
    int base = s_mem_idx(sx, sy, sz);

    float dEz_dy = (s_Emfz[s_mem_idx(sx, sy+1, sz)] - s_Emfz[s_mem_idx(sx, sy-1, sz)]) * inv2dx;
    float dEy_dz = (s_Emfy[s_mem_idx(sx, sy, sz+1)] - s_Emfy[s_mem_idx(sx, sy, sz-1)]) * inv2dx;
    float dEx_dz = (s_Emfx[s_mem_idx(sx, sy, sz+1)] - s_Emfx[s_mem_idx(sx, sy, sz-1)]) * inv2dx;
    float dEz_dx = (s_Emfz[s_mem_idx(sx+1, sy, sz)] - s_Emfz[s_mem_idx(sx-1, sy, sz)]) * inv2dx;
    float dEy_dx = (s_Emfy[s_mem_idx(sx+1, sy, sz)] - s_Emfy[s_mem_idx(sx-1, sy, sz)]) * inv2dx;
    float dEx_dy = (s_Emfx[s_mem_idx(sx, sy+1, sz)] - s_Emfx[s_mem_idx(sx, sy-1, sz)]) * inv2dx;

    Bx[idx] -= dt * (dEz_dy - dEy_dz);
    By[idx] -= dt * (dEx_dz - dEz_dx);
    Bz[idx] -= dt * (dEy_dx - dEx_dy);
}
''', 'ct_curl_kernel')
= cp.RawKernel(r''' ... ''', 'ct_curl_kernel')

# glm_kernel = cp.RawKernel(r'''
#define TILE_X 32
#define TILE_Y 4
#define TILE_Z 4
#define PAD 1
#define WARP_SIZE 32

__device__ inline int s_mem_idx(int x, int y, int z) {
    return x + (TILE_X + 2*PAD) * (y + (TILE_Y + 2*PAD) * z);
}

extern "C" __launch_bounds__(512, 4)
__global__ void glm_kernel(
    float* Bx, float* By, float* Bz,
    float* psi,
    int Ni, float dt, float dx, float ch, float damping) {

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    int base_i = blockIdx.x * TILE_X;
    int base_j = blockIdx.y * TILE_Y;
    int base_k = blockIdx.z * TILE_Z;

    int i = base_i + tx;
    int j = base_j + ty;
    int k = base_k + tz;

    bool active = (i >= 1 && j >= 1 && k >= 1 && 
                   i < Ni-1 && j < Ni-1 && k < Ni-1);

    int sx = tx + PAD;
    int sy = ty + PAD;
    int sz = tz + PAD;

    __shared__ float s_Bx [(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];
    __shared__ float s_By [(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];
    __shared__ float s_Bz [(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];
    __shared__ float s_psi[(TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD)];

    int total = (TILE_X + 2*PAD) * (TILE_Y + 2*PAD) * (TILE_Z + 2*PAD);

    int thread_id = tx + ty * TILE_X + tz * TILE_X * TILE_Y;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;

    for (int idx = warp_id * WARP_SIZE + lane; idx < total; idx += blockDim.x * blockDim.y * blockDim.z) {
        int lx = idx % (TILE_X + 2*PAD);
        int ly = (idx / (TILE_X + 2*PAD)) % (TILE_Y + 2*PAD);
        int lz = idx / ((TILE_X + 2*PAD) * (TILE_Y + 2*PAD));

        int li = base_i + (lx - PAD);
        int lj = base_j + (ly - PAD);
        int lk = base_k + (lz - PAD);

        int sidx = s_mem_idx(lx, ly, lz);

        if (li >= 0 && li < Ni && lj >= 0 && lj < Ni && lk >= 0 && lk < Ni) {
            int gidx = li * Ni * Ni + lj * Ni + lk;
            s_Bx[sidx]  = Bx[gidx];
            s_By[sidx]  = By[gidx];
            s_Bz[sidx]  = Bz[gidx];
            s_psi[sidx] = psi[gidx];
        } else {
            s_Bx[sidx] = s_By[sidx] = s_Bz[sidx] = s_psi[sidx] = 0.0f;
        }
    }
    __syncthreads();

    if (!active) return;

    int idx = i * Ni * Ni + j * Ni + k;
    int base = s_mem_idx(sx, sy, sz);

    float inv2dx = 0.5f / dx;
    float ch2 = ch * ch;

    float divB = ((s_Bx[s_mem_idx(sx+1,sy,sz)] - s_Bx[s_mem_idx(sx-1,sy,sz)]) +
                  (s_By[s_mem_idx(sx,sy+1,sz)] - s_By[s_mem_idx(sx,sy-1,sz)]) +
                  (s_Bz[s_mem_idx(sx,sy,sz+1)] - s_Bz[s_mem_idx(sx,sy,sz-1)])) * inv2dx;

    psi[idx] -= dt * (ch2 * divB + damping * s_psi[base]);

    Bx[idx] -= dt * (s_psi[s_mem_idx(sx+1,sy,sz)] - s_psi[s_mem_idx(sx-1,sy,sz)]) * inv2dx;
    By[idx] -= dt * (s_psi[s_mem_idx(sx,sy+1,sz)] - s_psi[s_mem_idx(sx,sy-1,sz)]) * inv2dx;
    Bz[idx] -= dt * (s_psi[s_mem_idx(sx,sy,sz+1)] - s_psi[s_mem_idx(sx,sy,sz-1)]) * inv2dx;
}
''', 'glm_kernel')
= cp.RawKernel(r''' ... ''', 'glm_kernel')

# gyrotropic_12_kernel = cp.RawKernel(r'''
#define N_FEAT 12
#define N_FREE 9
#define MAX_NEWTON 32
#define RES_TOL 5e-2f

// Feature evaluation: gyrotropic basis in (c_parallel, c_perp^2)
__device__ __forceinline__ void eval_phi(float cp, float c2, float p[N_FEAT]) {
    float cp2 = cp*cp, cp3 = cp2*cp, cp4 = cp2*cp2;
    float c4 = c2*c2, c6 = c4*c2;
    float idp = 1.f/(1.f+cp2), ide = 1.f/(1.f+c2);
    p[0]=1.f; p[1]=cp; p[2]=c2; p[3]=cp2;
    p[4]=cp*idp; p[5]=cp*c2*ide;
    p[6]=c4; p[7]=cp2*c2; p[8]=cp4;
    p[9]=cp3*idp*idp; p[10]=c6*ide; p[11]=cp2*c4*ide;
}

// Feature scales used for residual normalization
__constant__ float scale[N_FEAT] = {1.f,1.f,1.f,1.f,1.f,1.f,3.f,2.f,3.f,1.f,4.f,3.f};

// Fixed quadrature nodes and weights
__constant__ float np[16] = {-6.5f,-5.5f,-4.5f,-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,-7.5f};
__constant__ float wp[16] = {0.001f,0.004f,0.012f,0.03f,0.07f,0.13f,0.19f,0.19f,0.13f,0.07f,0.03f,0.012f,0.004f,0.001f,0.0003f,0.0003f};
__constant__ float nq[12] = {0.2f,0.55f,1.0f,1.5f,2.1f,2.8f,3.6f,4.5f,5.6f,6.8f,8.2f,10.0f};
__constant__ float wq[12] = {0.06f,0.12f,0.16f,0.17f,0.15f,0.12f,0.08f,0.05f,0.03f,0.015f,0.008f,0.003f};

// Map free parameters (eta, zeta, xi, low) to constrained lambda
__device__ void set_from_free(float eta, float zeta, float xi, const float low[6], float* lambda) {
    float a = __expf(eta), b = __expf(zeta), s = 0.95f * tanhf(xi);
    float A = a*a, B = b*b, C = 2.f*a*b*s;
    lambda[0] = 0.f;
    lambda[1] = low[0]; lambda[2] = low[1]; lambda[3] = low[2];
    lambda[4] = low[3]; lambda[5] = low[4]; lambda[9] = low[5];
    lambda[6] = lambda[10] = -0.5f * B;
    lambda[7] = lambda[11] = -0.5f * C;
    lambda[8] = -A;
}

// Structural tail certificate: positive-definite quartic form
__device__ bool tail_certificate(float eta, float zeta, float xi, float* margin_out) {
    float a = __expf(eta), b = __expf(zeta), s = 0.95f * tanhf(xi);
    float tr = a*a + b*b;
    float detQ = a*a * b*b * (1.f - s*s);
    float disc = (a*a - b*b)*(a*a - b*b) + 4.f*a*a*b*b*s*s;
    float lmax = 0.5f * (tr + sqrtf(fmaxf(disc, 0.f)));
    float lmin = (lmax > 1e-12f) ? detQ / lmax : 0.f;
    float margin = 0.5f - 0.5f * s*s;
    if (margin_out) *margin_out = margin;
    return (detQ > 0.f) && (lmin > 1e-5f * tr) && (margin > 0.04f) && isfinite(detQ);
}

// Recover free parameters from previous lambda (warm start)
__device__ void extract_free(const float* lambda, float* eta, float* zeta, float* xi, float low[6]) {
    float A = fmaxf(-lambda[8], 1e-8f);
    float B = fmaxf(-2.f * lambda[6], 1e-8f);
    *eta  = fmaxf(fminf(0.5f * logf(A), 2.f), -5.f);
    *zeta = fmaxf(fminf(0.5f * logf(B), 2.f), -5.f);
    float a = __expf(*eta), b = __expf(*zeta);
    float ab = fmaxf(a * b, 1e-8f);
    float s = fmaxf(fminf(-lambda[7] / ab, 0.94f), -0.94f);
    *xi = atanhf(s / 0.95f);
    low[0] = lambda[1]; low[1] = lambda[2]; low[2] = lambda[3];
    low[3] = lambda[4]; low[4] = lambda[5]; low[5] = lambda[9];
}

// Parameter-to-multiplier Jacobian
__device__ void jacobian(float eta, float zeta, float xi, float* J) {
    for (int i = 0; i < N_FEAT * N_FREE; ++i) J[i] = 0.f;
    float a = __expf(eta), b = __expf(zeta);
    float th = tanhf(xi), s = 0.95f * th, ds = 0.95f * (1.f - th*th);
    J[8*N_FREE + 0] = -2.f * a*a;
    J[6*N_FREE + 1] = J[10*N_FREE + 1] = -b*b;
    float abs_ = a * b * s;
    J[7*N_FREE + 0] = J[11*N_FREE + 0] = -abs_;
    J[7*N_FREE + 1] = J[11*N_FREE + 1] = -abs_;
    J[7*N_FREE + 2] = J[11*N_FREE + 2] = -a * b * ds;
    J[1*N_FREE + 3] = 1.f; J[2*N_FREE + 4] = 1.f; J[3*N_FREE + 5] = 1.f;
    J[4*N_FREE + 6] = 1.f; J[5*N_FREE + 7] = 1.f; J[9*N_FREE + 8] = 1.f;
}

// Scaled least-squares moment objective
__device__ float moment_objective(const float* M, const float* Mt) {
    float obj = 0.f;
    for (int k = 1; k < N_FEAT; ++k) {
        float r = (M[k] - Mt[k]) / scale[k];
        obj += 0.5f * r * r;
    }
    return obj;
}

// Moments only (no covariance) — used in line search
__device__ bool moments_only(const float* lambda, float ap, float ape, float* M,
                             const float* s_np, const float* s_wp, const float* s_nq, const float* s_wq) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = s_np[ip], c2 = s_nq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cp = s_np[ip], wp_ = s_wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = s_nq[iq]*s_nq[iq], w = wp_*s_wq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            for (int i = 0; i < N_FEAT; ++i) M1[i] += wgt * p[i];
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * iZ;
    return true;
}

// Full moments + covariance matrix
__device__ bool moments(const float* lambda, float ap, float ape, float* M, float* G,
                        const float* s_np, const float* s_wp, const float* s_nq, const float* s_wq) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = s_np[ip], c2 = s_nq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0}, M2[N_FEAT*N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cp = s_np[ip], wp_ = s_wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = s_nq[iq]*s_nq[iq], w = wp_*s_wq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            for (int i = 0; i < N_FEAT; ++i) {
                float wi = wgt * p[i];
                M1[i] += wi;
                for (int j = 0; j <= i; ++j) M2[i*N_FEAT + j] += wi * p[j];
            }
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    for (int i = 0; i < N_FEAT; ++i) {
        M[i] = M1[i] * iZ;
        for (int j = 0; j <= i; ++j) {
            float v = M2[i*N_FEAT + j]*iZ - M[i]*M[j];
            G[i*N_FEAT + j] = G[j*N_FEAT + i] = v;
        }
    }
    return true;
}

// Final moments + true third-order heat-flux accumulators
__device__ bool moments_final(const float* lambda, float ap, float ape, float* M,
                              float* m3_par, float* m3_mix,
                              const float* s_np, const float* s_wp, const float* s_nq, const float* s_wq) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = s_np[ip], c2 = s_nq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0}, acc_par = 0.f, acc_mix = 0.f;
    for (int ip = 0; ip < 16; ++ip) {
        float cp = s_np[ip], wp_ = s_wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = s_nq[iq]*s_nq[iq], w = wp_*s_wq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            acc_par += wgt * (cp*cp*cp);
            acc_mix += wgt * (cp * c2);
            for (int i = 0; i < N_FEAT; ++i) M1[i] += wgt * p[i];
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * iZ;
    *m3_par = acc_par * iZ;
    *m3_mix = acc_mix * iZ;
    return true;
}

// 9x9 Cholesky solve
__device__ bool chol9(float* H, float* rhs, float* dx) {
    float L[N_FREE*N_FREE] = {0};
    for (int i = 0; i < N_FREE; ++i)
        for (int j = 0; j <= i; ++j) {
            float s = H[i*N_FREE + j];
            for (int k = 0; k < j; ++k) s -= L[i*N_FREE + k] * L[j*N_FREE + k];
            if (i == j) {
                if (!(s > 1e-12f)) return false;
                L[i*N_FREE + j] = sqrtf(s);
            } else L[i*N_FREE + j] = s / L[j*N_FREE + j];
        }
    float y[N_FREE];
    for (int i = 0; i < N_FREE; ++i) {
        float s = rhs[i];
        for (int k = 0; k < i; ++k) s -= L[i*N_FREE + k] * y[k];
        y[i] = s / L[i*N_FREE + i];
    }
    for (int i = N_FREE-1; i >= 0; --i) {
        float s = y[i];
        for (int k = i+1; k < N_FREE; ++k) s -= L[k*N_FREE + i] * dx[k];
        dx[i] = s / L[i*N_FREE + i];
    }
    return true;
}

extern "C" __global__
__launch_bounds__(64, 2)
void gyrotropic_12_kernel(
    const float* __restrict__ M_target,
    const float* __restrict__ n_in,
    float* __restrict__ lambda_io,
    float* __restrict__ n_out,
    float* __restrict__ upar_out,
    float* __restrict__ Ppar_out,
    float* __restrict__ Pperp_out,
    float* __restrict__ qpar_out,
    float* __restrict__ qperp_out,
    float* __restrict__ margin_out,
    float* __restrict__ M_recon,
    int* __restrict__ status,
    int Ncells,
    float mass,
    float thermal_scale,
    float alpha_par,
    float alpha_perp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncells) return;

    // Shared memory for quadrature (loaded once per block)
    __shared__ float s_np[16], s_wp[16], s_nq[12], s_wq[12];
    if (threadIdx.x < 16) { s_np[threadIdx.x] = np[threadIdx.x]; s_wp[threadIdx.x] = wp[threadIdx.x]; }
    if (threadIdx.x < 12) { s_nq[threadIdx.x] = nq[threadIdx.x]; s_wq[threadIdx.x] = wq[threadIdx.x]; }
    __syncthreads();

    // Load target moments
    float Mt[N_FEAT];
    const float4* Mt4 = reinterpret_cast<const float4*>(M_target + idx * N_FEAT);
    float4 t0 = Mt4[0], t1 = Mt4[1], t2 = Mt4[2];
    Mt[0]=t0.x; Mt[1]=t0.y; Mt[2]=t0.z; Mt[3]=t0.w;
    Mt[4]=t1.x; Mt[5]=t1.y; Mt[6]=t1.z; Mt[7]=t1.w;
    Mt[8]=t2.x; Mt[9]=t2.y; Mt[10]=t2.z; Mt[11]=t2.w;

    float* lo = lambda_io + idx * N_FEAT;
    int flag = 0;

    // Basic input validity
    float n = n_in[idx];
    if (!(n > 0.f) || !(mass > 0.f) || !(thermal_scale > 0.f)) flag = 5;
    if (Mt[2] < 1e-8f || Mt[3] < 1e-8f) flag = 5;

    // Warm-start free parameters
    float eta, zeta, xi, low[6];
    extract_free(lo, &eta, &zeta, &xi, low);

    float margin = 0.f;
    bool tail_ok = tail_certificate(eta, zeta, xi, &margin);
    if (!tail_ok && flag == 0) flag = 7;

    float ap = fmaxf(alpha_par, 0.3f), ae = fmaxf(alpha_perp, 0.3f);
    float lambda_lm = 1e-3f;
    int chol_fails = 0, ls_fails = 0;
    float lambda[N_FEAT], M[N_FEAT];

    // Gauss-Newton loop
    if (flag == 0) {
        for (int it = 0; it < MAX_NEWTON; ++it) {
            set_from_free(eta, zeta, xi, low, lambda);
            float G[N_FEAT * N_FEAT];
            if (!moments(lambda, ap, ae, M, G, s_np, s_wp, s_nq, s_wq)) { flag = 6; break; }

            float obj = moment_objective(M, Mt);
            float max_r = 0.f;
            for (int k = 1; k < N_FEAT; ++k)
                max_r = fmaxf(max_r, fabsf((M[k] - Mt[k]) / scale[k]));
            if (max_r < RES_TOL) { flag = 0; break; }

            float J[N_FEAT * N_FREE];
            jacobian(eta, zeta, xi, J);

            // Build D = G J (scaled) and normal equations
            float H[N_FREE * N_FREE] = {0}, rhs[N_FREE] = {0}, dx[N_FREE];
            for (int i = 1; i < N_FEAT; ++i) {
                float ri = (M[i] - Mt[i]) / scale[i];
                float Di[N_FREE];
                for (int p = 0; p < N_FREE; ++p) {
                    float s = 0.f;
                    for (int j = 0; j < N_FEAT; ++j)
                        s += (G[i*N_FEAT + j] * J[j*N_FREE + p]) / scale[i];
                    Di[p] = s;
                    rhs[p] += s * ri;
                }
                for (int p = 0; p < N_FREE; ++p)
                    for (int q = 0; q < N_FREE; ++q)
                        H[p*N_FREE + q] += Di[p] * Di[q];
            }
            float max_diag = 1e-8f;
            for (int p = 0; p < N_FREE; ++p) max_diag = fmaxf(max_diag, H[p*N_FREE + p]);
            float ridge = lambda_lm * max_diag;
            for (int p = 0; p < N_FREE; ++p) H[p*N_FREE + p] += ridge;

            if (!chol9(H, rhs, dx)) {
                lambda_lm *= 4.f;
                if (++chol_fails > 4) { flag = 2; break; }
                continue;
            }
            chol_fails = 0;

            // Line search on moment objective
            float al = 0.8f;
            bool accepted = false;
            for (int ls = 0; ls < 4; ++ls) {
                float e2 = eta - al*dx[0], z2 = zeta - al*dx[1], x2 = xi - al*dx[2];
                if (e2 < -5.f || e2 > 2.f || z2 < -5.f || z2 > 2.f) { al *= 0.5f; continue; }
                float l2[6];
                for (int k = 0; k < 6; ++k) l2[k] = low[k] - al*dx[3+k];
                float lam2[N_FEAT], M2[N_FEAT];
                set_from_free(e2, z2, x2, l2, lam2);
                if (!moments_only(lam2, ap, ae, M2, s_np, s_wp, s_nq, s_wq)) { al *= 0.5f; continue; }
                if (moment_objective(M2, Mt) < obj) {
                    eta = e2; zeta = z2; xi = x2;
                    for (int k = 0; k < 6; ++k) low[k] = l2[k];
                    accepted = true;
                    break;
                }
                al *= 0.5f;
            }
            if (!accepted) {
                if (++ls_fails > 3) { flag = 3; break; }
                lambda_lm = fminf(lambda_lm * 4.f, 10.f);
            } else ls_fails = 0;

            if (it == MAX_NEWTON - 1 && flag == 0) flag = 1;
        }
    }

    // Final reconstruction and heat-flux moments
    float m3_par = 0.f, m3_mix = 0.f;
    set_from_free(eta, zeta, xi, low, lambda);
    bool ok_final = moments_final(lambda, ap, ae, M, &m3_par, &m3_mix, s_np, s_wp, s_nq, s_wq);
    if (!ok_final && flag == 0) flag = 6;

    tail_ok = tail_certificate(eta, zeta, xi, &margin);
    if (!tail_ok && flag == 0) flag = 7;

    // Final residual check
    if (ok_final && (flag == 0 || flag == 1)) {
        float moment_r = 0.f;
        for (int k = 1; k < N_FEAT; ++k)
            moment_r = fmaxf(moment_r, fabsf((M[k] - Mt[k]) / scale[k]));
        if (moment_r < RES_TOL && tail_ok) flag = 0;
        else if (flag == 0) flag = 1;
    }

    // Transactional warm-start commit
    if (flag == 0)
        for (int k = 0; k < N_FEAT; ++k) lo[k] = lambda[k];

    status[idx] = flag;
    margin_out[idx] = margin;

    float* Mr = M_recon + idx * N_FEAT;
    for (int k = 0; k < N_FEAT; ++k) Mr[k] = ok_final ? M[k] : 0.f;

    // Physical outputs (only on success)
    bool valid = (flag == 0) && ok_final && tail_ok;
    float vth2 = thermal_scale * thermal_scale;
    float vth3 = vth2 * thermal_scale;
    float up = valid ? M[1] : 0.f;
    float cp2 = valid ? fmaxf(M[3] - up*up, 1e-6f) : 1e-4f;
    float ce2 = valid ? fmaxf(M[2], 1e-6f) : 1e-4f;

    float qpar = 0.f, qperp = 0.f;
    if (valid) {
        qpar  = mass * n * vth3 * (m3_par - 3.f*up*cp2 - up*up*up);
        qperp = 0.5f * mass * n * vth3 * (m3_mix - up * M[2]);
    }

    n_out[idx] = n;
    upar_out[idx] = up * thermal_scale;
    Ppar_out[idx] = mass * n * vth2 * cp2;
    Pperp_out[idx] = 0.5f * mass * n * vth2 * ce2;
    qpar_out[idx] = qpar;
    qperp_out[idx] = qperp;
}
''', 'gyrotropic_12_kernel')

# ---------- Base Gaussian test (v0.1) ----------
def compute_base_moments(alpha_par=0.5, alpha_perp=0.5):
    np_h = np.array([-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,-7.5], dtype=np.float32)
    wp_h = np.array([0.001,0.004,0.012,0.03,0.07,0.13,0.19,0.19,0.13,0.07,0.03,0.012,0.004,0.001,0.0003,0.0003], dtype=np.float32)
    nq_h = np.array([0.2,0.55,1.0,1.5,2.1,2.8,3.6,4.5,5.6,6.8,8.2,10.0], dtype=np.float32)
    wq_h = np.array([0.06,0.12,0.16,0.17,0.15,0.12,0.08,0.05,0.03,0.015,0.008,0.003], dtype=np.float32)
    def phi(cp, c2):
        cp2 = cp*cp; idp = 1.0/(1.0+cp2); ide = 1.0/(1.0+c2)
        return np.array([1, cp, c2, cp2, cp*idp, cp*c2*ide, c2*c2, cp2*c2, cp2*cp2,
                         (cp**3)*idp*idp, (c2**3)*ide, cp2*(c2**2)*ide], dtype=np.float32)
    Z, M = 0.0, np.zeros(12, dtype=np.float32)
    for ip in range(16):
        for iq in range(12):
            cp, c2 = np_h[ip], nq_h[iq]**2
            w = wp_h[ip] * wq_h[iq] * nq_h[iq]
            wgt = np.exp(-alpha_par*cp*cp - alpha_perp*c2) * w
            Z += wgt; M += wgt * phi(cp, c2)
    return M / Z

M_base = compute_base_moments()
Ncells = 128
M_target = cp.tile(cp.asarray(M_base), (Ncells, 1)).astype(cp.float32)
n_in = cp.ones(Ncells, dtype=cp.float32)
lambda_io = cp.zeros((Ncells, 12), dtype=cp.float32)
lambda_io[:,6] = lambda_io[:,8] = lambda_io[:,10] = -0.05

n_out = cp.empty(Ncells, dtype=cp.float32)
upar = cp.empty(Ncells, dtype=cp.float32)
Ppar = cp.empty(Ncells, dtype=cp.float32)
Pperp = cp.empty(Ncells, dtype=cp.float32)
qpar = cp.empty(Ncells, dtype=cp.float32)
qperp = cp.empty(Ncells, dtype=cp.float32)
margin = cp.empty(Ncells, dtype=cp.float32)
M_recon = cp.empty((Ncells, 12), dtype=cp.float32)
status = cp.zeros(Ncells, dtype=cp.int32)

threads, blocks = 64, (Ncells + 63) // 64
gyrotropic_12_kernel(
    (blocks,), (threads,),
    (M_target, n_in, lambda_io,
     n_out, upar, Ppar, Pperp, qpar, qperp,
     margin, M_recon, status, Ncells,
     np.float32(1.0), np.float32(1.0),
     np.float32(0.5), np.float32(0.5))
)

print("status unique:", cp.unique(status).get())
print("Ppar sample:", Ppar[:2].get())
print("Pperp sample:", Pperp[:2].get())
print("max |M_recon - target|:", float(cp.max(cp.abs(M_recon - M_target))))
print("margin sample:", margin[:2].get())

# ---------- Residual diagnostic ----------
# Per-feature absolute errors and scaled residual
err = M_recon - M_target
scale = cp.asarray([1,1,1,1,1,1,3,2,3,1,4,3], dtype=cp.float32)

print("Mean absolute errors:", cp.mean(cp.abs(err), axis=0).get())
print("Maximum absolute errors:", cp.max(cp.abs(err), axis=0).get())
print("Maximum scaled residual:", cp.max(cp.abs(err) / scale).get())
= cp.RawKernel(r''' ... ''', 'gyrotropic_12_kernel')

# ---------- Grid & parameters ----------
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
n_steps = 50

# ---------- Allocate ----------
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

# ---------- Disk initial condition ----------
def init_disk():
    i = cp.arange(Ni, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(i, i, i, indexing='ij')
    R = cp.sqrt((X - Ni/2)**2 + (Y - Ni/2)**2) + 1e-3
    rho[:] = cp.exp(-R / 12.0) * cp.exp(-cp.abs(Z - Ni/2) / 4.0)
    rho[:] = cp.clip(rho, 0.05, None)
    vphi = cp.sqrt(1.0 / (R + 2.0))          # toy rotation curve
    mx[:] = -vphi * (Y - Ni/2) / R * rho
    my[:] =  vphi * (X - Ni/2) / R * rho
    Bz[:] = 0.05

init_disk()

# ---------- Helpers ----------
threads   = (32, 4, 4)
blocks    = ((Ni + 31)//32, (Ni + 3)//4, (Ni + 3)//4)
threads_g = 128
blocks_g  = (Ncells + 127) // 128

def gravity_force(rho):
    i = cp.arange(Ni, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(i, i, i, indexing='ij')
    R = cp.sqrt((X - Ni/2)**2 + (Y - Ni/2)**2 + (Z - Ni/2)**2) * dx + 1e-3
    return -rho / (R**2 + 1.0)

def build_M_target():
    M_target[:, 0] = 1.0
    M_target[:, 2] = cp.clip(Pperp / (rho.ravel() + 1e-8), 1e-6, 5.0)
    M_target[:, 3] = cp.clip(Ppar  / (rho.ravel() + 1e-8), 1e-6, 5.0)

def write_pe():
    pe_xx[:] = Pperp.reshape(Ni, Ni, Ni)
    pe_yy[:] = Pperp.reshape(Ni, Ni, Ni)
    pe_zz[:] = Ppar.reshape(Ni, Ni, Ni)

def fluid_forces():
    jx = (cp.roll(By, -1, 2) - cp.roll(By, 1, 2) -
          (cp.roll(Bz, -1, 1) - cp.roll(Bz, 1, 1))) * 0.5 / dx
    jy = (cp.roll(Bz, -1, 2) - cp.roll(Bz, 1, 2) -
          (cp.roll(Bx, -1, 1) - cp.roll(Bx, 1, 1))) * 0.5 / dx
    jz = (cp.roll(Bx, -1, 1) - cp.roll(Bx, 1, 1) -
          (cp.roll(By, -1, 2) - cp.roll(By, 1, 2))) * 0.5 / dx

    Fx = jy*Bz - jz*By - (cp.roll(pe_xx, -1, 0) - cp.roll(pe_xx, 1, 0))*0.5/dx
    Fy = jz*Bx - jx*Bz - (cp.roll(pe_yy, -1, 1) - cp.roll(pe_yy, 1, 1))*0.5/dx
    Fz = jx*By - jy*Bx - (cp.roll(pe_zz, -1, 2) - cp.roll(pe_zz, 1, 2))*0.5/dx

    g = gravity_force(rho)
    Fx += g * (cp.arange(Ni)[:,None,None] - Ni/2)
    Fy += g * (cp.arange(Ni)[None,:,None] - Ni/2)
    Fz += g * (cp.arange(Ni)[None,None,:] - Ni/2)
    return Fx, Fy, Fz

# ---------- CUDA Graph (four kernels) ----------
stream = cp.cuda.Stream()
graph  = cp.cuda.graph.Graph()

with stream:
    with graph.capture():
        gyrotropic_12_kernel(
            (blocks_g,), (threads_g,),
            (M_target, n_in, lambda_io,
             n_out, upar, Ppar, Pperp, qpar, qperp,
             status, Ncells, mass, thermal_scale, alpha_par, alpha_perp)
        )
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

instance = graph.instantiate()

# ---------- RK3 step ----------
def rk3_step():
    # stage 1
    build_M_target()
    instance.launch(stream)
    stream.synchronize()
    write_pe()
    Fx, Fy, Fz = fluid_forces()
    mx1 = mx + dt * Fx
    my1 = my + dt * Fy
    mz1 = mz + dt * Fz

    # stage 2
    mx[:], my[:], mz[:] = mx1, my1, mz1
    build_M_target()
    instance.launch(stream)
    stream.synchronize()
    write_pe()
    Fx, Fy, Fz = fluid_forces()
    mx2 = 0.75*mx + 0.25*(mx1 + dt * Fx)
    my2 = 0.75*my + 0.25*(my1 + dt * Fy)
    mz2 = 0.75*mz + 0.25*(mz1 + dt * Fz)

    # stage 3
    mx[:], my[:], mz[:] = mx2, my2, mz2
    build_M_target()
    instance.launch(stream)
    stream.synchronize()
    write_pe()
    Fx, Fy, Fz = fluid_forces()
    mx[:] = (1./3.)*mx + (2./3.)*(mx2 + dt * Fx)
    my[:] = (1./3.)*my + (2./3.)*(my2 + dt * Fy)
    mz[:] = (1./3.)*mz + (2./3.)*(mz2 + dt * Fz)

# ---------- Main ----------
print("Plasma simulator starting ...")
for step in range(n_steps):
    rk3_step()
    if step % 10 == 0:
        vphi = cp.sqrt(mx**2 + my**2) / (rho + 1e-8)
        print(f"step {step:3d}  status={cp.unique(status)}  "
              f"mean v_phi={float(cp.mean(vphi)):.4f}  "
              f"mean |B|={float(cp.mean(cp.sqrt(Bx**2+By**2+Bz**2))):.4f}")

print("Done.")
print("Final mean Ppar / Pperp =", float(cp.mean(Ppar)), float(cp.mean(Pperp)))
