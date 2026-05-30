ct_curl_kernel = cp.RawKernel(r'''
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
