glm_kernel = cp.RawKernel(r'''
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
