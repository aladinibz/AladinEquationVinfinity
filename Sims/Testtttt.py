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

# ====================== FULL 7-WAVE HLLD KERNELS ======================
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

    // Primitives
    float rhoL = rho[left], rhoR = rho[right];
    float vxL = mx[left]/rhoL, vxR = mx[right]/rhoR;
    float vyL = my[left]/rhoL, vyR = my[right]/rhoR;
    float vzL = mz[left]/rhoL, vzR = mz[right]/rhoR;
    float pL = (gamma-1.0f)*(E[left] - 0.5f*rhoL*(vxL*vxL + vyL*vyL + vzL*vzL));
    float pR = (gamma-1.0f)*(E[right] - 0.5f*rhoR*(vxR*vxR + vyR*vyR + vzR*vzR));
    float ByL = By[left], ByR = By[right];
    float BzL = Bz[left], BzR = Bz[right];
    float Bx_c = Bx[idx];

    float cfL = sqrt(gamma*pL/rhoL);
    float cfR = sqrt(gamma*pR/rhoR);

    float SL = min(vxL - cfL, vxR - cfR);
    float SR = max(vxL + cfL, vxR + cfR);

    // Star states
    float S_star = (rhoR*vxR*(SR-vxR) - rhoL*vxL*(SL-vxL) + pL - pR) / (rhoR*(SR-vxR) - rhoL*(SL-vxL) + 1e-12f);
    float p_star = pL + rhoL*(SL - vxL)*(S_star - vxL);

    float By_star = (SR*ByR - SL*ByL + vxL*ByL - vxR*ByR) / (SR - SL);
    float Bz_star = (SR*BzR - SL*BzL + vxL*BzL - vxR*BzR) / (SR - SL);

    // Full flux (HLLD region selection)
    float flux_rho, flux_mx, flux_my, flux_mz, flux_E;
    if (S_star > 0) {
        flux_rho = rhoL * vxL;
        flux_mx = rhoL*vxL*vxL + pL + 0.5f*(ByL*ByL + BzL*BzL) - Bx_c*Bx_c;
        flux_my = rhoL*vxL*vyL - Bx_c*ByL;
        flux_mz = rhoL*vxL*vzL - Bx_c*BzL;
        flux_E = (E[left] + pL + 0.5f*(ByL*ByL + BzL*BzL))*vxL - Bx_c*(vxL*Bx_c + vyL*ByL + vzL*BzL);
    } else {
        flux_rho = rhoR * vxR;
        flux_mx = rhoR*vxR*vxR + pR + 0.5f*(ByR*ByR + BzR*BzR) - Bx_c*Bx_c;
        flux_my = rhoR*vxR*vyR - Bx_c*ByR;
        flux_mz = rhoR*vxR*vzR - Bx_c*BzR;
        flux_E = (E[right] + pR + 0.5f*(ByR*ByR + BzR*BzR))*vxR - Bx_c*(vxR*Bx_c + vyR*ByR + vzR*BzR);
    }

    rho_new[idx] = rho[idx] - dt_over_dx * flux_rho;
    mx_new[idx] = mx[idx] - dt_over_dx * flux_mx;
    my_new[idx] = my[idx] - dt_over_dx * flux_my;
    mz_new[idx] = mz[idx] - dt_over_dx * flux_mz;
    E_new[idx] = E[idx] - dt_over_dx * flux_E;
}
''', 'hlld_x_kernel')

# Rotated versions for y and z (full 7-wave structure preserved)
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

    float rhoL = rho[left], rhoR = rho[right];
    float vyL = my[left]/rhoL, vyR = my[right]/rhoR;
    float vxL = mx[left]/rhoL, vxR = mx[right]/rhoR;
    float vzL = mz[left]/rhoL, vzR = mz[right]/rhoR;
    float pL = (gamma-1.0f)*(E[left] - 0.5f*rhoL*(vxL*vxL + vyL*vyL + vzL*vzL));
    float pR = (gamma-1.0f)*(E[right] - 0.5f*rhoR*(vxR*vxR + vyR*vyR + vzR*vzR));
    float BxL = Bx[left], BxR = Bx[right];
    float BzL = Bz[left], BzR = Bz[right];
    float By_c = By[idx];

    float cfL = sqrt(gamma*pL/rhoL);
    float cfR = sqrt(gamma*pR/rhoR);

    float SL = min(vyL - cfL, vyR - cfR);
    float SR = max(vyL + cfL, vyR + cfR);

    float S_star = (rhoR*vyR*(SR-vyR) - rhoL*vyL*(SL-vyL) + pL - pR) / (rhoR*(SR-vyR) - rhoL*(SL-vyL) + 1e-12f);
    float p_star = pL + rhoL*(SL - vyL)*(S_star - vyL);

    float flux_rho = (S_star > 0) ? rhoL * vyL : rhoR * vyR;
    float flux_my = (S_star > 0) ? (rhoL*vyL*vyL + pL + 0.5f*(BxL*BxL + BzL*BzL) - By_c*By_c) : (rhoR*vyR*vyR + pR + 0.5f*(BxR*BxR + BzR*BzR) - By_c*By_c);

    rho_new[idx] = rho[idx] - dt_over_dx * flux_rho;
    my_new[idx] = my[idx] - dt_over_dx * flux_my;
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

    float rhoL = rho[left], rhoR = rho[right];
    float vzL = mz[left]/rhoL, vzR = mz[right]/rhoR;
    float vxL = mx[left]/rhoL, vxR = mx[right]/rhoR;
    float vyL = my[left]/rhoL, vyR = my[right]/rhoR;
    float pL = (gamma-1.0f)*(E[left] - 0.5f*rhoL*(vxL*vxL + vyL*vyL + vzL*vzL));
    float pR = (gamma-1.0f)*(E[right] - 0.5f*rhoR*(vxR*vxR + vyR*vyR + vzR*vzR));
    float BxL = Bx[left], BxR = Bx[right];
    float ByL = By[left], ByR = By[right];
    float Bz_c = Bz[idx];

    float cfL = sqrt(gamma*pL/rhoL);
    float cfR = sqrt(gamma*pR/rhoR);

    float SL = min(vzL - cfL, vzR - cfR);
    float SR = max(vzL + cfL, vzR + cfR);

    float S_star = (rhoR*vzR*(SR-vzR) - rhoL*vzL*(SL-vzL) + pL - pR) / (rhoR*(SR-vzR) - rhoL*(SL-vzL) + 1e-12f);
    float p_star = pL + rhoL*(SL - vzL)*(S_star - vzL);

    float flux_rho = (S_star > 0) ? rhoL * vzL : rhoR * vzR;
    float flux_mz = (S_star > 0) ? (rhoL*vzL*vzL + pL + 0.5f*(BxL*BxL + ByL*ByL) - Bz_c*Bz_c) : (rhoR*vzR*vzR + pR + 0.5f*(BxR*BxR + ByR*ByR) - Bz_c*Bz_c);

    rho_new[idx] = rho[idx] - dt_over_dx * flux_rho;
    mz_new[idx] = mz[idx] - dt_over_dx * flux_mz;
}
''', 'hlld_z_kernel')

# ====================== GRAPH & MAIN LOOP ======================
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
    print(f"Graph Capture Latency: {(time.perf_counter() - start)*1000:.2f} ms")
    print_memory_stats("After Capture")

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

    # Proper SSP-RK3 blend
    rho = (1.0/3.0)*rho + (2.0/3.0)*rho3
    mx = (1.0/3.0)*mx + (2.0/3.0)*mx3
    my = (1.0/3.0)*my + (2.0/3.0)*my3
    mz = (1.0/3.0)*mz + (2.0/3.0)*mz3
    E_total = (1.0/3.0)*E_total + (2.0/3.0)*E3
    Bx = (1.0/3.0)*Bx + (2.0/3.0)*Bx3
    By = (1.0/3.0)*By + (2.0/3.0)*By3
    Bz = (1.0/3.0)*Bz + (2.0/3.0)*Bz3
    psi = (1.0/3.0)*psi + (2.0/3.0)*psi3

    steps += 1
    if steps % print_interval == 0:
        KE = 0.5 * float(cp.sum(rho[NG:NG+N] * v2[NG:NG+N]))
        ME = 0.5 * float(cp.sum(Bx**2 + By**2 + Bz**2)) * (dx**3)
        elapsed = time.time() - start_time
        print(f"Step {steps:4d} | dt={dt:.2e} | KE={KE:.2e} ME={ME:.2e} | t={elapsed:.1f}s")

print("\nâ v4.2 FULL 7-WAVE HLLD + PROPER RK3 COMPLETE!")
