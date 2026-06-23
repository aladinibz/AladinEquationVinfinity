import cupy as cp

ct_entropy_solver_14momentSOTA = cp.RawKernel(r'''
#define TILE_X 32
#define TILE_Y 8
#define TILE_Z 4
#define PAD 1
#define EPS 1e-12f
#define MAX_FLUX 1e6f
#define WARP_SIZE 32
#define GLM_BASE_CH 1.0f
#define GLM_DAMPING 0.1f
#define SINGULAR_TOL 1e-8f

// === Constants for ML part ===
#define ORIGINAL_DIM 15
#define MAX_K 16
#define FFT_SIZE 16

__constant__ float d_embedding_basis[ORIGINAL_DIM * MAX_K];
__constant__ float d_spectral_weights[FFT_SIZE];

// === Original helpers (unchanged) ===
__device__ inline int periodic_idx(int i, int N) {
    return (i + N) % N;
}

__device__ float det_P(float Pxx, float Pxy, float Pxz,
                       float Pyy, float Pyz, float Pzz) {
    return Pxx * (Pyy * Pzz - Pyz * Pyz) -
           Pxy * (Pxy * Pzz - Pxz * Pyz) +
           Pxz * (Pxy * Pyz - Pyy * Pxz);
}

__device__ float quadratic_heat_flux_correction(float* P, float* q) {
    float a = P[0], b = P[1], c = P[2];
    float d = P[3], e = P[4], f = P[5];
    float det = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);
    if (fabsf(det) < SINGULAR_TOL) return 0.0f;
    float inv_det = 1.0f / det;
    float Pinv_xx = (d*f - e*e) * inv_det;
    float Pinv_yy = (a*f - c*c) * inv_det;
    float Pinv_zz = (a*d - b*b) * inv_det;
    return q[0]*q[0]*Pinv_xx + q[1]*q[1]*Pinv_yy + q[2]*q[2]*Pinv_zz;
}

__device__ float convex_entropy_14moment(float rho, float* P, float* q, float kurtosis) {
    float detp = det_P(P[0], P[1], P[2], P[3], P[4], P[5]);
    float q2_Pinv = quadratic_heat_flux_correction(P, q);

    float trace_P = P[0] + P[3] + P[5];
    float kurtosis_term = 0.0f;
    if (rho > EPS && trace_P > EPS) {
        float r_norm = kurtosis / (trace_P * trace_P / rho);
        float dev = r_norm - 15.0f;
        kurtosis_term = -0.05f * dev * dev * dev;
    }

    return -rho * logf(fmaxf(rho, EPS)) 
           + 0.5f * logf(fmaxf(detp, EPS)) 
           - 0.5f * q2_Pinv / fmaxf(rho, EPS)
           + kurtosis_term;
}

__device__ void entropy_variable_projection(float* rho, float* P, float* q, float* r, float entropy_force) {
    *rho = fmaxf(*rho, EPS);
    refined_log_cholesky(&P[0], &P[1], &P[2], &P[3], &P[4], &P[5]);

    float trace_P = P[0] + P[3] + P[5];
    float q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];

    float sigma = q2 / (fmaxf(*rho * trace_P * trace_P * trace_P, EPS));
    sigma = fminf(fmaxf(sigma, 0.0f), 1.0f);

    float relax = fmaxf(0.90f, 1.0f - 0.5f * sigma - 0.25f * fabsf(entropy_force));

    float max_q2 = 2.0f * trace_P * trace_P * trace_P * *rho;
    if (q2 > max_q2) {
        float scale = sqrtf(max_q2 / (q2 + EPS));
        q[0] *= scale * relax;
        q[1] *= scale * relax;
        q[2] *= scale * relax;
    }

    float lower_r = 3.0f * trace_P * trace_P / *rho;
    float upper_r = 15.0f * trace_P * trace_P / *rho;
    *r = fmaxf(fminf(*r, upper_r), lower_r);
}

__device__ void refined_log_cholesky(float* Pxx, float* Pxy, float* Pxz,
                                     float* Pyy, float* Pyz, float* Pzz) {
    float min_eig = EPS * 1e-4f;
    float pxx = fmaxf(*Pxx, min_eig);
    float pyy = fmaxf(*Pyy, min_eig);
    float pzz = fmaxf(*Pzz, min_eig);

    float l11 = sqrtf(pxx);
    float l21 = *Pxy / l11;
    float l31 = *Pxz / l11;
    float l22 = sqrtf(fmaxf(pyy - l21*l21, min_eig));
    float l32 = (*Pyz - l21*l31) / l22;
    float l33 = sqrtf(fmaxf(pzz - l31*l31 - l32*l32, min_eig));

    float strength = 0.96f;

    *Pxx = strength * l11*l11 + (1.0f - strength) * pxx;
    *Pxy = strength * l11*l21 + (1.0f - strength) * *Pxy;
    *Pxz = strength * l11*l31 + (1.0f - strength) * *Pxz;
    *Pyy = strength * (l21*l21 + l22*l22) + (1.0f - strength) * pyy;
    *Pyz = strength * (l21*l31 + l22*l32) + (1.0f - strength) * *Pyz;
    *Pzz = strength * (l31*l31 + l32*l32 + l33*l33) + (1.0f - strength) * pzz;
}

// === ML Helpers (Embedding + Spectral FNO) ===
__device__ void project_to_embedding(float* moment, float* embedded) {
    for (int d = 0; d < MAX_K; d++) {
        embedded[d] = 0.0f;
        for (int m = 0; m < ORIGINAL_DIM; m++) {
            embedded[d] += d_embedding_basis[m * MAX_K + d] * moment[m];
        }
    }
}

__device__ void manual_small_fft(float* data_in, float* data_out, int size) {
    for (int i = 0; i < size; i++) {
        data_out[i] = 0.0f;
        for (int j = 0; j < size; j++) {
            float angle = -2.0f * 3.1415926535f * i * j / (float)size;
            data_out[i] += data_in[j % MAX_K] * cosf(angle);
        }
    }
}

__device__ float spectral_fno_correction(float* embedded) {
    float freq[FFT_SIZE];
    manual_small_fft(embedded, freq, FFT_SIZE);

    float weighted = 0.0f;
    for (int f = 0; f < FFT_SIZE; f++) {
        weighted += freq[f] * d_spectral_weights[f];
    }
    return weighted * 0.1f;
}

// === GLM with shared memory ===
__device__ void entropy_modulated_glm(float* Bx, float* By, float* Bz, float* psi,
                                      float dt, float dx, float entropy_force,
                                      int i, int j, int k, int Ni) {
    float ch = GLM_BASE_CH * (1.0f + 0.15f * fabsf(entropy_force));
    float damping = fmaxf(0.01f, GLM_DAMPING * (1.0f - 0.3f * fabsf(entropy_force)));

    float ch2 = ch * ch;
    float inv2dx = 0.5f / dx;

    __shared__ float s_Bx[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_By[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Bz[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_psi[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int sx = tx + PAD; int sy = ty + PAD; int sz = tz + PAD;

    int gidx = i * Ni * Ni + j * Ni + k;

    s_Bx[sx][sy][sz] = Bx[gidx];
    s_By[sx][sy][sz] = By[gidx];
    s_Bz[sx][sy][sz] = Bz[gidx];
    s_psi[sx][sy][sz] = psi[gidx];
    __syncthreads();

    float divB = ((s_Bx[sx+1][sy][sz] - s_Bx[sx-1][sy][sz]) +
                  (s_By[sx][sy+1][sz] - s_By[sx][sy-1][sz]) +
                  (s_Bz[sx][sy][sz+1] - s_Bz[sx][sy][sz-1])) * inv2dx;

    float psi_val = s_psi[sx][sy][sz];
    psi_val -= dt * (ch2 * divB + damping * psi_val);
    *psi = psi_val;

    *Bx -= dt * (s_psi[sx+1][sy][sz] - s_psi[sx-1][sy][sz]) * inv2dx;
    *By -= dt * (s_psi[sx][sy+1][sz] - s_psi[sx][sy-1][sz]) * inv2dx;
    *Bz -= dt * (s_psi[sx][sy][sz+1] - s_psi[sx][sy][sz-1]) * inv2dx;
}

// === Curl EMF with shared memory ===
__device__ void ct_curl_emf(float* Bx, float* By, float* Bz,
                            float* Ex, float* Ey, float* Ez,
                            float dt, float dx, int i, int j, int k, int Ni) {
    float inv_dx = 1.0f / dx;

    __shared__ float s_Ex[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Ey[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Ez[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int sx = tx + PAD; int sy = ty + PAD; int sz = tz + PAD;

    int gidx = i * Ni * Ni + j * Ni + k;

    s_Ex[sx][sy][sz] = Ex[gidx];
    s_Ey[sx][sy][sz] = Ey[gidx];
    s_Ez[sx][sy][sz] = Ez[gidx];
    __syncthreads();

    float dEy_dz = (s_Ey[sx][sy][sz+1] - s_Ey[sx][sy][sz-1]) * 0.5f * inv_dx;
    float dEz_dy = (s_Ez[sx][sy+1][sz] - s_Ez[sx][sy-1][sz]) * 0.5f * inv_dx;
    *Bx += dt * (dEy_dz - dEz_dy);

    float dEz_dx = (s_Ez[sx+1][sy][sz] - s_Ez[sx-1][sy][sz]) * 0.5f * inv_dx;
    float dEx_dz = (s_Ex[sx][sy][sz+1] - s_Ex[sx][sy][sz-1]) * 0.5f * inv_dx;
    *By += dt * (dEz_dx - dEx_dz);

    float dEx_dy = (s_Ex[sx][sy+1][sz] - s_Ex[sx][sy-1][sz]) * 0.5f * inv_dx;
    float dEy_dx = (s_Ey[sx+1][sy][sz] - s_Ey[sx-1][sy][sz]) * 0.5f * inv_dx;
    *Bz += dt * (dEx_dy - dEy_dx);
}

// === MAIN KERNEL ===
extern "C" __launch_bounds__(256, 4)
__global__ void ct_entropy_solver_14momentSOTA(
    float* rho, float* mx, float* my, float* mz,
    float* Pxx, float* Pxy, float* Pxz, float* Pyy, float* Pyz, float* Pzz,
    float* qx, float* qy, float* qz, float* r_kurtosis,
    float* Bx, float* By, float* Bz, float* psi,
    float* Ex, float* Ey, float* Ez,
    int Ni, float dt, float dx, float damping) {

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int base_i = blockIdx.x * TILE_X; int base_j = blockIdx.y * TILE_Y; int base_k = blockIdx.z * TILE_Z;
    int i = base_i + tx; int j = base_j + ty; int k = base_k + tz;

    bool active = (i >= 2 && j >= 2 && k >= 2 && i < Ni-3 && j < Ni-3 && k < Ni-3);

    __shared__ float s_rho[2][TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_mx[2][TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    int buf = 0;

    if (active) {
        int ii = periodic_idx(i, Ni);
        int jj = periodic_idx(j, Ni);
        int kk = periodic_idx(k, Ni);
        int gidx = ii * Ni * Ni + jj * Ni + kk;

        __cp_async_ca(&s_rho[buf][tx + PAD][ty + PAD][tz + PAD], &rho[gidx]);
        __cp_async_ca(&s_mx[buf][tx + PAD][ty + PAD][tz + PAD], &mx[gidx]);
    }
    __cp_async_commit_group();

    __cp_async_wait_group(0);
    __syncthreads();

    if (!active) return;

    int idx = i * Ni * Ni + j * Ni + k;

    // === Embedding + Spectral FNO (Hybrid ML) ===
    float moment[ORIGINAL_DIM];
    moment[0] = rho[idx];
    moment[1] = mx[idx]; moment[2] = my[idx]; moment[3] = mz[idx];
    moment[4] = Pxx[idx]; moment[5] = Pxy[idx]; moment[6] = Pxz[idx];
    moment[7] = Pyy[idx]; moment[8] = Pyz[idx]; moment[9] = Pzz[idx];
    moment[10] = qx[idx]; moment[11] = qy[idx]; moment[12] = qz[idx];
    moment[13] = r_kurtosis[idx];

    float embedded[MAX_K];
    project_to_embedding(moment, embedded);

    float fno_correction = spectral_fno_correction(embedded);

    // === Original + Refined Adaptive Blending ===
    float P[6] = {Pxx[idx], Pxy[idx], Pxz[idx], Pyy[idx], Pyz[idx], Pzz[idx]};
    float q[3] = {qx[idx], qy[idx], qz[idx]};
    float r = r_kurtosis[idx];

    float entropy_val = convex_entropy_14moment(rho[idx], P, q, r);
    float entropy_force = fminf(fmaxf(damping * entropy_val, -MAX_FLUX), MAX_FLUX);

    float analytic_correction = quadratic_heat_flux_correction(P, q);

    float trace_P = P[0] + P[3] + P[5];
    float q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    float sigma = q2 / (fmaxf(rho[idx] * trace_P * trace_P * trace_P, EPS));
    sigma = fminf(fmaxf(sigma, 0.0f), 1.0f);

    float diff = fabsf(fno_correction - analytic_correction);
    float fno_conf = fminf(diff / (fabsf(analytic_correction) + EPS + 1e-6f), 1.0f);
    sigma = fminf(1.0f, sigma + 0.25f * fno_conf);

    float relax = fmaxf(0.90f, 1.0f - 0.5f * sigma - 0.25f * fabsf(entropy_force));

    float fno_weight = relax * (1.0f - 0.2f * fabsf(entropy_force));
    fno_weight = fmaxf(0.0f, fminf(fno_weight, 0.8f));

    float blended = (1.0f - fno_weight) * analytic_correction + fno_weight * fno_correction;

    q[0] += blended * 0.05f;
    q[1] += blended * 0.05f;
    q[2] += blended * 0.05f;

    entropy_variable_projection(&rho[idx], P, q, &r, entropy_force);

    Pxx[idx] = P[0]; Pxy[idx] = P[1]; Pxz[idx] = P[2];
    Pyy[idx] = P[3]; Pyz[idx] = P[4]; Pzz[idx] = P[5];
    qx[idx] = q[0]; qy[idx] = q[1]; qz[idx] = q[2];
    r_kurtosis[idx] = r;

    entropy_modulated_glm(&Bx[idx], &By[idx], &Bz[idx], &psi[idx], dt, dx, entropy_force, i, j, k, Ni);
    ct_curl_emf(&Bx[idx], &By[idx], &Bz[idx], &Ex[idx], &Ey[idx], &Ez[idx], dt, dx, i, j, k, Ni);
}
''', 'ct_entropy_solver_14momentSOTA')
