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

__device__ inline int periodic_idx(int i, int N) {
    return (i + N) % N;
}

// === Core entropy helpers ===
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

// === Entropy-Variable Projection ===
__device__ void entropy_variable_projection(float* rho, float* P, float* q, float* r, float entropy_force) {
    *rho = fmaxf(*rho, EPS);

    // Entropy space projection (minimal entropy change)
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

// === Refined Log-Cholesky ===
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

// === Main kernel (enforcement focused) ===
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

    float P[6] = {Pxx[idx], Pxy[idx], Pxz[idx], Pyy[idx], Pyz[idx], Pzz[idx]};
    float q[3] = {qx[idx], qy[idx], qz[idx]};
    float r = r_kurtosis[idx];

    float entropy_val = convex_entropy_14moment(rho[idx], P, q, r);
    float entropy_force = fminf(fmaxf(damping * entropy_val, -MAX_FLUX), MAX_FLUX);

    entropy_variable_projection(&rho[idx], P, q, &r, entropy_force);

    Pxx[idx] = P[0]; Pxy[idx] = P[1]; Pxz[idx] = P[2];
    Pyy[idx] = P[3]; Pyz[idx] = P[4]; Pzz[idx] = P[5];
    qx[idx] = q[0]; qy[idx] = q[1]; qz[idx] = q[2];
    r_kurtosis[idx] = r;

    entropy_modulated_glm(&Bx[idx], &By[idx], &Bz[idx], &psi[idx], dt, dx, entropy_force);
    ct_curl_emf(&Bx[idx], &By[idx], &Bz[idx], &Ex[idx], &Ey[idx], &Ez[idx], dt, dx);
}
''', 'ct_entropy_solver_14momentSOTA')
