import cupy as cp

ct_entropy_solver_14momentSOTA = cp.RawKernel(r'''
/*
 * ALADIN ∞ ℂ(t) - Refactored Kernel with float4 Vector Loads
 * =============================================================
 * Improved memory coalescing and bandwidth
 */

#define TILE_X 32
#define TILE_Y 8
#define TILE_Z 4
#define PAD 1
#define EPS 1e-12f
#define MAX_FLUX 1e6f
#define GLM_BASE_CH 1.0f
#define GLM_DAMPING 0.1f
#define SINGULAR_TOL 1e-8f

#define ORIGINAL_DIM 15
#define MAX_K 16
#define FFT_SIZE 16

__constant__ float d_embedding_basis[ORIGINAL_DIM * MAX_K];
__constant__ __half d_lift_weights[256];
__constant__ float d_spectral_low[8];
__constant__ float d_spectral_high[8];

// === Core Helpers (unchanged) ===
__device__ inline int periodic_idx(int i, int N) { return (i + N) % N; }

// ... keep det_P, quadratic_heat_flux_correction, convex_entropy_14moment, refined_log_cholesky, entropy_variable_projection ...

// === Tensor Core Lift (unchanged) ===
__device__ float tensor_core_lift(float* __restrict__ embedded) {
    // ... same optimized Tensor Core lift as before ...
}

// === Hybrid Spectral Correction ===
__device__ float hybrid_spectral_fno_correction(float* __restrict__ embedded) {
    // ... same as before ...
}

// === Vectorized GLM (example with float4 where possible) ===
__device__ void entropy_modulated_glm(float* __restrict__ B, float* __restrict__ psi,
                                      float dt, float dx, float entropy_force,
                                      int i, int j, int k, int Ni) {
    // B is now float4 array for Bx, By, Bz (padded)
    __shared__ float s_B[3][TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];  // or use float4
    // ... shared memory loading and GLM logic (adapted for vector access) ...
}

// === Main Kernel with float4 ===
extern "C" __launch_bounds__(256, 8)
__global__ void ct_entropy_solver_14momentSOTA(
    float* __restrict__ rho,
    float4* __restrict__ momentum,   // mx, my, mz
    float4* __restrict__ B,          // Bx, By, Bz
    float4* __restrict__ E,          // Ex, Ey, Ez
    float4* __restrict__ q,          // qx, qy, qz
    float* __restrict__ P,           // Pxx ... Pzz (6 components)
    float* __restrict__ r_kurtosis,
    float* __restrict__ psi,
    int Ni, float dt, float dx, float damping) {

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int i = blockIdx.x * TILE_X + tx;
    int j = blockIdx.y * TILE_Y + ty;
    int k = blockIdx.z * TILE_Z + tz;

    bool active = (i >= 2 && j >= 2 && k >= 2 && i < Ni-3 && j < Ni-3 && k < Ni-3);

    __shared__ float s_rho[2][TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];

    int buf = 0;

    if (active) {
        int gidx = periodic_idx(i, Ni) * Ni * Ni + periodic_idx(j, Ni) * Ni + periodic_idx(k, Ni);
        __cp_async_ca(&s_rho[buf][tx + PAD][ty + PAD][tz + PAD], &rho[gidx]);
    }
    __cp_async_commit_group();
    __cp_async_wait_group(0);
    __syncthreads();

    if (!active) return;

    int idx = i * Ni * Ni + j * Ni + k;

    // === Vectorized Loads ===
    float mom[3], Bvec[3], Evec[3], qvec[3];
    float4 m4 = momentum[idx];
    mom[0] = m4.x; mom[1] = m4.y; mom[2] = m4.z;

    float4 B4 = B[idx];
    Bvec[0] = B4.x; Bvec[1] = B4.y; Bvec[2] = B4.z;

    float4 E4 = E[idx];
    Evec[0] = E4.x; Evec[1] = E4.y; Evec[2] = E4.z;

    float4 q4 = q[idx];
    qvec[0] = q4.x; qvec[1] = q4.y; qvec[2] = q4.z;

    // Pressure tensor (scalar for now)
    float P_arr[6] = {P[idx*6], P[idx*6+1], P[idx*6+2], P[idx*6+3], P[idx*6+4], P[idx*6+5]};

    float r = r_kurtosis[idx];

    // === Hybrid ML Path ===
    float moment[ORIGINAL_DIM];
    moment[0] = rho[idx];
    moment[1] = mom[0]; moment[2] = mom[1]; moment[3] = mom[2];
    moment[4] = P_arr[0]; moment[5] = P_arr[1]; moment[6] = P_arr[2];
    moment[7] = P_arr[3]; moment[8] = P_arr[4]; moment[9] = P_arr[5];
    moment[10] = qvec[0]; moment[11] = qvec[1]; moment[12] = qvec[2];
    moment[13] = r;

    float embedded[MAX_K];
    #pragma unroll
    for (int d = 0; d < MAX_K; d++) {
        embedded[d] = 0.0f;
        #pragma unroll
        for (int m = 0; m < ORIGINAL_DIM; m++) {
            embedded[d] += d_embedding_basis[m * MAX_K + d] * moment[m];
        }
    }

    float fno_correction = hybrid_spectral_fno_correction(embedded);

    // === Analytic + Blending (using vector data) ===
    float P[6] = {P_arr[0], P_arr[1], P_arr[2], P_arr[3], P_arr[4], P_arr[5]};
    float q[3] = {qvec[0], qvec[1], qvec[2]};

    float entropy_val = convex_entropy_14moment(rho[idx], P, q, r);
    float entropy_force = fminf(fmaxf(damping * entropy_val, -MAX_FLUX), MAX_FLUX);

    float analytic_correction = quadratic_heat_flux_correction(P, q);

    float trace_P = P[0] + P[3] + P[5];
    float q2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    float sigma = q2 / (fmaxf(rho[idx] * trace_P * trace_P * trace_P, EPS));
    sigma = fminf(fmaxf(sigma, 0.0f), 1.0f);

    float fno_conf = fabsf(fno_correction - analytic_correction) / (fabsf(analytic_correction) + EPS + 1e-6f);
    sigma = fminf(1.0f, sigma + 0.25f * fno_conf);
    float relax = fmaxf(0.90f, 1.0f - 0.5f * sigma - 0.25f * fabsf(entropy_force));
    float fno_weight = fmaxf(0.0f, fminf(relax * (1.0f - 0.2f * fabsf(entropy_force)), 0.8f));

    float blended = (1.0f - fno_weight) * analytic_correction + fno_weight * fno_correction;
    q[0] += blended * 0.05f; q[1] += blended * 0.05f; q[2] += blended * 0.05f;

    entropy_variable_projection(&rho[idx], P, q, &r, entropy_force);

    // Write back
    Pxx[idx] = P[0]; Pxy[idx] = P[1]; Pxz[idx] = P[2];
    Pyy[idx] = P[3]; Pyz[idx] = P[4]; Pzz[idx] = P[5];
    qx[idx] = q[0]; qy[idx] = q[1]; qz[idx] = q[2];
    r_kurtosis[idx] = r;

    // GLM and Curl (adapt to vector fields if needed)
    entropy_modulated_glm(&Bx[idx], &By[idx], &Bz[idx], &psi[idx], dt, dx, entropy_force, i, j, k, Ni);
    ct_curl_emf(&Bx[idx], &By[idx], &Bz[idx], &Ex[idx], &Ey[idx], &Ez[idx], dt, dx, i, j, k, Ni);
}
''', 'ct_entropy_solver_14momentSOTA')
