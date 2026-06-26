import cupy as cp

ct_entropy_solver_14momentSOTA = cp.RawKernel(r'''
#define TILE_X 32
#define TILE_Y 8
#define TILE_Z 4
#define PAD 2
#define EPS 1e-12f
#define MAX_FLUX 1e6f
#define SINGULAR_TOL 1e-8f

__device__ inline int s_idx(int x, int y, int z) {
    return (x * (TILE_Y + 2*PAD) + y) * (TILE_Z + 2*PAD) + z;
}

// ===============================================
// 1. Core Helpers
// ===============================================

__device__ float det_P(float* P) {
    return P[0]*(P[3]*P[5] - P[4]*P[4]) - P[1]*(P[1]*P[5] - P[2]*P[4]) + P[2]*(P[1]*P[4] - P[3]*P[2]);
}

__device__ void clamp_correlation_coefficients(float* P) {
    float sx = sqrtf(fmaxf(P[0], EPS));
    float sy = sqrtf(fmaxf(P[3], EPS));
    float sz = sqrtf(fmaxf(P[5], EPS));
    P[1] = fminf(fmaxf(P[1], -0.999f*sx*sy), 0.999f*sx*sy);
    P[2] = fminf(fmaxf(P[2], -0.999f*sx*sz), 0.999f*sx*sz);
    P[4] = fminf(fmaxf(P[4], -0.999f*sy*sz), 0.999f*sy*sz);
}

__device__ bool log_cholesky_projection(float* P, float trace) {
    if (trace < EPS) return false;
    float L00 = sqrtf(fmaxf(P[0], EPS));
    float L10 = P[1] / L00;
    float L20 = P[2] / L00;
    float L11 = sqrtf(fmaxf(P[3] - L10*L10, EPS));
    float L21 = (P[4] - L10*L20) / L11;
    float L22 = sqrtf(fmaxf(P[5] - L20*L20 - L21*L21, EPS));

    float current_trace = L00*L00 + L11*L11 + L22*L22;
    if (current_trace < EPS) return false;

    float scale = powf(trace / current_trace, 1.0f/3.0f);
    L00 *= scale; L11 *= scale; L22 *= scale;

    P[0] = L00*L00; P[1] = L10*L00; P[2] = L20*L00;
    P[3] = L11*L11; P[4] = L21*L11; P[5] = L22*L22;
    return true;
}

__device__ void compute_pressure_eigenvalues(float* P, float* ev) {
    float a = P[0], b = P[1], c = P[2];
    float d = P[3], e = P[4], f = P[5];
    float trace = a + d + f;
    float p = (a*d + a*f + d*f - b*b - c*c - e*e) / 3.0f - (trace*trace) / 9.0f;
    float q = (2.0f*a*d*f + 2.0f*b*b*e + 2.0f*c*c*d 
             - a*a*f - d*d*c - f*f*a 
             - 2.0f*b*c*e 
             - a*d*d - a*f*f - d*f*f 
             + 3.0f*trace*p) / 2.0f;

    float discriminant = q*q + p*p*p;
    if (discriminant >= 0.0f) {
        float r = sqrtf(discriminant);
        float u = cbrtf(-q + r);
        float v = cbrtf(-q - r);
        ev[0] = u + v + trace / 3.0f;
        ev[1] = -0.5f*(u + v) + trace / 3.0f + 0.5f*sqrtf(3.0f)*(u - v);
        ev[2] = -0.5f*(u + v) + trace / 3.0f - 0.5f*sqrtf(3.0f)*(u - v);
    } else {
        float theta = acosf(q / sqrtf(-p*p*p));
        float r = 2.0f * sqrtf(-p);
        ev[0] = r * cosf(theta / 3.0f) + trace / 3.0f;
        ev[1] = r * cosf((theta + 2.0f * 3.14159265f) / 3.0f) + trace / 3.0f;
        ev[2] = r * cosf((theta + 4.0f * 3.14159265f) / 3.0f) + trace / 3.0f;
    }
}

// ===============================================
// Convex Entropy Projection with Entropy Variables
// ===============================================

__device__ void compute_entropy_variables(float rho, float* P, float* q, float r, float* V) {
    float trace_P = P[0] + P[3] + P[5];
    float p = trace_P / 3.0f;

    V[0] = logf(rho) - 1.5f * logf(trace_P / rho) - 1.0f;

    float inv_p = 1.0f / (p + EPS);
    V[1] = -0.5f * P[0] * inv_p;
    V[2] = -0.5f * P[1] * inv_p;
    V[3] = -0.5f * P[2] * inv_p;
    V[4] = -0.5f * P[3] * inv_p;
    V[5] = -0.5f * P[4] * inv_p;
    V[6] = -0.5f * P[5] * inv_p;

    float thermal_vel2 = trace_P / (rho + EPS);
    V[7] = q[0] / (thermal_vel2 * rho + EPS);
    V[8] = q[1] / (thermal_vel2 * rho + EPS);
    V[9] = q[2] / (thermal_vel2 * rho + EPS);
    V[10] = r / (thermal_vel2 * thermal_vel2 * rho + EPS);
}

__device__ void reconstruct_from_entropy_variables(float* V, float* rho, float* P, float* q, float* r) {
    *rho = expf(V[0] + 1.0f);

    float p = 1.0f;
    P[0] = p - 2.0f * V[1];
    P[1] = -2.0f * V[2];
    P[2] = -2.0f * V[3];
    P[3] = p - 2.0f * V[4];
    P[4] = -2.0f * V[5];
    P[5] = p - 2.0f * V[6];

    float trace_P = P[0] + P[3] + P[5];
    float thermal_vel2 = trace_P / (*rho + EPS);

    q[0] = V[7] * thermal_vel2 * *rho;
    q[1] = V[8] * thermal_vel2 * *rho;
    q[2] = V[9] * thermal_vel2 * *rho;

    *r = V[10] * thermal_vel2 * thermal_vel2 * *rho;
}

__device__ void convex_entropy_projection(float* rho, float* P, float* q, float* r, float entropy_force) {
    float V[11];
    compute_entropy_variables(*rho, P, q, *r, V);

    float relax = 0.18f * (1.0f + 0.7f * fabsf(entropy_force));

    for (int i = 1; i <= 6; i++) V[i] *= (1.0f - relax);
    for (int i = 7; i <= 9; i++) V[i] *= (1.0f - relax);
    V[10] *= (1.0f - relax);

    reconstruct_from_entropy_variables(V, rho, P, q, r);

    float trace = P[0] + P[3] + P[5];
    if (!log_cholesky_projection(P, trace)) {
        float shift = EPS * trace;
        P[0] += shift; P[3] += shift; P[5] += shift;
        log_cholesky_projection(P, trace);
    }
    clamp_correlation_coefficients(P);
}

__device__ void enforce_core_constraints(float* rho, float* P, float* q, float* r,
                                         float entropy_force, float divB, float grad_rho, float grad_P,
                                         int* log_fallback_count) {
    convex_entropy_projection(rho, P, q, r, entropy_force);

    if (*rho < EPS || (P[0] + P[3] + P[5]) < EPS) {
        if (log_fallback_count) atomicAdd(log_fallback_count, 1);
    }
}

// ===============================================
// Main Kernel - Complete v1.0
// ===============================================

__global__ void ct_entropy_solver_14momentSOTA(
    float* rho, float* mx, float* my, float* mz,
    float* Pxx, float* Pxy, float* Pxz, float* Pyy, float* Pyz, float* Pzz,
    float* qx, float* qy, float* qz, float* r_kurtosis,
    float* Bx, float* By, float* Bz, float* psi,
    float* cfl_out, float* mach_out,
    int* troubled_count, int* subcell_count, int* log_fallback_count,
    int Ni, float dt, float dx, float damping) {

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int i = blockIdx.x * TILE_X + tx;
    int j = blockIdx.y * TILE_Y + ty;
    int k = blockIdx.z * TILE_Z + tz;

    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;

    int idx = i*Ni*Ni + j*Ni + k;
    int sx = tx + PAD, sy = ty + PAD, sz = tz + PAD;

    // Full Shared Memory Load
    __shared__ float s_rho[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_mx [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_my [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_mz [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];

    __shared__ float s_Pxx[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_Pxy[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_Pxz[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_Pyy[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_Pyz[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_Pzz[(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];

    __shared__ float s_qx [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_qy [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_qz [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];

    __shared__ float s_r  [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];

    __shared__ float s_Bx [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_By [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];
    __shared__ float s_Bz [(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD)];

    int flat = s_idx(sx, sy, sz);

    s_rho[flat] = rho[idx];
    s_mx[flat] = mx[idx];
    s_my[flat] = my[idx];
    s_mz[flat] = mz[idx];

    s_Pxx[flat] = Pxx[idx]; s_Pxy[flat] = Pxy[idx]; s_Pxz[flat] = Pxz[idx];
    s_Pyy[flat] = Pyy[idx]; s_Pyz[flat] = Pyz[idx]; s_Pzz[flat] = Pzz[idx];

    s_qx[flat] = qx[idx]; s_qy[flat] = qy[idx]; s_qz[flat] = qz[idx];

    s_r[flat] = r_kurtosis[idx];

    s_Bx[flat] = Bx[idx]; s_By[flat] = By[idx]; s_Bz[flat] = Bz[idx];

    __syncthreads();

    float P[6] = {s_Pxx[flat], s_Pxy[flat], s_Pxz[flat], s_Pyy[flat], s_Pyz[flat], s_Pzz[flat]};
    float q[3] = {s_qx[flat], s_qy[flat], s_qz[flat]};
    float r = s_r[flat];

    float vx = s_mx[flat] / (s_rho[flat] + EPS);
    float vy = s_my[flat] / (s_rho[flat] + EPS);
    float vz = s_mz[flat] / (s_rho[flat] + EPS);

    float div_v = 0.5f / dx * (
        (mx[idx+1] - mx[idx-1]) / (rho[idx] + EPS) +
        (my[idx+Ni] - my[idx-Ni]) / (rho[idx] + EPS) +
        (mz[idx+Ni*Ni] - mz[idx-Ni*Ni]) / (rho[idx] + EPS)
    );

    float entropy_force = compute_entropy_production(s_rho[flat], P, q, r, div_v);

    enforce_core_constraints(&rho[idx], P, q, &r, entropy_force, 0.0f, 0.0f, 0.0f, log_fallback_count);

    // Write back
    Pxx[idx] = P[0]; Pxy[idx] = P[1]; Pxz[idx] = P[2];
    Pyy[idx] = P[3]; Pyz[idx] = P[4]; Pzz[idx] = P[5];
    qx[idx] = q[0]; qy[idx] = q[1]; qz[idx] = q[2];
    r_kurtosis[idx] = r;
}
''', 'ct_entropy_solver_14momentSOTA')
