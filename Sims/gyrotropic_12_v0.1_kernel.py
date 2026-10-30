import cupy as cp
import numpy as np

# Gyrotropic 12-Feature Exponential Manifold
# Positivity, Tail Confinement and GPU Moment Reconstruction (v0.1)
# Experimental numerical implementation

gyrotropic_12_kernel = cp.RawKernel(r'''
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
