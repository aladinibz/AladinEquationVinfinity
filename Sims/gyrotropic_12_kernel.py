import cupy as cp

gyrotropic_12_kernel = cp.RawKernel(r'''
#define N_FEAT 12
#define N_FREE 9
#define MAX_NEWTON 5
#define RES_TOL 2e-4f

__device__ __forceinline__ void eval_phi(float cp, float c2, float p[N_FEAT]) {
    float cp2 = cp*cp, cp3 = cp2*cp, cp4 = cp2*cp2;
    float c4 = c2*c2, c6 = c4*c2;
    float idp = 1.f/(1.f+cp2), ide = 1.f/(1.f+c2);
    p[0]=1.f; p[1]=cp; p[2]=c2; p[3]=cp2;
    p[4]=cp*idp; p[5]=cp*c2*ide;
    p[6]=c4; p[7]=cp2*c2; p[8]=cp4;
    p[9]=cp3*idp*idp; p[10]=c6*ide; p[11]=cp2*c4*ide;
}

__constant__ float np[16]={-6.5f,-5.5f,-4.5f,-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,-7.5f};
__constant__ float wp[16]={0.001f,0.004f,0.012f,0.03f,0.07f,0.13f,0.19f,0.19f,0.13f,0.07f,0.03f,0.012f,0.004f,0.001f,0.0003f,0.0003f};
__constant__ float nq[12]={0.2f,0.55f,1.0f,1.5f,2.1f,2.8f,3.6f,4.5f,5.6f,6.8f,8.2f,10.0f};
__constant__ float wq[12]={0.06f,0.12f,0.16f,0.17f,0.15f,0.12f,0.08f,0.05f,0.03f,0.015f,0.008f,0.003f};

__device__ bool finite9(const float* v) {
    for (int i = 0; i < 9; ++i) if (!isfinite(v[i])) return false;
    return true;
}

__device__ bool admissible_target(const float* Mt) {
    for (int k = 0; k < N_FEAT; ++k) if (!isfinite(Mt[k])) return false;
    if (Mt[0] < 0.5f || Mt[0] > 1.5f) return false;
    if (Mt[2] < 1e-8f || Mt[3] < 1e-8f) return false;
    if (Mt[6] < Mt[2]*Mt[2] - 1e-5f) return false;
    if (Mt[8] < Mt[3]*Mt[3] - 1e-5f) return false;
    float qmax_p = 2.0f * sqrtf(Mt[3]) * Mt[3];
    float qmax_e = 2.0f * sqrtf(Mt[2]) * Mt[2];
    if (fabsf(Mt[4]) > qmax_p || fabsf(Mt[5]) > qmax_e) return false;
    return true;
}

__device__ void set_from_free(float eta, float zeta, float xi, const float low[6], float* lambda) {
    float a = __expf(eta);
    float b = __expf(zeta);
    float s = 0.95f * tanhf(xi);
    float A = a*a, B = b*b, C = 2.f*a*b*s;
    lambda[0] = 0.f;
    lambda[1] = low[0]; lambda[2] = low[1]; lambda[3] = low[2];
    lambda[4] = low[3]; lambda[5] = low[4]; lambda[9] = low[5];
    lambda[6] = lambda[10] = -0.5f * B;
    lambda[7] = lambda[11] = -0.5f * C;
    lambda[8] = -A;
}

__device__ void jacobian(float eta, float zeta, float xi, float* J) {
    for (int i = 0; i < N_FEAT * N_FREE; ++i) J[i] = 0.f;
    float a = __expf(eta), b = __expf(zeta);
    float th = tanhf(xi), s = 0.95f * th, ds = 0.95f * (1.f - th*th);
    J[8*N_FREE + 0] = -2.f * a * a;
    J[6*N_FREE + 1] = J[10*N_FREE + 1] = -b * b;
    float abs_ = a * b * s;
    J[7*N_FREE + 0] = J[11*N_FREE + 0] = -abs_;
    J[7*N_FREE + 1] = J[11*N_FREE + 1] = -abs_;
    J[7*N_FREE + 2] = J[11*N_FREE + 2] = -a * b * ds;
    J[1*N_FREE + 3] = 1.f; J[2*N_FREE + 4] = 1.f; J[3*N_FREE + 5] = 1.f;
    J[4*N_FREE + 6] = 1.f; J[5*N_FREE + 7] = 1.f; J[9*N_FREE + 8] = 1.f;
}

__device__ float moment_objective(const float* M, const float* Mt) {
    float obj = 0.f;
    for (int k = 1; k < N_FEAT; ++k) {
        float r = M[k] - Mt[k];
        obj += 0.5f * r * r;
    }
    return obj;
}

__device__ bool moments(const float* lambda, float ap, float ape, float* M, float* G) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = np[ip], c2 = nq[iq]*nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            if (!isfinite(e)) return false;
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0}, M2[N_FEAT*N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cp = np[ip], wp_ = wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = nq[iq]*nq[iq], w = wp_*wq[iq]*nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            if (!isfinite(wgt)) return false;
            Z += wgt;
            #pragma unroll
            for (int i = 0; i < N_FEAT; ++i) {
                float wi = wgt * p[i];
                M1[i] += wi;
                #pragma unroll
                for (int j = 0; j <= i; ++j) M2[i*N_FEAT + j] += wi * p[j];
            }
        }
    }
    if (!(Z > 1e-40f) || !isfinite(Z)) return false;
    float iZ = 1.f / Z;
    #pragma unroll
    for (int i = 0; i < N_FEAT; ++i) {
        M[i] = M1[i] * iZ;
        if (!isfinite(M[i])) return false;
        #pragma unroll
        for (int j = 0; j <= i; ++j) {
            float v = M2[i*N_FEAT + j] * iZ - M[i]*M[j];
            G[i*N_FEAT + j] = G[j*N_FEAT + i] = v;
        }
    }
    return true;
}

__device__ bool moments_final(const float* lambda, float ap, float ape,
                              float* M, float* m3_par, float* m3_mix) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = np[ip], c2 = nq[iq]*nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0};
    float acc_par = 0.f, acc_mix = 0.f;
    for (int ip = 0; ip < 16; ++ip) {
        float cp = np[ip], wp_ = wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = nq[iq]*nq[iq], w = wp_*wq[iq]*nq[iq], p[N_FEAT];
            eval_phi(cp, c2, p);
            float e = -ap*cp*cp - ape*c2;
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            acc_par += wgt * (cp*cp*cp);
            acc_mix += wgt * (cp * c2);
            #pragma unroll
            for (int i = 0; i < N_FEAT; ++i) M1[i] += wgt * p[i];
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    #pragma unroll
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * iZ;
    *m3_par = acc_par * iZ;
    *m3_mix = acc_mix * iZ;
    return true;
}

__device__ bool chol9(float* H, float* rhs, float* dx) {
    float L[N_FREE*N_FREE] = {0};
    for (int i = 0; i < N_FREE; ++i)
        for (int j = 0; j <= i; ++j) {
            float s = H[i*N_FREE + j];
            for (int k = 0; k < j; ++k) s -= L[i*N_FREE + k]*L[j*N_FREE + k];
            if (i == j) {
                if (!(s > 1e-14f) || !isfinite(s)) return false;
                L[i*N_FREE + j] = sqrtf(s);
            } else L[i*N_FREE + j] = s / L[j*N_FREE + j];
        }
    float y[N_FREE];
    for (int i = 0; i < N_FREE; ++i) {
        float s = rhs[i];
        for (int k = 0; k < i; ++k) s -= L[i*N_FREE + k]*y[k];
        y[i] = s / L[i*N_FREE + i];
    }
    for (int i = N_FREE - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < N_FREE; ++k) s -= L[k*N_FREE + i]*dx[k];
        dx[i] = s / L[i*N_FREE + i];
    }
    return finite9(dx);
}

extern "C" __global__
__launch_bounds__(64, 4)
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
    int* __restrict__ status,
    int Ncells,
    float mass,
    float thermal_scale,
    float alpha_par,
    float alpha_perp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncells) return;

    const float* Mt = M_target + idx * N_FEAT;
    float* lo = lambda_io + idx * N_FEAT;

    int flag = admissible_target(Mt) ? 0 : 5;

    float eta = -0.7f, zeta = -0.7f, xi = 0.f, low[6] = {0};
    float ap = fmaxf(alpha_par, 0.25f), ae = fmaxf(alpha_perp, 0.25f);
    float lambda_lm = 1e-4f;

    float lambda[N_FEAT], M[N_FEAT], G[N_FEAT * N_FEAT];
    float J[N_FEAT * N_FREE], H[N_FREE * N_FREE], rhs[N_FREE], dx[N_FREE];
    float D[N_FEAT * N_FREE];

    if (flag == 0) {
        for (int it = 0; it < MAX_NEWTON; ++it) {
            set_from_free(eta, zeta, xi, low, lambda);
            if (!moments(lambda, ap, ae, M, G)) { flag = 6; break; }

            float obj = moment_objective(M, Mt);
            float max_r = 0.f;
            for (int k = 1; k < N_FEAT; ++k) max_r = fmaxf(max_r, fabsf(M[k]-Mt[k]));
            if (max_r < RES_TOL) { flag = 0; break; }

            jacobian(eta, zeta, xi, J);

            for (int i = 0; i < N_FEAT; ++i)
                for (int p = 0; p < N_FREE; ++p) {
                    float s = 0.f;
                    for (int j = 0; j < N_FEAT; ++j) s += G[i*N_FEAT + j] * J[j*N_FREE + p];
                    D[i*N_FREE + p] = s;
                }

            for (int p = 0; p < N_FREE*N_FREE; ++p) H[p] = 0.f;
            for (int p = 0; p < N_FREE; ++p) rhs[p] = 0.f;
            for (int i = 1; i < N_FEAT; ++i) {
                float ri = M[i] - Mt[i];
                for (int p = 0; p < N_FREE; ++p) {
                    float Dp = D[i*N_FREE + p];
                    rhs[p] += Dp * ri;
                    for (int q = 0; q < N_FREE; ++q)
                        H[p*N_FREE + q] += Dp * D[i*N_FREE + q];
                }
            }
            float max_diag = 0.f;
            for (int p = 0; p < N_FREE; ++p) max_diag = fmaxf(max_diag, H[p*N_FREE + p]);
            float ridge = lambda_lm * fmaxf(max_diag, 1e-8f);
            for (int p = 0; p < N_FREE; ++p) H[p*N_FREE + p] += ridge;

            if (!chol9(H, rhs, dx)) {
                lambda_lm *= 4.f;
                if (lambda_lm > 1e3f) { flag = 2; break; }
                continue;
            }

            float al = 1.f;
            bool accepted = false;
            for (int ls = 0; ls < 6; ++ls) {
                float e2 = eta - al * dx[0];
                float z2 = zeta - al * dx[1];
                float x2 = xi - al * dx[2];
                if (e2 < -8.f || e2 > 4.f || z2 < -8.f || z2 > 4.f) { al *= 0.5f; continue; }
                float l2[6];
                for (int k = 0; k < 6; ++k) l2[k] = low[k] - al * dx[3 + k];
                float lam2[N_FEAT], M2[N_FEAT], G2[N_FEAT*N_FEAT];
                set_from_free(e2, z2, x2, l2, lam2);
                if (!moments(lam2, ap, ae, M2, G2)) { al *= 0.5f; continue; }
                float obj2 = moment_objective(M2, Mt);
                if (obj2 < obj) {
                    eta = e2; zeta = z2; xi = x2;
                    for (int k = 0; k < 6; ++k) low[k] = l2[k];
                    accepted = true;
                    lambda_lm = fmaxf(lambda_lm * 0.5f, 1e-8f);
                    break;
                }
                al *= 0.5f;
            }
            if (!accepted) {
                lambda_lm *= 4.f;
                if (lambda_lm > 1e3f) { flag = 3; break; }
            }
            if (it == MAX_NEWTON - 1) flag = 1;
        }
    }

    float m3_par = 0.f, m3_mix = 0.f;
    set_from_free(eta, zeta, xi, low, lambda);
    bool ok_final = moments_final(lambda, ap, ae, M, &m3_par, &m3_mix);
    if (!ok_final) flag = 6;

    if (ok_final && flag != 5) {
        float moment_r = 0.f;
        for (int k = 1; k < N_FEAT; ++k)
            moment_r = fmaxf(moment_r, fabsf(M[k] - Mt[k]));
        float up = M[1];
        float theta_par = M[3] - up*up;
        float theta_perp = M[2];
        bool outputs_ok = isfinite(up) && isfinite(theta_par) && isfinite(theta_perp)
                       && theta_par > 0.f && theta_perp > 0.f
                       && isfinite(m3_par) && isfinite(m3_mix);
        if (moment_r < RES_TOL && outputs_ok) flag = 0;
    }

    for (int k = 0; k < N_FEAT; ++k) lo[k] = lambda[k];
    status[idx] = flag;

    float n = n_in[idx];
    float up = ok_final ? M[1] : 0.f;
    float cp2 = ok_final ? fmaxf(M[3] - up*up, 1e-8f) : 1e-4f;
    float ce2 = ok_final ? fmaxf(M[2], 1e-8f) : 1e-4f;

    float qpar = 0.f, qperp = 0.f;
    if (ok_final) {
        float theta_par = fmaxf(cp2, 0.f);
        qpar  = mass * n * (m3_par - 3.f * up * theta_par - up*up*up);
        qperp = 0.5f * mass * n * (m3_mix - up * M[2]);
    }

    n_out[idx] = n;
    upar_out[idx] = up;
    Ppar_out[idx] = mass * n * cp2;
    Pperp_out[idx] = 0.5f * mass * n * ce2;
    qpar_out[idx] = qpar;
    qperp_out[idx] = qperp;
}
''', 'gyrotropic_12_kernel')
