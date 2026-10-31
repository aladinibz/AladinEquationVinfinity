import cupy as cp
import numpy as np

gyrotropic_12_kernel = cp.RawKernel(r'''
extern "C" __global__
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
    float* __restrict__ residual_out,
    int Ncells,
    float mass,
    float alpha_par,
    float alpha_perp,
    float thermal_scale);

#define N_FEAT 12
#define N_FREE 9
#define MAX_NEWTON 8
#define RES_TOL 2.0e-3f

__device__ __forceinline__ void eval_phi(float cp, float c2, float p[N_FEAT]) {
    float cp2 = cp*cp;
    float cp3 = cp2*cp;
    float cp4 = cp2*cp2;
    float c4 = c2*c2;
    float c6 = c4*c2;
    float idp = 1.f/(1.f+cp2);
    float ide = 1.f/(1.f+c2);
    p[0]=1.f; p[1]=cp; p[2]=c2; p[3]=cp2;
    p[4]=cp*idp; p[5]=cp*c2*ide;
    p[6]=c4; p[7]=cp2*c2; p[8]=cp4;
    p[9]=cp3*idp*idp; p[10]=c6*ide; p[11]=cp2*c4*ide;
}

__constant__ float np[16]={-6.5f,-5.5f,-4.5f,-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,-7.5f};
__constant__ float wp[16]={0.001f,0.004f,0.012f,0.03f,0.07f,0.13f,0.19f,0.19f,0.13f,0.07f,0.03f,0.012f,0.004f,0.001f,0.0003f,0.0003f};
__constant__ float nq[12]={0.2f,0.55f,1.0f,1.5f,2.1f,2.8f,3.6f,4.5f,5.6f,6.8f,8.2f,10.0f};
__constant__ float wq[12]={0.06f,0.12f,0.16f,0.17f,0.15f,0.12f,0.08f,0.05f,0.03f,0.015f,0.008f,0.003f};

__device__ void set_from_free(float eta, float zeta, float xi, const float low[6], float lambda[N_FEAT]) {
    float a = expf(eta);
    float b = expf(zeta);
    float s = 0.95f * tanhf(xi);
    lambda[0] = low[0];
    lambda[1] = low[1];
    lambda[2] = low[2];
    lambda[3] = low[3];
    lambda[4] = low[4];
    lambda[5] = low[5];
    lambda[6] = -a*a;
    lambda[7] = -2.f*a*b*s;
    lambda[8] = -b*b;
    lambda[9] = 0.f;
    lambda[10]= 0.f;
    lambda[11]= 0.f;
}

__device__ void jacobian(float eta, float zeta, float xi, float J[N_FEAT*N_FREE]) {
    float a = expf(eta);
    float b = expf(zeta);
    float s = 0.95f * tanhf(xi);
    float ds = 0.95f * (1.f - tanhf(xi)*tanhf(xi));
    for (int i = 0; i < N_FEAT*N_FREE; ++i) J[i] = 0.f;
    for (int k = 0; k < 6; ++k) J[k*N_FREE + k] = 1.f;
    J[6*N_FREE + 6] = -2.f*a;
    J[7*N_FREE + 6] = -2.f*b*s;
    J[7*N_FREE + 7] = -2.f*a*s;
    J[8*N_FREE + 7] = -2.f*b;
    J[7*N_FREE + 8] = -2.f*a*b*ds;
}

__device__ bool moments(const float lambda[N_FEAT], float alpha_par, float alpha_perp,
                        float M[N_FEAT], float G[N_FEAT*N_FEAT]) {
    float Z = 0.f;
    float sum_phi[N_FEAT];
    float sum_phi_phi[N_FEAT*N_FEAT];
    for (int k = 0; k < N_FEAT; ++k) sum_phi[k] = 0.f;
    for (int k = 0; k < N_FEAT*N_FEAT; ++k) sum_phi_phi[k] = 0.f;

    float m = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = np[ip];
            float c2 = nq[iq]*nq[iq];
            float expo = 0.f;
            float phi[N_FEAT];
            eval_phi(cp, c2, phi);
            for (int k = 0; k < N_FEAT; ++k) expo += lambda[k]*phi[k];
            expo -= alpha_par*cp*cp - alpha_perp*c2;
            if (expo > m) m = expo;
        }
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cp = np[ip];
            float c2 = nq[iq]*nq[iq];
            float w = wp[ip]*wq[iq];
            float phi[N_FEAT];
            eval_phi(cp, c2, phi);
            float expo = 0.f;
            for (int k = 0; k < N_FEAT; ++k) expo += lambda[k]*phi[k];
            expo -= alpha_par*cp*cp - alpha_perp*c2;
            float e = expf(expo - m);
            Z += w * e;
            for (int k = 0; k < N_FEAT; ++k) sum_phi[k] += w * e * phi[k];
            for (int i = 0; i < N_FEAT; ++i)
                for (int j = 0; j < N_FEAT; ++j)
                    sum_phi_phi[i*N_FEAT+j] += w * e * phi[i]*phi[j];
        }
    if (Z < 1e-30f) return false;
    float invZ = 1.f / Z;
    for (int k = 0; k < N_FEAT; ++k) M[k] = sum_phi[k] * invZ;
    for (int i = 0; i < N_FEAT; ++i)
        for (int j = 0; j < N_FEAT; ++j)
            G[i*N_FEAT+j] = sum_phi_phi[i*N_FEAT+j]*invZ - M[i]*M[j];
    return true;
}

extern "C" __global__
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
    float* __restrict__ residual_out,
    int Ncells,
    float mass,
    float alpha_par,
    float alpha_perp,
    float thermal_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncells) return;

    const float* Mt = M_target + idx*N_FEAT;
    float* lo = lambda_io + idx*N_FEAT;

    float eta = 0.f, zeta = 0.f, xi = 0.f;
    float low[6];
    for (int k = 0; k < 6; ++k) low[k] = 0.f;
    float lambda[N_FEAT];
    float M[N_FEAT];
    int flag = 1;
    float final_res = 1e9f;

    set_from_free(eta, zeta, xi, low, lambda);

    for (int it = 0; it < MAX_NEWTON; ++it) {
        float G[N_FEAT*N_FEAT];
        if (!moments(lambda, alpha_par, alpha_perp, M, G)) { flag = 6; break; }

        float res = 0.f;
        for (int k = 1; k < N_FEAT; ++k) res = fmaxf(res, fabsf(M[k]-Mt[k]));
        final_res = res;

        if (res < RES_TOL) { flag = 0; break; }

        float J[N_FEAT*N_FREE];
        jacobian(eta, zeta, xi, J);

        float D[N_FEAT*N_FREE];
        for (int i = 0; i < N_FEAT; ++i)
            for (int p = 0; p < N_FREE; ++p) {
                float s = 0.f;
                for (int j = 0; j < N_FEAT; ++j) s += G[i*N_FEAT+j] * J[j*N_FREE+p];
                D[i*N_FREE+p] = s;
            }

        float H[N_FREE*N_FREE];
        float rhs[N_FREE];
        for (int p = 0; p < N_FREE; ++p) {
            rhs[p] = 0.f;
            for (int q = 0; q < N_FREE; ++q) H[p*N_FREE+q] = 0.f;
        }
        float ridge = 1e-4f;
        for (int p = 0; p < N_FREE; ++p) {
            for (int q = 0; q < N_FREE; ++q) {
                float s = 0.f;
                for (int i = 1; i < N_FEAT; ++i) s += D[i*N_FREE+p]*D[i*N_FREE+q];
                H[p*N_FREE+q] = s;
            }
            H[p*N_FREE+p] += ridge;
            float s = 0.f;
            for (int i = 1; i < N_FEAT; ++i) s += D[i*N_FREE+p]*(M[i]-Mt[i]);
            rhs[p] = s;
        }

        float L[N_FREE*N_FREE];
        for (int k = 0; k < N_FREE*N_FREE; ++k) L[k] = 0.f;
        bool ok = true;
        for (int i = 0; i < N_FREE; ++i) {
            for (int j = 0; j <= i; ++j) {
                float s = H[i*N_FREE+j];
                for (int k = 0; k < j; ++k) s -= L[i*N_FREE+k]*L[j*N_FREE+k];
                if (i == j) {
                    if (s <= 1e-12f) { ok = false; break; }
                    L[i*N_FREE+j] = sqrtf(s);
                } else {
                    L[i*N_FREE+j] = s / L[j*N_FREE+j];
                }
            }
            if (!ok) break;
        }
        if (!ok) { flag = 4; break; }

        float y[N_FREE];
        float dx[N_FREE];
        for (int i = 0; i < N_FREE; ++i) {
            float s = rhs[i];
            for (int k = 0; k < i; ++k) s -= L[i*N_FREE+k]*y[k];
            y[i] = s / L[i*N_FREE+i];
        }
        for (int i = N_FREE-1; i >= 0; --i) {
            float s = y[i];
            for (int k = i+1; k < N_FREE; ++k) s -= L[k*N_FREE+i]*dx[k];
            dx[i] = s / L[i*N_FREE+i];
        }

        float alpha = 1.f;
        bool accepted = false;
        for (int ls = 0; ls < 4; ++ls) {
            float eta2 = eta - alpha*dx[6];
            float zeta2 = zeta - alpha*dx[7];
            float xi2 = xi - alpha*dx[8];
            float low2[6];
            for (int k = 0; k < 6; ++k) low2[k] = low[k] - alpha*dx[k];
            float lam2[N_FEAT];
            set_from_free(eta2, zeta2, xi2, low2, lam2);
            float M2[N_FEAT];
            float Gdummy[N_FEAT*N_FEAT];
            if (!moments(lam2, alpha_par, alpha_perp, M2, Gdummy)) { alpha *= 0.5f; continue; }
            float res2 = 0.f;
            for (int k = 1; k < N_FEAT; ++k) res2 = fmaxf(res2, fabsf(M2[k]-Mt[k]));
            if (res2 < res) {
                eta = eta2; zeta = zeta2; xi = xi2;
                for (int k = 0; k < 6; ++k) low[k] = low2[k];
                set_from_free(eta, zeta, xi, low, lambda);
                accepted = true;
                break;
            }
            alpha *= 0.5f;
        }
        if (!accepted) { flag = 3; break; }
    }

    float Gf[N_FEAT*N_FEAT];
    bool ok_final = moments(lambda, alpha_par, alpha_perp, M, Gf);
    if (ok_final) {
        final_res = 0.f;
        for (int k = 1; k < N_FEAT; ++k) final_res = fmaxf(final_res, fabsf(M[k]-Mt[k]));
        if (final_res < RES_TOL) flag = 0;
        else if (final_res < 0.5f) flag = 1;
        else flag = 2;
    } else flag = 6;

    for (int k = 0; k < N_FEAT; ++k) lo[k] = lambda[k];
    status[idx] = flag;
    residual_out[idx] = final_res;

    float n = n_in[idx];
    float up = ok_final ? M[1] : 0.f;
    float cp2 = ok_final ? fmaxf(M[3]-up*up, 1e-8f) : 1e-4f;
    float ce2 = ok_final ? fmaxf(M[2], 1e-8f) : 1e-4f;
    n_out[idx] = n;
    upar_out[idx] = up;
    Ppar_out[idx] = mass * n * cp2;
    Pperp_out[idx] = 0.5f * mass * n * ce2;
    qpar_out[idx] = ok_final ? mass * n * M[4] * thermal_scale : 0.f;
    qperp_out[idx] = ok_final ? mass * n * M[5] * thermal_scale : 0.f;
}
''', 'gyrotropic_12_kernel')


# ---------- minimal driver ----------
if __name__ == "__main__":
    Ncells = 64
    mass = 1.0

    M_target = cp.zeros((Ncells, 12), dtype=cp.float32)
    M_target[:, 0] = 1.0
    M_target[:, 2] = 1.0
    M_target[:, 3] = 1.0

    n_in         = cp.ones(Ncells, dtype=cp.float32)
    lambda_io    = cp.zeros((Ncells, 12), dtype=cp.float32)
    n_out        = cp.empty(Ncells, dtype=cp.float32)
    upar         = cp.empty(Ncells, dtype=cp.float32)
    Ppar         = cp.empty(Ncells, dtype=cp.float32)
    Pperp        = cp.empty(Ncells, dtype=cp.float32)
    qpar         = cp.empty(Ncells, dtype=cp.float32)
    qperp        = cp.empty(Ncells, dtype=cp.float32)
    status       = cp.empty(Ncells, dtype=cp.int32)
    residual_out = cp.empty(Ncells, dtype=cp.float32)

    threads = 128
    blocks  = (Ncells + 127) // 128

    gyrotropic_12_kernel(
        (blocks,), (threads,),
        (M_target, n_in, lambda_io,
         n_out, upar, Ppar, Pperp, qpar, qperp,
         status, residual_out, Ncells,
         np.float32(mass),
         np.float32(0.5),
         np.float32(0.5),
         np.float32(1.0))
    )

    print("status unique:", cp.unique(status))
    print("final residual (cell 0):", float(residual_out[0]))
    print("Ppar  sample:", Ppar[:4])
    print("Pperp sample:", Pperp[:4])
