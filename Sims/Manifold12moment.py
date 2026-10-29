import cupy as cp

gyrotropic_12_kernel = cp.RawKernel(r'''
#define N_FEAT 12
#define MAX_NEWTON 5
#define RES_TOL 2e-5f

__device__ __forceinline__ void eval_phi(float cp, float c2, float p[N_FEAT]) {
    float cp2 = cp*cp, cp3 = cp2*cp, cp4 = cp2*cp2;
    float c4 = c2*c2, c6 = c4*c2;
    float inv_dp = 1.0f/(1.0f+cp2);
    float inv_dpe = 1.0f/(1.0f+c2);
    p[0]=1.0f; p[1]=cp; p[2]=c2; p[3]=cp2;
    p[4]=cp*inv_dp; p[5]=cp*c2*inv_dpe;
    p[6]=c4; p[7]=cp2*c2; p[8]=cp4;
    p[9]=cp3*inv_dp*inv_dp;
    p[10]=c6*inv_dpe; p[11]=cp2*c4*inv_dpe;
}

__constant__ float np[16]={-6.5f,-5.5f,-4.5f,-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,-7.5f};
__constant__ float wp[16]={0.001f,0.004f,0.012f,0.03f,0.07f,0.13f,0.19f,0.19f,0.13f,0.07f,0.03f,0.012f,0.004f,0.001f,0.0003f,0.0003f};
__constant__ float nq[12]={0.2f,0.55f,1.0f,1.5f,2.1f,2.8f,3.6f,4.5f,5.6f,6.8f,8.2f,10.0f};
__constant__ float wq[12]={0.06f,0.12f,0.16f,0.17f,0.15f,0.12f,0.08f,0.05f,0.03f,0.015f,0.008f,0.003f};

__device__ __forceinline__ void project_lambda(float l[N_FEAT]) {
    l[6]  = fminf(l[6],  -1e-4f);
    l[7]  = fminf(l[7],  -1e-4f);
    l[8]  = fminf(l[8],  -1e-4f);
    l[10] = fminf(l[10], -1e-4f);
    l[11] = fminf(l[11], -1e-4f);
}

__device__ void project_moments(float M[N_FEAT]) {
    M[0] = 1.0f;
    M[2] = fmaxf(M[2], 1e-8f);
    M[3] = fmaxf(M[3], 1e-8f);
    M[6] = fmaxf(M[6], M[2]*M[2] + 1e-8f);
    M[8] = fmaxf(M[8], M[3]*M[3] + 1e-8f);
    float vth2_par  = M[3];
    float vth2_perp = M[2];
    float qmax_par  = 1.5f * sqrtf(vth2_par) * vth2_par;
    float qmax_perp = 1.5f * sqrtf(vth2_perp) * vth2_perp;
    M[4] = fmaxf(fminf(M[4], qmax_par),  -qmax_par);
    M[5] = fmaxf(fminf(M[5], qmax_perp), -qmax_perp);
}

__device__ float safe_exp(float expo) {
    return __expf(fminf(fmaxf(expo, -80.0f), 80.0f));
}

__device__ void moments_and_G(const float* lambda, float alpha_par, float alpha_perp,
                              float* M, float* G) {
    float Z = 0.0f;
    float M1[N_FEAT] = {0};
    float M2[N_FEAT*N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cp = np[ip], wpar = wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = nq[iq]*nq[iq];
            float w = wpar * wq[iq] * nq[iq];
            float p[N_FEAT];
            eval_phi(cp, c2, p);
            float expo = -alpha_par * cp*cp - alpha_perp * c2;
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) expo += lambda[k]*p[k];
            float weight = safe_exp(expo) * w;
            Z += weight;
            #pragma unroll
            for (int i = 0; i < N_FEAT; ++i) {
                float wpi = weight * p[i];
                M1[i] += wpi;
                #pragma unroll
                for (int j = 1; j <= i; ++j)
                    M2[i*N_FEAT + j] += wpi * p[j];
            }
        }
    }
    float invZ = 1.0f / fmaxf(Z, 1e-40f);
    #pragma unroll
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * invZ;
    for (int i = 1; i < N_FEAT; ++i) {
        for (int j = 1; j <= i; ++j) {
            float v = M2[i*N_FEAT + j]*invZ - M[i]*M[j];
            G[i*N_FEAT + j] = v;
            G[j*N_FEAT + i] = v;
        }
    }
    for (int k = 0; k < N_FEAT; ++k) {
        G[k*N_FEAT + 0] = 0.0f;
        G[0*N_FEAT + k] = 0.0f;
    }
    G[0] = 1.0f;
}

__device__ void moments_only(const float* lambda, float alpha_par, float alpha_perp, float* M) {
    float Z = 0.0f;
    float M1[N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cp = np[ip], wpar = wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = nq[iq]*nq[iq];
            float w = wpar * wq[iq] * nq[iq];
            float p[N_FEAT];
            eval_phi(cp, c2, p);
            float expo = -alpha_par * cp*cp - alpha_perp * c2;
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) expo += lambda[k]*p[k];
            float weight = safe_exp(expo) * w;
            Z += weight;
            #pragma unroll
            for (int i = 0; i < N_FEAT; ++i) M1[i] += weight * p[i];
        }
    }
    float invZ = 1.0f / fmaxf(Z, 1e-40f);
    #pragma unroll
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * invZ;
}

__device__ bool chol_solve_reduced(float* A, float* b, float* x) {
    for (int i = 1; i < N_FEAT; ++i) A[i*N_FEAT + i] += 1e-6f;
    float L[N_FEAT*N_FEAT] = {0};
    for (int i = 1; i < N_FEAT; ++i) {
        for (int j = 1; j <= i; ++j) {
            float s = A[i*N_FEAT + j];
            for (int k = 1; k < j; ++k) s -= L[i*N_FEAT + k]*L[j*N_FEAT + k];
            if (i == j) {
                if (s < 1e-12f) return false;
                L[i*N_FEAT + j] = sqrtf(s);
            } else L[i*N_FEAT + j] = s / L[j*N_FEAT + j];
        }
    }
    float y[N_FEAT] = {0};
    for (int i = 1; i < N_FEAT; ++i) {
        float s = b[i];
        for (int k = 1; k < i; ++k) s -= L[i*N_FEAT + k]*y[k];
        y[i] = s / L[i*N_FEAT + i];
    }
    for (int i = N_FEAT-1; i >= 1; --i) {
        float s = y[i];
        for (int k = i+1; k < N_FEAT; ++k) s -= L[k*N_FEAT + i]*x[k];
        x[i] = s / L[i*N_FEAT + i];
    }
    x[0] = 0.0f;
    return true;
}

extern "C" __global__
__launch_bounds__(128, 4)
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

    const float4* Mt4 = reinterpret_cast<const float4*>(M_target + idx*N_FEAT);
    float4* lam4 = reinterpret_cast<float4*>(lambda_io + idx*N_FEAT);

    float lambda[N_FEAT], lambda_backup[N_FEAT], Mt[N_FEAT];
    float4 t0=Mt4[0], t1=Mt4[1], t2=Mt4[2];
    float4 l0=lam4[0], l1=lam4[1], l2=lam4[2];

    Mt[0]=t0.x; Mt[1]=t0.y; Mt[2]=t0.z; Mt[3]=t0.w;
    Mt[4]=t1.x; Mt[5]=t1.y; Mt[6]=t1.z; Mt[7]=t1.w;
    Mt[8]=t2.x; Mt[9]=t2.y; Mt[10]=t2.z; Mt[11]=t2.w;
    lambda[0]=l0.x; lambda[1]=l0.y; lambda[2]=l0.z; lambda[3]=l0.w;
    lambda[4]=l1.x; lambda[5]=l1.y; lambda[6]=l1.z; lambda[7]=l1.w;
    lambda[8]=l2.x; lambda[9]=l2.y; lambda[10]=l2.z; lambda[11]=l2.w;

    // safe initial guess
    lambda[6]  = -0.1f;
    lambda[7]  = -0.1f;
    lambda[8]  = -0.1f;
    lambda[10] = -0.05f;
    lambda[11] = -0.05f;

    project_moments(Mt);
    project_lambda(lambda);
    #pragma unroll
    for (int k = 0; k < N_FEAT; ++k) lambda_backup[k] = lambda[k];

    float a_par  = fmaxf(alpha_par,  1e-4f);
    float a_perp = fmaxf(alpha_perp, 1e-4f);

    int flag = 0;
    float Mfinal[N_FEAT];

    for (int it = 0; it < MAX_NEWTON; ++it) {
        float M[N_FEAT], G[N_FEAT*N_FEAT];
        moments_and_G(lambda, a_par, a_perp, M, G);
        float max_r = 0.0f;
        #pragma unroll
        for (int k = 1; k < N_FEAT; ++k) max_r = fmaxf(max_r, fabsf(M[k]-Mt[k]));
        if (max_r < RES_TOL) {
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) Mfinal[k] = M[k];
            flag = 0;
            break;
        }
        float r[N_FEAT] = {0}, dlam[N_FEAT] = {0};
        #pragma unroll
        for (int k = 1; k < N_FEAT; ++k) r[k] = M[k] - Mt[k];
        if (!chol_solve_reduced(G, r, dlam)) { flag = 2; break; }

        float alpha = 1.0f;
        bool ok = false;
        for (int ls = 0; ls < 6; ++ls) {
            float trial[N_FEAT];
            #pragma unroll
            for (int k = 0; k < N_FEAT; ++k) trial[k] = lambda[k] - alpha * dlam[k];
            project_lambda(trial);
            float Mtst[N_FEAT];
            moments_only(trial, a_par, a_perp, Mtst);
            float max_rt = 0.0f;
            #pragma unroll
            for (int k = 1; k < N_FEAT; ++k) max_rt = fmaxf(max_rt, fabsf(Mtst[k]-Mt[k]));
            if (max_rt <= max_r * 1.01f) {
                #pragma unroll
                for (int k = 0; k < N_FEAT; ++k) lambda[k] = trial[k];
                ok = true;
                break;
            }
            alpha *= 0.5f;
        }
        if (!ok) { flag = 3; break; }
        if (it == MAX_NEWTON-1) {
            moments_only(lambda, a_par, a_perp, Mfinal);
            flag = 1;
        }
    }
    if (flag != 0) {
        #pragma unroll
        for (int k = 0; k < N_FEAT; ++k) lambda[k] = lambda_backup[k];
        moments_only(lambda, a_par, a_perp, Mfinal);
    }
    lam4[0] = make_float4(lambda[0],lambda[1],lambda[2],lambda[3]);
    lam4[1] = make_float4(lambda[4],lambda[5],lambda[6],lambda[7]);
    lam4[2] = make_float4(lambda[8],lambda[9],lambda[10],lambda[11]);
    status[idx] = flag;

    float n_phys = n_in[idx];
    float upar   = Mfinal[1];
    float raw_cpar2  = Mfinal[3];
    float raw_cperp2 = Mfinal[2];
    float cpar2 = raw_cpar2 - upar*upar;
    float Ppar  = mass * n_phys * fmaxf(cpar2, 0.0f);
    float Pperp = 0.5f * mass * n_phys * fmaxf(raw_cperp2, 0.0f);
    float qpar  = mass * n_phys * Mfinal[4] * thermal_scale;
    float qperp = mass * n_phys * Mfinal[5] * thermal_scale;

    n_out[idx]     = n_phys;
    upar_out[idx]  = upar;
    Ppar_out[idx]  = Ppar;
    Pperp_out[idx] = Pperp;
    qpar_out[idx]  = qpar;
    qperp_out[idx] = qperp;
}
''', 'gyrotropic_12_kernel')

# ---- test ----
Ncells = 4096
M_target = cp.zeros((Ncells, 12), dtype=cp.float32)
M_target[:, 0] = 1.0
M_target[:, 2] = 1.0
M_target[:, 3] = 1.0
M_target[:, 4] = 0.1
M_target[:, 5] = 0.05
n_in = cp.ones(Ncells, dtype=cp.float32)
lambda_io = cp.zeros((Ncells, 12), dtype=cp.float32)
n_out = cp.empty(Ncells, dtype=cp.float32)
upar = cp.empty(Ncells, dtype=cp.float32)
Ppar = cp.empty(Ncells, dtype=cp.float32)
Pperp = cp.empty(Ncells, dtype=cp.float32)
qpar = cp.empty(Ncells, dtype=cp.float32)
qperp = cp.empty(Ncells, dtype=cp.float32)
status = cp.zeros(Ncells, dtype=cp.int32)

threads = 128
blocks = (Ncells + 127) // 128
gyrotropic_12_kernel(
    (blocks,), (threads,),
    (M_target, n_in, lambda_io,
     n_out, upar, Ppar, Pperp, qpar, qperp,
     status, Ncells, 1.0, 1.0, 0.5, 0.5)
)
print("status unique:", cp.unique(status))
print("Ppar sample:", Ppar[:4])
print("Pperp sample:", Pperp[:4])
print("max |qpar|, |qperp|:", float(cp.max(cp.abs(qpar))), float(cp.max(cp.abs(qperp))))
