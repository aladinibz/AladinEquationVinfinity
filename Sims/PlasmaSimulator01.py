import subprocess, sys
try:
    import cupy as cp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    import cupy as cp
import numpy as np

# GPU check
try:
    assert cp.cuda.runtime.getDeviceCount() > 0
    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
except Exception:
    raise RuntimeError("Enable T4/A100 GPU: Runtime → Change runtime type → GPU")

# ---------- gyrotropic_12_kernel (eta/zeta/xi free, zeta step = 0.01) ----------
gyrotropic_12_kernel = cp.RawKernel(r'''
#define N_FEAT 12
#define MAX_NEWTON 8
#define RES_TOL 2.0e-3f

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

__device__ void set_from_free(float eta, float zeta, float xi, const float* low, float* lam) {
    float a = expf(eta), b = expf(zeta);
    float s = 0.95f * tanhf(xi);
    lam[0] = low[0];
    lam[1] = low[1];
    lam[2] = low[2];
    lam[3] = low[3];
    lam[4] = low[4];
    lam[5] = low[5];
    lam[6] = -b*b;
    lam[7] = -2.f*a*b*s;
    lam[8] = -a*a;
    lam[9] = 0.f;
    lam[10] = 0.f;
    lam[11] = 0.f;
}

__device__ bool moments(const float* lam, float ap, float ae,
                        float* M, float* m3_par, float* m3_mix) {
    float Z = 0.f, sum_c3 = 0.f, sum_mix = 0.f;
    for (int k = 0; k < N_FEAT; ++k) M[k] = 0.f;
    for (int ip = 0; ip < 16; ++ip) {
        float cp = np[ip];
        float w_par = wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = nq[iq]*nq[iq];
            float w = w_par * wq[iq] * expf(-0.5f*(ap*cp*cp + ae*c2));
            float expo = 0.f;
            float phi[N_FEAT];
            eval_phi(cp, c2, phi);
            for (int k = 0; k < N_FEAT; ++k) expo += lam[k]*phi[k];
            float f = w * expf(fminf(expo, 40.f));
            Z += f;
            for (int k = 0; k < N_FEAT; ++k) M[k] += f * phi[k];
            sum_c3  += f * cp*cp*cp;
            sum_mix += f * cp * c2;
        }
    }
    if (Z < 1e-30f) return false;
    float invZ = 1.f / Z;
    for (int k = 0; k < N_FEAT; ++k) M[k] *= invZ;
    *m3_par = sum_c3 * invZ;
    *m3_mix = sum_mix * invZ;
    return true;
}

extern "C" __global__
void gyrotropic_12_kernel(
    const float* M_target, const float* n_in, float* lambda_io,
    float* n_out, float* upar_out, float* Ppar_out, float* Pperp_out,
    float* qpar_out, float* qperp_out, int* status,
    int Ncells, float mass, float thermal_scale, float alpha_par, float alpha_perp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncells) return;

    const float* Mt = M_target + idx * N_FEAT;
    float* lo = lambda_io + idx * N_FEAT;

    float eta = 0.f, zeta = 0.f, xi = 0.f;
    float low[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lam[N_FEAT];
    float M[N_FEAT];
    float m3_par = 0.f, m3_mix = 0.f;

    int flag = 0;
    for (int it = 0; it < MAX_NEWTON; ++it) {
        set_from_free(eta, zeta, xi, low, lam);
        if (!moments(lam, alpha_par, alpha_perp, M, &m3_par, &m3_mix)) { flag = 6; break; }

        float res = 0.f;
        for (int k = 1; k < N_FEAT; ++k) res = fmaxf(res, fabsf(M[k] - Mt[k]));
        if (res < RES_TOL) { flag = 0; break; }

        for (int k = 0; k < 6; ++k) low[k] += 0.12f * (Mt[k+1] - M[k+1]);

        float r6 = Mt[6] - M[6];
        float r7 = Mt[7] - M[7];
        float r8 = Mt[8] - M[8];
        eta  += 0.04f * r8;
        zeta += 0.01f * r6;
        xi   += 0.02f * r7;
        eta  = fmaxf(fminf(eta,  2.5f), -2.5f);
        zeta = fmaxf(fminf(zeta, 2.5f), -2.5f);
        xi   = fmaxf(fminf(xi,   2.0f), -2.0f);

        if (it == MAX_NEWTON-1) flag = 1;
    }

    float n = n_in[idx];
    float up = M[1];
    float cp2 = fmaxf(M[3] - up*up, 1e-6f);
    float ce2 = fmaxf(M[2], 1e-6f);

    float qpar  = mass * n * (m3_par - 3.f*up*cp2 - up*up*up);
    float qperp = 0.5f * mass * n * (m3_mix - up * M[2]);

    n_out[idx] = n;
    upar_out[idx] = up;
    Ppar_out[idx]  = mass * n * cp2;
    Pperp_out[idx] = 0.5f * mass * n * ce2;
    qpar_out[idx]  = qpar;
    qperp_out[idx] = qperp;
    status[idx] = flag;
    for (int k = 0; k < N_FEAT; ++k) lo[k] = lam[k];
}
''', 'gyrotropic_12_kernel')

# ---------- ct_emf with anisotropic pressure (scale = 1.0) ----------
ct_emf_kernel = cp.RawKernel(r'''
extern "C" __global__
void utc_emf_kernel(
    const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz,
    const float* Ppar, const float* Pperp,
    float* Emfx, float* Emfy, float* Emfz, int Ni)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;

    int g = i * Ni * Ni + j * Ni + k;
    float r = fmaxf(rho[g], 1e-8f);
    float vx = mx[g] / r;
    float vy = my[g] / r;
    float vz = mz[g] / r;

    float Bz_yp = (k+1 < Ni) ? Bz[g + 1] : Bz[g];
    float By_zp = (j+1 < Ni) ? By[g + Ni] : By[g];
    float Bx_xp = (i+1 < Ni) ? Bx[g + Ni*Ni] : Bx[g];
    float Bz_xp = (i+1 < Ni) ? Bz[g + Ni*Ni] : Bz[g];
    float By_yp = (k+1 < Ni) ? By[g + 1] : By[g];
    float Bx_yp = (j+1 < Ni) ? Bx[g + Ni] : Bx[g];

    float Ex = -(vy * 0.5f * (Bz[g] + Bz_yp) - vz * 0.5f * (By[g] + By_zp));
    float Ey = -(vz * 0.5f * (Bx[g] + Bx_xp) - vx * 0.5f * (Bz[g] + Bz_xp));
    float Ez = -(vx * 0.5f * (By[g] + By_yp) - vy * 0.5f * (Bx[g] + Bx_yp));

    float dPpar_dx  = (Ppar[g + Ni*Ni] - Ppar[g - Ni*Ni]) * 0.5f;
    float dPpar_dy  = (Ppar[g + Ni]    - Ppar[g - Ni])    * 0.5f;
    float dPpar_dz  = (Ppar[g + 1]     - Ppar[g - 1])     * 0.5f;
    float dPperp_dx = (Pperp[g + Ni*Ni] - Pperp[g - Ni*Ni]) * 0.5f;
    float dPperp_dy = (Pperp[g + Ni]    - Pperp[g - Ni])    * 0.5f;
    float dPperp_dz = (Pperp[g + 1]     - Pperp[g - 1])     * 0.5f;

    float Bmag = sqrtf(Bx[g]*Bx[g] + By[g]*By[g] + Bz[g]*Bz[g]) + 1e-8f;
    float bx = Bx[g] / Bmag, by = By[g] / Bmag, bz = Bz[g] / Bmag;
    float dP = (Ppar[g] - Pperp[g]);

    float scale = 1.0f / r;
    Ex += scale * (dP * bx * (bx*dPpar_dx + by*dPpar_dy + bz*dPpar_dz) - dPperp_dx);
    Ey += scale * (dP * by * (bx*dPpar_dx + by*dPpar_dy + bz*dPpar_dz) - dPperp_dy);
    Ez += scale * (dP * bz * (bx*dPpar_dx + by*dPpar_dy + bz*dPpar_dz) - dPperp_dz);

    Emfx[g] = Ex;
    Emfy[g] = Ey;
    Emfz[g] = Ez;
}
''', 'utc_emf_kernel')

# ---------- ct_curl (three-component) ----------
ct_curl_kernel = cp.RawKernel(r'''
extern "C" __global__
void ct_curl_kernel(
    const float* Emfx, const float* Emfy, const float* Emfz,
    float* Bx, float* By, float* Bz,
    int Ni, float dt, float inv2dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
    int idx = i * Ni * Ni + j * Ni + k;

    float dEz_dy = (Emfz[idx + Ni] - Emfz[idx - Ni]) * inv2dx;
    float dEy_dz = (Emfy[idx + 1]  - Emfy[idx - 1])  * inv2dx;
    Bx[idx] -= dt * (dEz_dy - dEy_dz);

    float dEx_dz = (Emfx[idx + 1]     - Emfx[idx - 1])     * inv2dx;
    float dEz_dx = (Emfz[idx + Ni*Ni] - Emfz[idx - Ni*Ni]) * inv2dx;
    By[idx] -= dt * (dEx_dz - dEz_dx);

    float dEy_dx = (Emfy[idx + Ni*Ni] - Emfy[idx - Ni*Ni]) * inv2dx;
    float dEx_dy = (Emfx[idx + Ni]    - Emfx[idx - Ni])    * inv2dx;
    Bz[idx] -= dt * (dEy_dx - dEx_dy);
}
''', 'ct_curl_kernel')

# ---------- glm (race-free: separate psi_new buffer) ----------
glm_kernel = cp.RawKernel(r'''
extern "C" __global__
void glm_kernel(const float* Bx, const float* By, const float* Bz,
                const float* psi_old, float* psi_new,
                float* Bx_out, float* By_out, float* Bz_out,
                int Ni, float dt, float dx, float ch, float damping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
    int idx = i * Ni * Ni + j * Ni + k;
    float inv2dx = 0.5f / dx;
    float divB = ((Bx[idx+Ni*Ni] - Bx[idx-Ni*Ni]) +
                  (By[idx+Ni]    - By[idx-Ni]) +
                  (Bz[idx+1]     - Bz[idx-1])) * inv2dx;
    float psi_upd = psi_old[idx] - dt * (ch*ch*divB + damping * psi_old[idx]);
    psi_new[idx] = psi_upd;
    Bx_out[idx] = Bx[idx] - dt * (psi_old[idx+Ni*Ni] - psi_old[idx-Ni*Ni]) * inv2dx;
    By_out[idx] = By[idx] - dt * (psi_old[idx+Ni]    - psi_old[idx-Ni])    * inv2dx;
    Bz_out[idx] = Bz[idx] - dt * (psi_old[idx+1]     - psi_old[idx-1])     * inv2dx;
}
''', 'glm_kernel')

# ---------- Driver ----------
Ni = 64
Ncells = Ni * Ni * Ni
dx = np.float32(1.0)
dt = np.float32(0.001)
inv2dx = np.float32(0.5 / dx)
mass = np.float32(1.0)
thermal_scale = np.float32(1.0)
alpha_par = np.float32(0.5)
alpha_perp = np.float32(0.5)
n_steps = 20

rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx = cp.zeros_like(rho)
my = cp.zeros_like(rho)
mz = cp.zeros_like(rho)
Bx = cp.zeros_like(rho)
By = cp.zeros_like(rho)
Bz = cp.ones_like(rho) * 0.05
psi = cp.zeros_like(rho)
psi_new = cp.zeros_like(rho)
Bx_new = cp.zeros_like(rho)
By_new = cp.zeros_like(rho)
Bz_new = cp.zeros_like(rho)
Emfx = cp.zeros_like(rho)
Emfy = cp.zeros_like(rho)
Emfz = cp.zeros_like(rho)

M_target = cp.zeros((Ncells, 12), dtype=cp.float32)
lambda_io = cp.zeros((Ncells, 12), dtype=cp.float32)
n_in = cp.ones(Ncells, dtype=cp.float32)
n_out = cp.zeros(Ncells, dtype=cp.float32)
upar = cp.zeros(Ncells, dtype=cp.float32)
Ppar = cp.zeros(Ncells, dtype=cp.float32)
Pperp = cp.zeros(Ncells, dtype=cp.float32)
qpar = cp.zeros(Ncells, dtype=cp.float32)
qperp = cp.zeros(Ncells, dtype=cp.float32)
status = cp.zeros(Ncells, dtype=cp.int32)

def init_disk():
    i = cp.arange(Ni, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(i, i, i, indexing='ij')
    R = cp.sqrt((X - Ni/2)**2 + (Y - Ni/2)**2) + 1e-3
    rho[:] = cp.clip(cp.exp(-R/12) * cp.exp(-cp.abs(Z - Ni/2)/4), 0.05, None)
    vphi = cp.sqrt(1.0 / (R + 2.0))
    mx[:] = -vphi * (Y - Ni/2) / R * rho
    my[:] =  vphi * (X - Ni/2) / R * rho
init_disk()

threads = (16, 4, 4)
blocks = ((Ni + 15) // 16, (Ni + 3) // 4, (Ni + 3) // 4)
threads_g = 128
blocks_g = (Ncells + 127) // 128

def build_M_target():
    r = rho.ravel() + 1e-8
    M_target[:, 0] = 1.0
    M_target[:, 2] = cp.clip(2.0 * Pperp / r, 1e-6, 8.0)
    M_target[:, 3] = cp.clip(Ppar / r,        1e-6, 8.0)

print("Plasma simulator starting (race-free GLM) ...")
for step in range(n_steps):
    build_M_target()
    gyrotropic_12_kernel((blocks_g,), (threads_g,),
        (M_target, n_in, lambda_io, n_out, upar, Ppar, Pperp, qpar, qperp,
         status, np.int32(Ncells), mass, thermal_scale, alpha_par, alpha_perp))
    ct_emf_kernel(blocks, threads,
        (rho, mx, my, mz, Bx, By, Bz, Ppar, Pperp, Emfx, Emfy, Emfz, np.int32(Ni)))
    ct_curl_kernel(blocks, threads,
        (Emfx, Emfy, Emfz, Bx, By, Bz, np.int32(Ni), dt, inv2dx))
    glm_kernel(blocks, threads,
        (Bx, By, Bz, psi, psi_new, Bx_new, By_new, Bz_new,
         np.int32(Ni), dt, dx, np.float32(1.0), np.float32(0.5)))
    # swap buffers
    psi, psi_new = psi_new, psi
    Bx, Bx_new = Bx_new, Bx
    By, By_new = By_new, By
    Bz, Bz_new = Bz_new, Bz

    Bmag = cp.sqrt(Bx**2 + By**2 + Bz**2)
    if step % 5 == 0 or not cp.all(cp.isfinite(Bmag)):
        uniq, counts = cp.unique(status, return_counts=True)
        ratio = float(cp.mean(Ppar) / (cp.mean(Pperp) + 1e-12))
        print(f"step {step:3d}  status={dict(zip(uniq.get().tolist(), counts.get().tolist()))}  "
              f"mean|B|={float(cp.mean(Bmag)):.6f}  max|B|={float(cp.max(Bmag)):.6f}  "
              f"Ppar={float(cp.mean(Ppar)):.6f} (min {float(cp.min(Ppar)):.6f})  "
              f"Pperp={float(cp.mean(Pperp)):.6f} (min {float(cp.min(Pperp)):.6f})  "
              f"ratio={ratio:.4f}")
        if not cp.all(cp.isfinite(Bmag)):
            print("NaN detected — stopping")
            break

print("Done.")
print("Final mean Ppar / Pperp =", float(cp.mean(Ppar)), float(cp.mean(Pperp)))
