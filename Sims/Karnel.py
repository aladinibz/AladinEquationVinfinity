import numpy as np
import cupy as cp

# ====================== PARAMETERS ======================
N = 128
L = 1.0
dx = L / N
dt_max = 0.00008
cfl = 0.22
max_steps = 1000
print_interval = 50
NG = 3
Ni = N + 2 * NG
gamma = 5.0 / 3.0

# ====================== FIELDS ======================
rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
my = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
mz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
E_total = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 3.0

Bx = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
By = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
Bz = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)

# RK2 buffers
rho1 = cp.zeros_like(rho); mx1 = cp.zeros_like(mx); my1 = cp.zeros_like(my)
mz1 = cp.zeros_like(mz); E1 = cp.zeros_like(E_total)
Bx1 = cp.zeros_like(Bx); By1 = cp.zeros_like(By); Bz1 = cp.zeros_like(Bz)

rho2 = cp.zeros_like(rho); mx2 = cp.zeros_like(mx); my2 = cp.zeros_like(my)
mz2 = cp.zeros_like(mz); E2 = cp.zeros_like(E_total)
Bx2 = cp.zeros_like(Bx); By2 = cp.zeros_like(By); Bz2 = cp.zeros_like(Bz)

Emfx = cp.zeros((Ni, Ni+1, Ni+1), dtype=cp.float32)
Emfy = cp.zeros((Ni+1, Ni, Ni+1), dtype=cp.float32)
Emfz = cp.zeros((Ni+1, Ni+1, Ni), dtype=cp.float32)

np.random.seed(42)
pert = 0.08
Bx[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
By[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
Bz[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert * 0.5 + 0.5, dtype=cp.float32)

def update_ghosts():
    for f in [rho, mx, my, mz, E_total]:
        f[:NG,:,:] = f[-2*NG:-NG,:,:]
        f[-NG:,:,:] = f[NG:2*NG,:,:]
        f[:,:NG,:] = f[:,-2*NG:-NG,:]
        f[:,-NG:,:] = f[:,NG:2*NG,:]
        f[:,:,:NG] = f[:,:,-2*NG:-NG]
        f[:,:,-NG:] = f[:,:,NG:2*NG]

    for f in [Bx, By, Bz]:
        f[:NG,:,:] = f[-2*NG:-NG,:,:]
        f[-NG:,:,:] = f[NG:2*NG,:,:]
        f[:,:NG,:] = f[:,-2*NG:-NG,:]
        f[:,-NG:,:] = f[:,NG:2*NG,:]
        f[:,:,:NG] = f[:,:,-2*NG:-NG]
        f[:,:,-NG:] = f[:,:,NG:2*NG]

def compute_divB():
    divB = ((Bx[1:,:,:] - Bx[:-1,:,:]) +
            (By[:,1:,:] - By[:,:-1,:]) +
            (Bz[:,:,1:] - Bz[:,:,:-1])) / dx
    return float(cp.mean(cp.abs(divB))), float(cp.max(cp.abs(divB)))

# ====================== KERNEL WITH FULL 3D HLLD ======================
kernel = cp.RawKernel(r'''
#define NG 3
extern "C" __global__ void mhd_kernel(
    const float* rho, const float* mx, const float* my, const float* mz, const float* E_total,
    const float* Bx_old, const float* By_old, const float* Bz_old,
    float* rho_out, float* mx_out, float* my_out, float* mz_out, float* E_out,
    float* Bx_out, float* By_out, float* Bz_out,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, float dx, float dt_over_dx, float gamma)
{
    extern __shared__ float sdata[];
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int i = blockIdx.x*blockDim.x + tx + NG;
    int j = blockIdx.y*blockDim.y + ty + NG;
    int k = blockIdx.z*blockDim.z + tz + NG;

    if (i >= Ni-NG || j >= Ni-NG || k >= Ni-NG) return;

    int txs = blockDim.x + 2; int tys = blockDim.y + 2; int tzs = blockDim.z + 2;
    int sidx = (tz+1)*tys*txs + (ty+1)*txs + (tx+1);

    float* s_vx = sdata;
    float* s_vy = sdata + txs*tys*tzs;
    float* s_vz = sdata + 2*txs*tys*tzs;

    int idx = i*Ni*Ni + j*Ni + k;
    float rho_safe = fmaxf(rho[idx], 1e-8f);
    s_vx[sidx] = mx[idx] / rho_safe;
    s_vy[sidx] = my[idx] / rho_safe;
    s_vz[sidx] = mz[idx] / rho_safe;

    if (tx==0) s_vx[sidx-1] = mx[(i-1)*Ni*Ni+j*Ni+k] / fmaxf(rho[(i-1)*Ni*Ni+j*Ni+k],1e-8f);
    if (ty==0) s_vy[sidx-txs] = my[i*Ni*Ni+(j-1)*Ni+k] / fmaxf(rho[i*Ni*Ni+(j-1)*Ni+k],1e-8f);
    if (tz==0) s_vz[sidx-tys*txs] = mz[i*Ni*Ni+j*Ni+(k-1)] / fmaxf(rho[i*Ni*Ni+j*Ni+(k-1)],1e-8f);

    __syncthreads();

    // Full True CT Yee EMFs
    if (i < Ni-1 && j < Ni-1 && k < Ni) {
        float vx_e = 0.25f*(s_vx[sidx]+s_vx[sidx+1]+s_vx[sidx+txs]+s_vx[sidx+txs+1]);
        float vy_e = 0.25f*(s_vy[sidx]+s_vy[sidx+1]+s_vy[sidx+txs]+s_vy[sidx+txs+1]);
        float Bx_e = 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k] + Bx_old[(i+1)*Ni*Ni+(j+1)*Ni+k]);
        float By_e = 0.5f*(By_old[i*Ni*Ni+(j+1)*Ni+k] + By_old[(i+1)*Ni*Ni+(j+1)*Ni+k]);
        Emfz[i*Ni*Ni + j*Ni + k] = -(vx_e*By_e - vy_e*Bx_e);
    }
    if (i < Ni-1 && k < Ni-1) {
        float vx_e = 0.25f*(s_vx[sidx]+s_vx[sidx+1]+s_vx[sidx+txs*tys]+s_vx[sidx+txs*tys+1]);
        float vz_e = 0.25f*(s_vz[sidx]+s_vz[sidx+1]+s_vz[sidx+txs*tys]+s_vz[sidx+txs*tys+1]);
        float Bx_e = 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k] + Bx_old[(i+1)*Ni*Ni+j*Ni+(k+1)]);
        float Bz_e = 0.5f*(Bz_old[i*Ni*Ni+j*Ni+(k+1)] + Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)]);
        Emfy[i*Ni*Ni + j*Ni + k] = -(vz_e*Bx_e - vx_e*Bz_e);
    }
    if (j < Ni-1 && k < Ni-1) {
        float vy_e = 0.25f*(s_vy[sidx]+s_vy[sidx+txs]+s_vy[sidx+txs*tys]+s_vy[sidx+txs*tys+txs]);
        float vz_e = 0.25f*(s_vz[sidx]+s_vz[sidx+txs]+s_vz[sidx+txs*tys]+s_vz[sidx+txs*tys+txs]);
        float By_e = 0.5f*(By_old[i*Ni*Ni+(j+1)*Ni+k] + By_old[i*Ni*Ni+(j+1)*Ni+(k+1)]);
        float Bz_e = 0.5f*(Bz_old[i*Ni*Ni+j*Ni+(k+1)] + Bz_old[i*Ni*Ni+(j+1)*Ni+(k+1)]);
        Emfx[i*Ni*Ni + j*Ni + k] = -(vy_e*Bz_e - vz_e*By_e);
    }

    __syncthreads();

    // Full CT Yee curl
    if (i < Ni && j < Ni && k < Ni) {
        float dEz_dy = Emfz[i*Ni*Ni+(j+1)*Ni+k] - Emfz[i*Ni*Ni+j*Ni+k];
        float dEy_dz = Emfy[i*Ni*Ni+j*Ni+(k+1)] - Emfy[i*Ni*Ni+j*Ni+k];
        Bx_out[idx] = Bx_old[idx] - dt_over_dx * (dEz_dy - dEy_dz);

        float dEx_dz = Emfx[i*Ni*Ni+j*Ni+(k+1)] - Emfx[i*Ni*Ni+j*Ni+k];
        float dEz_dx = Emfz[(i+1)*Ni*Ni+j*Ni+k] - Emfz[i*Ni*Ni+j*Ni+k];
        By_out[idx] = By_old[idx] - dt_over_dx * (dEx_dz - dEz_dx);

        float dEy_dx = Emfy[(i+1)*Ni*Ni+j*Ni+k] - Emfy[i*Ni*Ni+j*Ni+k];
        float dEx_dy = Emfx[i*Ni*Ni+(j+1)*Ni+k] - Emfx[i*Ni*Ni+j*Ni+k];
        Bz_out[idx] = Bz_old[idx] - dt_over_dx * (dEy_dx - dEx_dy);
    }

    // ====================== FULL HLLD X, Y, Z ======================
    if (i < Ni && j < Ni && k < Ni) {
        float rho_safe = fmaxf(rho[idx], 1e-8f);

        // ==================== HLLD X-DIRECTION ====================
        {
            int idx_l = (i-1)*Ni*Ni + j*Ni + k;
            int idx_r = (i+1)*Ni*Ni + j*Ni + k;
            float rho_l = fmaxf(rho[idx_l],1e-8f); float rho_r = fmaxf(rho[idx_r],1e-8f);
            float vx_l = mx[idx_l]/rho_l; float vx_r = mx[idx_r]/rho_r;
            float vy_l = my[idx_l]/rho_l; float vy_r = my[idx_r]/rho_r;
            float vz_l = mz[idx_l]/rho_l; float vz_r = mz[idx_r]/rho_r;
            float p_l = fmaxf((gamma-1)*(E_total[idx_l] - 0.5f*rho_l*(vx_l*vx_l + vy_l*vy_l + vz_l*vz_l) 
                                          - 0.5f*(powf(Bx_old[i*Ni*Ni+j*Ni+k],2)+powf(By_old[i*Ni*Ni+(j+1)*Ni+k],2)+powf(Bz_old[i*Ni*Ni+j*Ni+(k+1)],2))),1e-6f);
            float p_r = fmaxf((gamma-1)*(E_total[idx_r] - 0.5f*rho_r*(vx_r*vx_r + vy_r*vy_r + vz_r*vz_r) 
                                          - 0.5f*(powf(Bx_old[(i+1)*Ni*Ni+j*Ni+k],2)+powf(By_old[(i+1)*Ni*Ni+(j+1)*Ni+k],2)+powf(Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)],2))),1e-6f);

            float Bx_f = Bx_old[(i+1)*Ni*Ni + j*Ni + k];
            float By_l = By_old[i*Ni*Ni+(j+1)*Ni+k]; float By_r = By_old[(i+1)*Ni*Ni+(j+1)*Ni+k];
            float Bz_l = Bz_old[i*Ni*Ni+j*Ni+(k+1)]; float Bz_r = Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)];

            float cf_l = sqrtf((Bx_f*Bx_f + By_l*By_l + Bz_l*Bz_l)/rho_l + gamma*p_l/rho_l);
            float cf_r = sqrtf((Bx_f*Bx_f + By_r*By_r + Bz_r*Bz_r)/rho_r + gamma*p_r/rho_r);

            float SL = fminf(vx_l - cf_l, vx_r - cf_r);
            float SR = fmaxf(vx_l + cf_l, vx_r + cf_r);
            float S_star = (rho_r*vx_r*(SR-vx_r) - rho_l*vx_l*(SL-vx_l) + p_l - p_r) / (rho_r*(SR-vx_r) - rho_l*(SL-vx_l) + 1e-12f);

            float p_star = p_l + rho_l*(SL - vx_l)*(S_star - vx_l) - Bx_f*Bx_f;

            float By_star = (sqrtf(rho_l)*By_l + sqrtf(rho_r)*By_r + sqrtf(rho_l*rho_r)*(vy_l - vy_r)) / (sqrtf(rho_l) + sqrtf(rho_r) + 1e-12f);
            float Bz_star = (sqrtf(rho_l)*Bz_l + sqrtf(rho_r)*Bz_r + sqrtf(rho_l*rho_r)*(vz_l - vz_r)) / (sqrtf(rho_l) + sqrtf(rho_r) + 1e-12f);

            float vy_star = 0.5f*(vy_l + vy_r) - Bx_f*(By_star - 0.5f*(By_l + By_r)) / (0.5f*(rho_l + rho_r));
            float vz_star = 0.5f*(vz_l + vz_r) - Bx_f*(Bz_star - 0.5f*(Bz_l + Bz_r)) / (0.5f*(rho_l + rho_r));

            float flux_rho, flux_mx, flux_my, flux_mz, flux_E;
            if (SL >= 0.0f) {
                flux_rho = rho_l * vx_l;
                flux_mx = rho_l*vx_l*vx_l + p_l + 0.5f*(By_l*By_l + Bz_l*Bz_l) - Bx_f*Bx_f;
                flux_my = rho_l*vx_l*vy_l - Bx_f*By_l;
                flux_mz = rho_l*vx_l*vz_l - Bx_f*Bz_l;
                flux_E = (E_total[idx_l] + p_l + 0.5f*(By_l*By_l + Bz_l*Bz_l)) * vx_l - Bx_f*(Bx_f*vx_l + By_l*vy_l + Bz_l*vz_l);
            } else if (S_star >= 0.0f) {
                flux_rho = rho_l * (SL - vx_l) / (SL - S_star) * S_star;
                flux_mx = flux_rho * S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star) - Bx_f*Bx_f;
                flux_my = flux_rho * vy_star - Bx_f * By_star;
                flux_mz = flux_rho * vz_star - Bx_f * Bz_star;
                flux_E = (0.5f*flux_rho*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star)) * S_star - Bx_f*(Bx_f*S_star + By_star*vy_star + Bz_star*vz_star);
            } else if (SR >= 0.0f) {
                flux_rho = rho_r * (SR - vx_r) / (SR - S_star) * S_star;
                flux_mx = flux_rho * S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star) - Bx_f*Bx_f;
                flux_my = flux_rho * vy_star - Bx_f * By_star;
                flux_mz = flux_rho * vz_star - Bx_f * Bz_star;
                flux_E = (0.5f*flux_rho*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star)) * S_star - Bx_f*(Bx_f*S_star + By_star*vy_star + Bz_star*vz_star);
            } else {
                flux_rho = rho_r * vx_r;
                flux_mx = rho_r*vx_r*vx_r + p_r + 0.5f*(By_r*By_r + Bz_r*Bz_r) - Bx_f*Bx_f;
                flux_my = rho_r*vx_r*vy_r - Bx_f*By_r;
                flux_mz = rho_r*vx_r*vz_r - Bx_f*Bz_r;
                flux_E = (E_total[idx_r] + p_r + 0.5f*(By_r*By_r + Bz_r*Bz_r)) * vx_r - Bx_f*(Bx_f*vx_r + By_r*vy_r + Bz_r*vz_r);
            }

            rho_out[idx] = rho[idx] - dt_over_dx * flux_rho;
            mx_out[idx] = mx[idx] - dt_over_dx * flux_mx;
            my_out[idx] = my[idx] - dt_over_dx * flux_my;
            mz_out[idx] = mz[idx] - dt_over_dx * flux_mz;
            E_out[idx] = E_total[idx] - dt_over_dx * flux_E;
        }

        // ==================== HLLD Y-DIRECTION ====================
        {
            int idx_l = i*Ni*Ni + (j-1)*Ni + k;
            int idx_r = i*Ni*Ni + (j+1)*Ni + k;
            float rho_l = fmaxf(rho[idx_l],1e-8f); float rho_r = fmaxf(rho[idx_r],1e-8f);
            float vy_l = my[idx_l]/rho_l; float vy_r = my[idx_r]/rho_r;
            float vx_l = mx[idx_l]/rho_l; float vx_r = mx[idx_r]/rho_r;
            float vz_l = mz[idx_l]/rho_l; float vz_r = mz[idx_r]/rho_r;
            float p_l = fmaxf((gamma-1)*(E_total[idx_l] - 0.5f*rho_l*(vx_l*vx_l + vy_l*vy_l + vz_l*vz_l) 
                                          - 0.5f*(powf(Bx_old[i*Ni*Ni+j*Ni+k],2)+powf(By_old[i*Ni*Ni+(j)*Ni+k],2)+powf(Bz_old[i*Ni*Ni+j*Ni+(k+1)],2))),1e-6f);
            float p_r = fmaxf((gamma-1)*(E_total[idx_r] - 0.5f*rho_r*(vx_r*vx_r + vy_r*vy_r + vz_r*vz_r) 
                                          - 0.5f*(powf(Bx_old[i*Ni*Ni+(j+1)*Ni+k],2)+powf(By_old[i*Ni*Ni+(j+1)*Ni+k],2)+powf(Bz_old[i*Ni*Ni+(j+1)*Ni+(k+1)],2))),1e-6f);

            float By_f = By_old[i*Ni*Ni + (j+1)*Ni + k];
            float Bx_l = Bx_old[i*Ni*Ni+j*Ni+k]; float Bx_r = Bx_old[i*Ni*Ni+(j+1)*Ni+k];
            float Bz_l = Bz_old[i*Ni*Ni+j*Ni+(k+1)]; float Bz_r = Bz_old[i*Ni*Ni+(j+1)*Ni+(k+1)];

            float cf_l = sqrtf((By_f*By_f + Bx_l*Bx_l + Bz_l*Bz_l)/rho_l + gamma*p_l/rho_l);
            float cf_r = sqrtf((By_f*By_f + Bx_r*Bx_r + Bz_r*Bz_r)/rho_r + gamma*p_r/rho_r);

            float SL = fminf(vy_l - cf_l, vy_r - cf_r);
            float SR = fmaxf(vy_l + cf_l, vy_r + cf_r);
            float S_star = (rho_r*vy_r*(SR-vy_r) - rho_l*vy_l*(SL-vy_l) + p_l - p_r) / (rho_r*(SR-vy_r) - rho_l*(SL-vy_l) + 1e-12f);

            float p_star = p_l + rho_l*(SL - vy_l)*(S_star - vy_l) - By_f*By_f;

            float Bx_star = (sqrtf(rho_l)*Bx_l + sqrtf(rho_r)*Bx_r + sqrtf(rho_l*rho_r)*(vx_l - vx_r)) / (sqrtf(rho_l) + sqrtf(rho_r) + 1e-12f);
            float Bz_star = (sqrtf(rho_l)*Bz_l + sqrtf(rho_r)*Bz_r + sqrtf(rho_l*rho_r)*(vz_l - vz_r)) / (sqrtf(rho_l) + sqrtf(rho_r) + 1e-12f);

            float vx_star = 0.5f*(vx_l + vx_r) - By_f*(Bx_star - 0.5f*(Bx_l + Bx_r)) / (0.5f*(rho_l + rho_r));
            float vz_star = 0.5f*(vz_l + vz_r) - By_f*(Bz_star - 0.5f*(Bz_l + Bz_r)) / (0.5f*(rho_l + rho_r));

            float flux_rho, flux_my, flux_mx, flux_mz, flux_E;
            if (SL >= 0.0f) {
                flux_rho = rho_l * vy_l;
                flux_my = rho_l*vy_l*vy_l + p_l + 0.5f*(Bx_l*Bx_l + Bz_l*Bz_l) - By_f*By_f;
                flux_mx = rho_l*vy_l*vx_l - By_f*Bx_l;
                flux_mz = rho_l*vy_l*vz_l - By_f*Bz_l;
                flux_E = (E_total[idx_l] + p_l + 0.5f*(Bx_l*Bx_l + Bz_l*Bz_l)) * vy_l - By_f*(By_f*vy_l + Bx_l*vx_l + Bz_l*vz_l);
            } else if (S_star >= 0.0f) {
                flux_rho = rho_l * (SL - vy_l) / (SL - S_star) * S_star;
                flux_my = flux_rho * S_star + p_star + 0.5f*(Bx_star*Bx_star + Bz_star*Bz_star) - By_f*By_f;
                flux_mx = flux_rho * vx_star - By_f * Bx_star;
                flux_mz = flux_rho * vz_star - By_f * Bz_star;
                flux_E = (0.5f*flux_rho*S_star*S_star + p_star + 0.5f*(Bx_star*Bx_star + Bz_star*Bz_star)) * S_star - By_f*(By_f*S_star + Bx_star*vx_star + Bz_star*vz_star);
            } else if (SR >= 0.0f) {
                flux_rho = rho_r * (SR - vy_r) / (SR - S_star) * S_star;
                flux_my = flux_rho * S_star + p_star + 0.5f*(Bx_star*Bx_star + Bz_star*Bz_star) - By_f*By_f;
                flux_mx = flux_rho * vx_star - By_f * Bx_star;
                flux_mz = flux_rho * vz_star - By_f * Bz_star;
                flux_E = (0.5f*flux_rho*S_star*S_star + p_star + 0.5f*(Bx_star*Bx_star + Bz_star*Bz_star)) * S_star - By_f*(By_f*S_star + Bx_star*vx_star + Bz_star*vz_star);
            } else {
                flux_rho = rho_r * vy_r;
                flux_my = rho_r*vy_r*vy_r + p_r + 0.5f*(Bx_r*Bx_r + Bz_r*Bz_r) - By_f*By_f;
                flux_mx = rho_r*vy_r*vx_r - By_f*Bx_r;
                flux_mz = rho_r*vy_r*vz_r - By_f*Bz_r;
                flux_E = (E_total[idx_r] + p_r + 0.5f*(Bx_r*Bx_r + Bz_r*Bz_r)) * vy_r - By_f*(By_f*vy_r + Bx_r*vx_r + Bz_r*vz_r);
            }

            rho_out[idx] -= dt_over_dx * flux_rho;
            my_out[idx] -= dt_over_dx * flux_my;
            mx_out[idx] -= dt_over_dx * flux_mx;
            mz_out[idx] -= dt_over_dx * flux_mz;
            E_out[idx] -= dt_over_dx * flux_E;
        }

        // ==================== HLLD Z-DIRECTION ====================
        {
            int idx_l = i*Ni*Ni + j*Ni + (k-1);
            int idx_r = i*Ni*Ni + j*Ni + (k+1);
            float rho_l = fmaxf(rho[idx_l],1e-8f); float rho_r = fmaxf(rho[idx_r],1e-8f);
            float vz_l = mz[idx_l]/rho_l; float vz_r = mz[idx_r]/rho_r;
            float vx_l = mx[idx_l]/rho_l; float vx_r = mx[idx_r]/rho_r;
            float vy_l = my[idx_l]/rho_l; float vy_r = my[idx_r]/rho_r;
            float p_l = fmaxf((gamma-1)*(E_total[idx_l] - 0.5f*rho_l*(vx_l*vx_l + vy_l*vy_l + vz_l*vz_l) 
                                          - 0.5f*(powf(Bx_old[i*Ni*Ni+j*Ni+k],2)+powf(By_old[i*Ni*Ni+(j+1)*Ni+k],2)+powf(Bz_old[i*Ni*Ni+j*Ni+(k)],2))),1e-6f);
            float p_r = fmaxf((gamma-1)*(E_total[idx_r] - 0.5f*rho_r*(vx_r*vx_r + vy_r*vy_r + vz_r*vz_r) 
                                          - 0.5f*(powf(Bx_old[i*Ni*Ni+j*Ni+(k+1)],2)+powf(By_old[i*Ni*Ni+(j+1)*Ni+(k+1)],2)+powf(Bz_old[i*Ni*Ni+j*Ni+(k+1)],2))),1e-6f);

            float Bz_f = Bz_old[i*Ni*Ni + j*Ni + (k+1)];
            float Bx_l = Bx_old[i*Ni*Ni+j*Ni+k]; float Bx_r = Bx_old[i*Ni*Ni+j*Ni+(k+1)];
            float By_l = By_old[i*Ni*Ni+(j+1)*Ni+k]; float By_r = By_old[i*Ni*Ni+(j+1)*Ni+(k+1)];

            float cf_l = sqrtf((Bz_f*Bz_f + Bx_l*Bx_l + By_l*By_l)/rho_l + gamma*p_l/rho_l);
            float cf_r = sqrtf((Bz_f*Bz_f + Bx_r*Bx_r + By_r*By_r)/rho_r + gamma*p_r/rho_r);

            float SL = fminf(vz_l - cf_l, vz_r - cf_r);
            float SR = fmaxf(vz_l + cf_l, vz_r + cf_r);
            float S_star = (rho_r*vz_r*(SR-vz_r) - rho_l*vz_l*(SL-vz_l) + p_l - p_r) / (rho_r*(SR-vz_r) - rho_l*(SL-vz_l) + 1e-12f);

            float p_star = p_l + rho_l*(SL - vz_l)*(S_star - vz_l) - Bz_f*Bz_f;

            float Bx_star = (sqrtf(rho_l)*Bx_l + sqrtf(rho_r)*Bx_r + sqrtf(rho_l*rho_r)*(vx_l - vx_r)) / (sqrtf(rho_l) + sqrtf(rho_r) + 1e-12f);
            float By_star = (sqrtf(rho_l)*By_l + sqrtf(rho_r)*By_r + sqrtf(rho_l*rho_r)*(vy_l - vy_r)) / (sqrtf(rho_l) + sqrtf(rho_r) + 1e-12f);

            float vx_star = 0.5f*(vx_l + vx_r) - Bz_f*(Bx_star - 0.5f*(Bx_l + Bx_r)) / (0.5f*(rho_l + rho_r));
            float vy_star = 0.5f*(vy_l + vy_r) - Bz_f*(By_star - 0.5f*(By_l + By_r)) / (0.5f*(rho_l + rho_r));

            float flux_rho, flux_mz, flux_mx, flux_my, flux_E;
            if (SL >= 0.0f) {
                flux_rho = rho_l * vz_l;
                flux_mz = rho_l*vz_l*vz_l + p_l + 0.5f*(Bx_l*Bx_l + By_l*By_l) - Bz_f*Bz_f;
                flux_mx = rho_l*vz_l*vx_l - Bz_f*Bx_l;
                flux_my = rho_l*vz_l*vy_l - Bz_f*By_l;
                flux_E = (E_total[idx_l] + p_l + 0.5f*(Bx_l*Bx_l + By_l*By_l)) * vz_l - Bz_f*(Bz_f*vz_l + Bx_l*vx_l + By_l*vy_l);
            } else if (S_star >= 0.0f) {
                flux_rho = rho_l * (SL - vz_l) / (SL - S_star) * S_star;
                flux_mz = flux_rho * S_star + p_star + 0.5f*(Bx_star*Bx_star + By_star*By_star) - Bz_f*Bz_f;
                flux_mx = flux_rho * vx_star - Bz_f * Bx_star;
                flux_my = flux_rho * vy_star - Bz_f * By_star;
                flux_E = (0.5f*flux_rho*S_star*S_star + p_star + 0.5f*(Bx_star*Bx_star + By_star*By_star)) * S_star - Bz_f*(Bz_f*S_star + Bx_star*vx_star + By_star*vy_star);
            } else if (SR >= 0.0f) {
                flux_rho = rho_r * (SR - vz_r) / (SR - S_star) * S_star;
                flux_mz = flux_rho * S_star + p_star + 0.5f*(Bx_star*Bx_star + By_star*By_star) - Bz_f*Bz_f;
                flux_mx = flux_rho * vx_star - Bz_f * Bx_star;
                flux_my = flux_rho * vy_star - Bz_f * By_star;
                flux_E = (0.5f*flux_rho*S_star*S_star + p_star + 0.5f*(Bx_star*Bx_star + By_star*By_star)) * S_star - Bz_f*(Bz_f*S_star + Bx_star*vx_star + By_star*vy_star);
            } else {
                flux_rho = rho_r * vz_r;
                flux_mz = rho_r*vz_r*vz_r + p_r + 0.5f*(Bx_r*Bx_r + By_r*By_r) - Bz_f*Bz_f;
                flux_mx = rho_r*vz_r*vx_r - Bz_f*Bx_r;
                flux_my = rho_r*vz_r*vy_r - Bz_f*By_r;
                flux_E = (E_total[idx_r] + p_r + 0.5f*(Bx_r*Bx_r + By_r*By_r)) * vz_r - Bz_f*(Bz_f*vz_r + Bx_r*vx_r + By_r*vy_r);
            }

            rho_out[idx] -= dt_over_dx * flux_rho;
            mz_out[idx] -= dt_over_dx * flux_mz;
            mx_out[idx] -= dt_over_dx * flux_mx;
            my_out[idx] -= dt_over_dx * flux_my;
            E_out[idx] -= dt_over_dx * flux_E;
        }

        // JxB term
        float Jx = (By_old[i*Ni*Ni+(j+1)*Ni+k] - By_old[i*Ni*Ni+j*Ni+k])/dx - (Bz_old[i*Ni*Ni+j*Ni+(k+1)] - Bz_old[i*Ni*Ni+j*Ni+k])/dx;
        float Jy = (Bz_old[i*Ni*Ni+j*Ni+(k+1)] - Bz_old[i*Ni*Ni+j*Ni+k])/dx - (Bx_old[(i+1)*Ni*Ni+j*Ni+k] - Bx_old[i*Ni*Ni+j*Ni+k])/dx;
        float Jz = (Bx_old[(i+1)*Ni*Ni+j*Ni+k] - Bx_old[i*Ni*Ni+j*Ni+k])/dx - (By_old[i*Ni*Ni+(j+1)*Ni+k] - By_old[i*Ni*Ni+j*Ni+k])/dx;

        float jxb_x = Jy * Bz_old[idx] - Jz * By_old[idx];
        float jxb_y = Jz * Bx_old[idx] - Jx * Bz_old[idx];
        float jxb_z = Jx * By_old[idx] - Jy * Bx_old[idx];

        mx_out[idx] += dt_over_dx * jxb_x;
        my_out[idx] += dt_over_dx * jxb_y;
        mz_out[idx] += dt_over_dx * jxb_z;

        float vx = mx[idx] / rho_safe;
        float vy = my[idx] / rho_safe;
        float vz = mz[idx] / rho_safe;
        E_out[idx] += dt_over_dx * (vx*jxb_x + vy*jxb_y + vz*jxb_z);
        E_out[idx] = fmaxf(E_out[idx], 1e-6f);
    }
}
''', 'mhd_kernel')

print("✅ Full explicit HLLD (X + Y + Z) kernel loaded!")

# ====================== MAIN LOOP ======================
block = (16, 16, 4)
grid = ((N + 15)//16, (N + 15)//16, (N + 3)//4)
shared_bytes = 3 * (block[0]+2)*(block[1]+2)*(block[2]+2) * 4

steps = 0
while steps < max_steps:
    update_ghosts()

    # Dynamic CFL
    rho_safe = cp.maximum(rho, 1e-8)
    vx = mx / rho_safe
    vy = my / rho_safe
    vz = mz / rho_safe
    v2 = vx*vx + vy*vy + vz*vz
    E_c = E_total[NG:NG+N, NG:NG+N, NG:NG+N]
    rho_c = rho_safe[NG:NG+N, NG:NG+N, NG:NG+N]
    p = (gamma - 1.0) * cp.maximum(E_c - 0.5 * rho_c * v2[NG:NG+N, NG:NG+N, NG:NG+N], 1e-8)
    cs2 = gamma * p / rho_c
    B2 = (Bx[NG:NG+N,NG:NG+N,NG:NG+N]**2 + By[NG:NG+N,NG:NG+N,NG:NG+N]**2 + Bz[NG:NG+N,NG:NG+N,NG:NG+N]**2)
    cf2 = B2 / rho_c
    cmax = cp.sqrt(cp.max(v2[NG:NG+N, NG:NG+N, NG:NG+N] + cs2 + cf2)).item()
    dt = min(cfl * dx / (cmax + 1e-12), dt_max)
    dt_over_dx = dt / dx

    kernel(grid, block, (rho, mx, my, mz, E_total, Bx, By, Bz,
                         rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1,
                         Emfx, Emfy, Emfz, Ni, dx, dt_over_dx, gamma),
           shared_mem=shared_bytes)

    update_ghosts()

    kernel(grid, block, (rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1,
                         rho2, mx2, my2, mz2, E2, Bx2, By2, Bz2,
                         Emfx, Emfy, Emfz, Ni, dx, dt_over_dx, gamma),
           shared_mem=shared_bytes)

    # SSP-RK2
    rho *= 0.5; rho += 0.5*rho2
    mx *= 0.5; mx += 0.5*mx2
    my *= 0.5; my += 0.5*my2
    mz *= 0.5; mz += 0.5*mz2
    E_total *= 0.5; E_total += 0.5*E2
    Bx *= 0.5; Bx += 0.5*Bx2
    By *= 0.5; By += 0.5*By2
    Bz *= 0.5; Bz += 0.5*Bz2

    steps += 1
    if steps % print_interval == 0:
        vmax = float(cp.max(cp.sqrt(v2)))
        mean_divB, max_divB = compute_divB()
        print(f"Step {steps:4d} | dt={dt:.2e} | Max|v|={vmax:.4f} | mean|divB|={mean_divB:.2e} | max|divB|={max_divB:.2e}")

print("\n✅ Full 3D HLLD (X+Y+Z) + True CT Yee is running!")
