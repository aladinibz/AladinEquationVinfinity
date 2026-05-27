import numpy as np
import cupy as cp

# ====================== PARAMETERS ======================
N = 128
L = 1.0
dx = L / N
dt = 0.00003
max_steps = 1200
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
        f[0:NG] = f[Ni-NG:Ni]; f[Ni:] = f[NG:2*NG]
        f[:,0:NG] = f[:,Ni-NG:Ni]; f[:,Ni:] = f[:,NG:2*NG]
        f[:,:,0:NG] = f[:,:,Ni-NG:Ni]; f[:,:,Ni:] = f[:,:,NG:2*NG]
    for f in [Bx, By, Bz]:
        f[0:NG] = f[Ni-NG:Ni]; f[Ni:] = f[NG:2*NG]
        f[:,0:NG] = f[:,Ni-NG:Ni]; f[:,Ni:] = f[:,NG:2*NG]
        f[:,:,0:NG] = f[:,:,Ni-NG:Ni]; f[:,:,Ni:] = f[:,:,NG:2*NG]

def compute_divB():
    divB = ((Bx[1:,:,:] - Bx[:-1,:,:]) + 
            (By[:,1:,:] - By[:,:-1,:]) + 
            (Bz[:,:,1:] - Bz[:,:,:-1])) / dx
    return float(cp.mean(cp.abs(divB))), float(cp.max(cp.abs(divB)))

# ====================== FULL KERNEL WITH COMPLETE HLLD ======================
kernel = cp.RawKernel(r'''
extern "C" __global__ void full_hlld_ct_yee_kernel(
    const float* rho, const float* mx, const float* my, const float* mz, const float* E_total,
    const float* Bx_old, const float* By_old, const float* Bz_old,
    float* rho_out, float* mx_out, float* my_out, float* mz_out, float* E_out,
    float* Bx_out, float* By_out, float* Bz_out,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, float dt_over_dx, float gamma)
{
    extern __shared__ float sdata[];
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx + 3;
    int j = blockIdx.y * blockDim.y + ty + 3;
    int k = blockIdx.z * blockDim.z + tz + 3;

    if (i >= Ni-3 || j >= Ni-3 || k >= Ni-3) return;

    int txs = blockDim.x + 2; int tys = blockDim.y + 2; int tzs = blockDim.z + 2;
    int sidx = (tz + 1) * tys * txs + (ty + 1) * txs + (tx + 1);

    float* s_vx = sdata;
    float* s_vy = sdata + txs*tys*tzs;
    float* s_vz = sdata + 2*txs*tys*tzs;

    int idx = i*Ni*Ni + j*Ni + k;
    s_vx[sidx] = mx[idx] / rho[idx];
    s_vy[sidx] = my[idx] / rho[idx];
    s_vz[sidx] = mz[idx] / rho[idx];

    if (tx == 0) s_vx[sidx-1] = mx[(i-1)*Ni*Ni + j*Ni + k] / rho[(i-1)*Ni*Ni + j*Ni + k];
    if (ty == 0) s_vy[sidx-txs] = my[i*Ni*Ni + (j-1)*Ni + k] / rho[i*Ni*Ni + (j-1)*Ni + k];
    if (tz == 0) s_vz[sidx-tys*txs] = mz[i*Ni*Ni + j*Ni + (k-1)] / rho[i*Ni*Ni + j*Ni + (k-1)];

    __syncthreads();

    // True CT Yee EMFs
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

    // Strict True CT Yee curl
    if (i < Ni && j < Ni && k < Ni) {
        float dEz_dy = Emfz[i*Ni*Ni+(j+1)*Ni+k] - Emfz[i*Ni*Ni+j*Ni+k];
        float dEy_dz = Emfy[i*Ni*Ni+j*Ni+(k+1)] - Emfy[i*Ni*Ni+j*Ni+k];
        Bx_out[i*Ni*Ni+j*Ni+k] = Bx_old[i*Ni*Ni+j*Ni+k] - dt_over_dx*(dEz_dy - dEy_dz);

        float dEx_dz = Emfx[i*Ni*Ni+j*Ni+(k+1)] - Emfx[i*Ni*Ni+j*Ni+k];
        float dEz_dx = Emfz[(i+1)*Ni*Ni+j*Ni+k] - Emfz[i*Ni*Ni+j*Ni+k];
        By_out[i*Ni*Ni+j*Ni+k] = By_old[i*Ni*Ni+j*Ni+k] - dt_over_dx*(dEx_dz - dEz_dx);

        float dEy_dx = Emfy[(i+1)*Ni*Ni+j*Ni+k] - Emfy[i*Ni*Ni+j*Ni+k];
        float dEx_dy = Emfx[i*Ni*Ni+(j+1)*Ni+k] - Emfx[i*Ni*Ni+j*Ni+k];
        Bz_out[i*Ni*Ni+j*Ni+k] = Bz_old[i*Ni*Ni+j*Ni+k] - dt_over_dx*(dEy_dx - dEx_dy);
    }

    // ================ FULL HLLD RIEMANN SOLVER (X-direction) ================
    if (i < Ni && j < Ni && k < Ni) {
        int idx = i*Ni*Ni + j*Ni + k;
        int idx_l = (i-1)*Ni*Ni + j*Ni + k;
        int idx_r = (i+1)*Ni*Ni + j*Ni + k;

        // Component-wise Minmod
        float rho_l = rho[idx_l], rho_c = rho[idx], rho_r = rho[idx_r];
        float dr = 0.5f * copysignf(fminf(fabs(rho_r-rho_c), fabs(rho_c-rho_l)), (rho_r-rho_l));

        float vx_l = mx[idx_l]/rho_l, vx_c = mx[idx]/rho_c, vx_r = mx[idx_r]/rho_r;
        float dvx = 0.5f * copysignf(fminf(fabs(vx_r-vx_c), fabs(vx_c-vx_l)), (vx_r-vx_l));

        float p_l = fmaxf((gamma-1.0f)*(E_total[idx_l] - 0.5f*rho_l*(vx_l*vx_l + powf(my[idx_l]/rho_l,2) + powf(mz[idx_l]/rho_l,2)) - 0.5f*(powf(Bx_old[i*Ni*Ni+j*Ni+k],2) + powf(By_old[i*Ni*Ni+(j+1)*Ni+k],2) + powf(Bz_old[i*Ni*Ni+j*Ni+(k+1)],2))), 1e-6f);
        float p_r = fmaxf((gamma-1.0f)*(E_total[idx_r] - 0.5f*rho_r*(vx_r*vx_r + powf(my[idx_r]/rho_r,2) + powf(mz[idx_r]/rho_r,2)) - 0.5f*(powf(Bx_old[(i+1)*Ni*Ni+j*Ni+k],2) + powf(By_old[(i+1)*Ni*Ni+(j+1)*Ni+k],2) + powf(Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)],2))), 1e-6f);

        float rhoL = rho_c - 0.5f*dr;
        float rhoR = rho_c + 0.5f*dr;
        float vxL = vx_c - 0.5f*dvx;
        float vxR = vx_c + 0.5f*dvx;
        float pL = p_l;
        float pR = p_r;

        float Bx_f = Bx_old[(i+1)*Ni*Ni + j*Ni + k];
        float ByL = By_old[i*Ni*Ni + (j+1)*Ni + k];
        float ByR = By_old[(i+1)*Ni*Ni + (j+1)*Ni + k];
        float BzL = Bz_old[i*Ni*Ni + j*Ni + (k+1)];
        float BzR = Bz_old[(i+1)*Ni*Ni + j*Ni + (k+1)];

        // HLLD wave speeds
        float cfL = sqrtf((Bx_f*Bx_f + ByL*ByL + BzL*BzL)/rhoL + gamma*pL/rhoL);
        float cfR = sqrtf((Bx_f*Bx_f + ByR*ByR + BzR*BzR)/rhoR + gamma*pR/rhoR);

        float SL = fminf(vxL - cfL, vxR - cfR);
        float SR = fmaxf(vxL + cfL, vxR + cfR);
        float S_star = (rhoR*vxR*(SR - vxR) - rhoL*vxL*(SL - vxL) + pL - pR) / (rhoR*(SR - vxR) - rhoL*(SL - vxL));

        float p_star = pL + rhoL*(SL - vxL)*(S_star - vxL) - Bx_f*Bx_f;

        float rho_star_l = rhoL * (SL - vxL) / (SL - S_star);
        float rho_star_r = rhoR * (SR - vxR) / (SR - S_star);

        float By_star = (sqrtf(rhoL)*ByL + sqrtf(rhoR)*ByR + sqrtf(rhoL*rhoR)*( (my[idx_l]/rho_l) - (my[idx_r]/rho_r) )) / (sqrtf(rhoL) + sqrtf(rhoR));
        float Bz_star = (sqrtf(rhoL)*BzL + sqrtf(rhoR)*BzR + sqrtf(rhoL*rhoR)*( (mz[idx_l]/rho_l) - (mz[idx_r]/rho_r) )) / (sqrtf(rhoL) + sqrtf(rhoR));

        float vy_star = 0.5f*((my[idx_l]/rho_l) + (my[idx_r]/rho_r)) - Bx_f*(By_star - 0.5f*(ByL + ByR)) / (0.5f*(rhoL + rhoR));
        float vz_star = 0.5f*((mz[idx_l]/rho_l) + (mz[idx_r]/rho_r)) - Bx_f*(Bz_star - 0.5f*(BzL + BzR)) / (0.5f*(rhoL + rhoR));

        // Final flux
        float flux_rho, flux_mx, flux_my, flux_mz, flux_E;
        if (SL >= 0.0f) {
            flux_rho = rhoL * vxL;
            flux_mx = rhoL*vxL*vxL + pL + 0.5f*(ByL*ByL + BzL*BzL) - Bx_f*Bx_f;
            flux_my = rhoL*vxL*(my[idx_l]/rho_l) - Bx_f*ByL;
            flux_mz = rhoL*vxL*(mz[idx_l]/rho_l) - Bx_f*BzL;
            flux_E = (E_total[idx_l] + pL + 0.5f*(ByL*ByL + BzL*BzL)) * vxL - Bx_f*(Bx_f*vxL + ByL*(my[idx_l]/rho_l) + BzL*(mz[idx_l]/rho_l));
        } else if (S_star >= 0.0f) {
            flux_rho = rho_star_l * S_star;
            flux_mx = rho_star_l*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star) - Bx_f*Bx_f;
            flux_my = rho_star_l*S_star*vy_star - Bx_f*By_star;
            flux_mz = rho_star_l*S_star*vz_star - Bx_f*Bz_star;
            flux_E = (0.5f*rho_star_l*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star)) * S_star - Bx_f*(Bx_f*S_star + By_star*vy_star + Bz_star*vz_star);
        } else if (SR >= 0.0f) {
            flux_rho = rho_star_r * S_star;
            flux_mx = rho_star_r*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star) - Bx_f*Bx_f;
            flux_my = rho_star_r*S_star*vy_star - Bx_f*By_star;
            flux_mz = rho_star_r*S_star*vz_star - Bx_f*Bz_star;
            flux_E = (0.5f*rho_star_r*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star)) * S_star - Bx_f*(Bx_f*S_star + By_star*vy_star + Bz_star*vz_star);
        } else {
            flux_rho = rhoR * vxR;
            flux_mx = rhoR*vxR*vxR + pR + 0.5f*(ByR*ByR + BzR*BzR) - Bx_f*Bx_f;
            flux_my = rhoR*vxR*(my[idx_r]/rho_r) - Bx_f*ByR;
            flux_mz = rhoR*vxR*(mz[idx_r]/rho_r) - Bx_f*BzR;
            flux_E = (E_total[idx_r] + pR + 0.5f*(ByR*ByR + BzR*BzR)) * vxR - Bx_f*(Bx_f*vxR + ByR*(my[idx_r]/rho_r) + BzR*(mz[idx_r]/rho_r));
        }

        rho_out[idx] = rho[idx] - dt_over_dx * flux_rho;
        mx_out[idx] = mx[idx] - dt_over_dx * flux_mx;
        my_out[idx] = my[idx] - dt_over_dx * flux_my;
        mz_out[idx] = mz[idx] - dt_over_dx * flux_mz;
        E_out[idx] = E_total[idx] - dt_over_dx * flux_E;

        // JxB Lorentz force
        float Jx = (By_old[i*Ni*Ni+(j+1)*Ni+k] - By_old[i*Ni*Ni+j*Ni+k])/dx - (Bz_old[i*Ni*Ni+j*Ni+(k+1)] - Bz_old[i*Ni*Ni+j*Ni+k])/dx;
        float Jy = (Bz_old[i*Ni*Ni+j*Ni+(k+1)] - Bz_old[i*Ni*Ni+j*Ni+k])/dx - (Bx_old[(i+1)*Ni*Ni+j*Ni+k] - Bx_old[i*Ni*Ni+j*Ni+k])/dx;
        float Jz = (Bx_old[(i+1)*Ni*Ni+j*Ni+k] - Bx_old[i*Ni*Ni+j*Ni+k])/dx - (By_old[i*Ni*Ni+(j+1)*Ni+k] - By_old[i*Ni*Ni+j*Ni+k])/dx;

        float jxb_x = Jy * Bz_old[idx] - Jz * By_old[idx];
        float jxb_y = Jz * Bx_old[idx] - Jx * Bz_old[idx];
        float jxb_z = Jx * By_old[idx] - Jy * Bx_old[idx];

        mx_out[idx] += dt_over_dx * jxb_x;
        my_out[idx] += dt_over_dx * jxb_y;
        mz_out[idx] += dt_over_dx * jxb_z;

        float vx = mx[idx] / rho[idx];
        float vy = my[idx] / rho[idx];
        float vz = mz[idx] / rho[idx];
        float work = vx*jxb_x + vy*jxb_y + vz*jxb_z;
        E_out[idx] += dt_over_dx * work;
    }
}
''', 'full_hlld_ct_yee_kernel')

print("✅ Full HLLD + True CT Yee kernel ready!")

# ====================== LAUNCH ======================
block = (16, 16, 4)
grid = ((N + 15)//16, (N + 15)//16, (N + 3)//4)
shared_bytes = 3 * (block[0]+2)*(block[1]+2)*(block[2]+2) * 4

steps = 0
while steps < max_steps:
    update_ghosts()

    kernel(grid, block, (rho, mx, my, mz, E_total, Bx, By, Bz,
                         rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1,
                         Emfx, Emfy, Emfz, Ni, dt/dx, gamma),
           shared_mem=shared_bytes)

    update_ghosts()

    kernel(grid, block, (rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1,
                         rho2, mx2, my2, mz2, E2, Bx2, By2, Bz2,
                         Emfx, Emfy, Emfz, Ni, dt/dx, gamma),
           shared_mem=shared_bytes)

    # In-place SSP-RK2
    rho *= 0.5; rho += 0.5 * rho2
    mx *= 0.5; mx += 0.5 * mx2
    my *= 0.5; my += 0.5 * my2
    mz *= 0.5; mz += 0.5 * mz2
    E_total *= 0.5; E_total += 0.5 * E2
    Bx *= 0.5; Bx += 0.5 * Bx2
    By *= 0.5; By += 0.5 * By2
    Bz *= 0.5; Bz += 0.5 * Bz2

    steps += 1
    if steps % print_interval == 0:
        vmax = float(cp.max(cp.sqrt((mx/rho)**2 + (my/rho)**2 + (mz/rho)**2)))
        mean_divB, max_divB = compute_divB()
        print(f"Step {steps:4d} | Max|v| = {vmax:.4f} | mean|divB| = {mean_divB:.2e} | max|divB| = {max_divB:.2e}")

print("\n✅ Full HLLD implementation running! Test your theory now.")
