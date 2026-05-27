import numpy as np
import cupy as cp

# ====================== PARAMETERS ======================
N = 128
L = 1.0
dx = L / N
dt = 0.00005
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
pert = 0.1
Bx[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
By[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
Bz[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert * 0.5 + 0.5, dtype=cp.float32)

def update_ghosts():
    # Cell-centered
    for f in [rho, mx, my, mz, E_total]:
        f[0:NG,:,:] = f[Ni-NG:Ni,:,:]
        f[Ni:,:,:] = f[NG:2*NG,:,:]
        f[:,0:NG,:] = f[:,Ni-NG:Ni,:]
        f[:,Ni:,:] = f[:,NG:2*NG,:]
        f[:,:,0:NG] = f[:,:,Ni-NG:Ni]
        f[:,:,Ni:] = f[:,:,NG:2*NG]
    # Staggered B
    for f in [Bx, By, Bz]:
        f[0:NG,:,:] = f[Ni-NG:Ni,:,:]
        f[Ni:,:,:] = f[NG:2*NG,:,:]
        f[:,0:NG,:] = f[:,Ni-NG:Ni,:]
        f[:,Ni:,:] = f[:,NG:2*NG,:]
        f[:,:,0:NG] = f[:,:,Ni-NG:Ni]
        f[:,:,Ni:] = f[:,:,NG:2*NG]

def compute_divB():
    divB = ((Bx[1:,:,:] - Bx[:-1,:,:]) + 
            (By[:,1:,:] - By[:,:-1,:]) + 
            (Bz[:,:,1:] - Bz[:,:,:-1])) / dx
    return float(cp.mean(cp.abs(divB))), float(cp.max(cp.abs(divB)))

# ====================== FULL KERNEL ======================
kernel = cp.RawKernel(r'''
extern "C" __global__ void full_muscl_minmod_3d_hlld(
    const float* rho, const float* mx, const float* my, const float* mz, const float* E_total,
    const float* Bx_old, const float* By_old, const float* Bz_old,
    float* rho_out, float* mx_out, float* my_out, float* mz_out, float* E_out,
    float* Bx_out, float* By_out, float* Bz_out,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, int NG, float dt_over_dx, float gamma)
{
    extern __shared__ float sdata[];
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx + NG;
    int j = blockIdx.y * blockDim.y + ty + NG;
    int k = blockIdx.z * blockDim.z + tz + NG;

    if (i >= Ni-NG || j >= Ni-NG || k >= Ni-NG) return;

    int txs = blockDim.x + 2; int tys = blockDim.y + 2; int tzs = blockDim.z + 2;
    int sidx = (tz + 1) * tys * txs + (ty + 1) * txs + (tx + 1);

    float* s_vx = sdata;
    float* s_vy = sdata + txs*tys*tzs;
    float* s_vz = sdata + 2*txs*tys*tzs;

    s_vx[sidx] = mx[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];
    s_vy[sidx] = my[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];
    s_vz[sidx] = mz[i*Ni*Ni + j*Ni + k] / rho[i*Ni*Ni + j*Ni + k];

    __syncthreads();

    // True CT Yee EMFs
    if (i < Ni-1 && j < Ni-1 && k < Ni) {
        float vx_e = 0.25f*(s_vx[sidx]+s_vx[sidx+1]+s_vx[sidx+txs]+s_vx[sidx+txs+1]);
        float vy_e = 0.25f*(s_vy[sidx]+s_vy[sidx+1]+s_vy[sidx+txs]+s_vy[sidx+txs+1]);
        float Bx_e = 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k] + Bx_old[(i+1)*Ni*Ni+(j+1)*Ni+k]);
        float By_e = 0.5f*(By_old[i*Ni*Ni+(j+1)*Ni+k] + By_old[(i+1)*Ni*Ni+(j+1)*Ni+k]);
        Emfz[i*Ni*Ni + j*Ni + k] = -(vx_e * By_e - vy_e * Bx_e);
    }
    if (i < Ni-1 && k < Ni-1) {
        float vx_e = 0.25f*(s_vx[sidx]+s_vx[sidx+1]+s_vx[sidx+txs*tys]+s_vx[sidx+txs*tys+1]);
        float vz_e = 0.25f*(s_vz[sidx]+s_vz[sidx+1]+s_vz[sidx+txs*tys]+s_vz[sidx+txs*tys+1]);
        float Bx_e = 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k] + Bx_old[(i+1)*Ni*Ni+j*Ni+(k+1)]);
        float Bz_e = 0.5f*(Bz_old[i*Ni*Ni+j*Ni+(k+1)] + Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)]);
        Emfy[i*Ni*Ni + j*Ni + k] = -(vz_e * Bx_e - vx_e * Bz_e);
    }
    if (j < Ni-1 && k < Ni-1) {
        float vy_e = 0.25f*(s_vy[sidx]+s_vy[sidx+txs]+s_vy[sidx+txs*tys]+s_vy[sidx+txs*tys+txs]);
        float vz_e = 0.25f*(s_vz[sidx]+s_vz[sidx+txs]+s_vz[sidx+txs*tys]+s_vz[sidx+txs*tys+txs]);
        float By_e = 0.5f*(By_old[i*Ni*Ni+(j+1)*Ni+k] + By_old[i*Ni*Ni+(j+1)*Ni+(k+1)]);
        float Bz_e = 0.5f*(Bz_old[i*Ni*Ni+j*Ni+(k+1)] + Bz_old[i*Ni*Ni+(j+1)*Ni+(k+1)]);
        Emfx[i*Ni*Ni + j*Ni + k] = -(vy_e * Bz_e - vz_e * By_e);
    }

    __syncthreads();

    // True CT Yee curl
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

    // MUSCL Minmod + HLLD (X direction)
    if (i < Ni && j < Ni && k < Ni) {
        int idx = i*Ni*Ni + j*Ni + k;
        int idx_l = (i-1)*Ni*Ni + j*Ni + k;
        int idx_r = (i+1)*Ni*Ni + j*Ni + k;

        // Minmod slopes
        float rho_l = rho[idx_l]; float rho_c = rho[idx]; float rho_r = rho[idx_r];
        float slope_rho = 0.5f * copysignf(fminf(fabs(rho_r - rho_c), fabs(rho_c - rho_l)), (rho_r - rho_l));

        float rhoL = rho_c - 0.5f * slope_rho;
        float rhoR = rho_c + 0.5f * slope_rho;

        // vx, vy, vz, p (simplified - same minmod)
        float vx_l = mx[idx_l] / rho_l; float vx_c = mx[idx] / rho_c; float vx_r = mx[idx_r] / rho_r;
        float slope_vx = 0.5f * copysignf(fminf(fabs(vx_r - vx_c), fabs(vx_c - vx_l)), (vx_r - vx_l));

        float p_l = fmaxf((gamma-1.0f)*(E_total[idx_l] - 0.5f*rho_l*(vx_l*vx_l + (my[idx_l]/rho_l)*(my[idx_l]/rho_l) + (mz[idx_l]/rho_l)*(mz[idx_l]/rho_l)) - 0.5f*(Bx_old[i*Ni*Ni+j*Ni+k]*Bx_old[i*Ni*Ni+j*Ni+k] + By_old[i*Ni*Ni+(j+1)*Ni+k]*By_old[i*Ni*Ni+(j+1)*Ni+k] + Bz_old[i*Ni*Ni+j*Ni+(k+1)]*Bz_old[i*Ni*Ni+j*Ni+(k+1)])), 1e-6f);
        float p_r = fmaxf((gamma-1.0f)*(E_total[idx_r] - 0.5f*rho_r*(vx_r*vx_r + (my[idx_r]/rho_r)*(my[idx_r]/rho_r) + (mz[idx_r]/rho_r)*(mz[idx_r]/rho_r)) - 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k]*Bx_old[(i+1)*Ni*Ni+j*Ni+k] + By_old[(i+1)*Ni*Ni+(j+1)*Ni+k]*By_old[(i+1)*Ni*Ni+(j+1)*Ni+k] + Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)]*Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)])), 1e-6f);

        float Bx_f = Bx_old[(i+1)*Ni*Ni + j*Ni + k];
        float By_l = By_old[i*Ni*Ni + (j+1)*Ni + k];
        float By_r = By_old[(i+1)*Ni*Ni + (j+1)*Ni + k];
        float Bz_l = Bz_old[i*Ni*Ni + j*Ni + (k+1)];
        float Bz_r = Bz_old[(i+1)*Ni*Ni + j*Ni + (k+1)];

        float cf_l = sqrtf((Bx_f*Bx_f + By_l*By_l + Bz_l*Bz_l)/rho_l + gamma*p_l/rho_l);
        float cf_r = sqrtf((Bx_f*Bx_f + By_r*By_r + Bz_r*Bz_r)/rho_r + gamma*p_r/rho_r);

        float SL = fminf(vx_l - cf_l, vx_r - cf_r);
        float SR = fmaxf(vx_l + cf_l, vx_r + cf_r);
        float S_star = (rho_r*vx_r*(SR-vx_r) - rho_l*vx_l*(SL-vx_l) + p_l - p_r) / (rho_r*(SR-vx_r) - rho_l*(SL-vx_l));

        float p_star = p_l + rho_l*(SL - vx_l)*(S_star - vx_l) - Bx_f*Bx_f;

        float rho_star_l = rho_l * (SL - vx_l) / (SL - S_star);
        float rho_star_r = rho_r * (SR - vx_r) / (SR - S_star);

        float By_star = (sqrt(rho_l)*By_l + sqrt(rho_r)*By_r + sqrt(rho_l*rho_r)*(vy_l - vy_r)) / (sqrt(rho_l) + sqrt(rho_r));
        float Bz_star = (sqrt(rho_l)*Bz_l + sqrt(rho_r)*Bz_r + sqrt(rho_l*rho_r)*(vz_l - vz_r)) / (sqrt(rho_l) + sqrt(rho_r));

        float vy_star = 0.5f*(vy_l + vy_r) - Bx_f*(By_star - 0.5f*(By_l + By_r)) / (0.5f*(rho_l + rho_r));
        float vz_star = 0.5f*(vz_l + vz_r) - Bx_f*(Bz_star - 0.5f*(Bz_l + Bz_r)) / (0.5f*(rho_l + rho_r));

        float flux_rho, flux_mx, flux_my, flux_mz, flux_E;
        if (SL > 0.0f) {
            flux_rho = rho_l * vx_l;
            flux_mx = rho_l*vx_l*vx_l + p_l + 0.5f*(By_l*By_l + Bz_l*Bz_l) - Bx_f*Bx_f;
            flux_my = rho_l*vx_l*vy_l - Bx_f*By_l;
            flux_mz = rho_l*vx_l*vz_l - Bx_f*Bz_l;
            flux_E = (E_total[idx] + p_l + 0.5f*(By_l*By_l + Bz_l*Bz_l)) * vx_l - Bx_f*(Bx_f*vx_l + By_l*vy_l + Bz_l*vz_l);
        } else if (S_star > 0.0f) {
            flux_rho = rho_star_l * S_star;
            flux_mx = rho_star_l*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star) - Bx_f*Bx_f;
            flux_my = rho_star_l*S_star*vy_star - Bx_f*By_star;
            flux_mz = rho_star_l*S_star*vz_star - Bx_f*Bz_star;
            flux_E = (0.5f*rho_star_l*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star)) * S_star - Bx_f*(Bx_f*S_star + By_star*vy_star + Bz_star*vz_star);
        } else if (SR > 0.0f) {
            flux_rho = rho_star_r * S_star;
            flux_mx = rho_star_r*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star) - Bx_f*Bx_f;
            flux_my = rho_star_r*S_star*vy_star - Bx_f*By_star;
            flux_mz = rho_star_r*S_star*vz_star - Bx_f*Bz_star;
            flux_E = (0.5f*rho_star_r*S_star*S_star + p_star + 0.5f*(By_star*By_star + Bz_star*Bz_star)) * S_star - Bx_f*(Bx_f*S_star + By_star*vy_star + Bz_star*vz_star);
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

        // JxB
        float Jx = (By_old[i*Ni*Ni+(j+1)*Ni+k] - By_old[i*Ni*Ni+j*Ni+k])/dx - (Bz_old[i*Ni*Ni+j*Ni+(k+1)] - Bz_old[i*Ni*Ni+j*Ni+k])/dx;
        float Jy = (Bz_old[i*Ni*Ni+j*Ni+(k+1)] - Bz_old[i*Ni*Ni+j*Ni+k])/dx - (Bx_old[(i+1)*Ni*Ni+j*Ni+k] - Bx_old[i*Ni*Ni+j*Ni+k])/dx;
        float Jz = (Bx_old[(i+1)*Ni*Ni+j*Ni+k] - Bx_old[i*Ni*Ni+j*Ni+k])/dx - (By_old[i*Ni*Ni+(j+1)*Ni+k] - By_old[i*Ni*Ni+j*Ni+k])/dx;

        float jxb_x = Jy * Bz_old[i*Ni*Ni+j*Ni+k] - Jz * By_old[i*Ni*Ni+j*Ni+k];
        float jxb_y = Jz * Bx_old[i*Ni*Ni+j*Ni+k] - Jx * Bz_old[i*Ni*Ni+j*Ni+k];
        float jxb_z = Jx * By_old[i*Ni*Ni+j*Ni+k] - Jy * Bx_old[i*Ni*Ni+j*Ni+k];

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
''', 'full_muscl_minmod_3d_hlld')

print("✅ Full complete code with Minmod + 3D HLLD ready!")

# ====================== SSP-RK2 LOOP ======================
block = (16, 16, 4)
grid = ((N + 15)//16, (N + 15)//16, (N + 3)//4)
shared_bytes = 3 * (block[0]+2)*(block[1]+2)*(block[2]+2) * 4

steps = 0
while steps < max_steps:
    update_ghosts()

    kernel(grid, block, (rho, mx, my, mz, E_total, Bx, By, Bz,
                         rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1,
                         Emfx, Emfy, Emfz, Ni, NG, dt/dx, gamma),
           shared_mem=shared_bytes)

    update_ghosts()

    kernel(grid, block, (rho1, mx1, my1, mz1, E1, Bx1, By1, Bz1,
                         rho2, mx2, my2, mz2, E2, Bx2, By2, Bz2,
                         Emfx, Emfy, Emfz, Ni, NG, dt/dx, gamma),
           shared_mem=shared_bytes)

    rho = 0.5 * (rho + rho2)
    mx = 0.5 * (mx + mx2)
    my = 0.5 * (my + my2)
    mz = 0.5 * (mz + mz2)
    E_total = 0.5 * (E_total + E2)
    Bx = 0.5 * (Bx + Bx2)
    By = 0.5 * (By + By2)
    Bz = 0.5 * (Bz + Bz2)

    steps += 1
    if steps % print_interval == 0:
        vmax = float(cp.max(cp.sqrt((mx/rho)**2 + (my/rho)**2 + (mz/rho)**2)))
        mean_divB, max_divB = compute_divB()
        print(f"Step {steps:4d} | Max|v| = {vmax:.4f} | mean|divB| = {mean_divB:.2e} | max|divB| = {max_divB:.2e}")

print("\n✅ Simulation running! Now test your JxB theory.")
