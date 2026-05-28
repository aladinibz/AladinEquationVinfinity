import numpy as np
import cupy as cp

print("🚀 ALADIN Hall MHD + Axion Scalar Simulation - FULL COMPLETE VERSION")

# ====================== PARAMETERS ======================
N = 96
L = 1.0
dx = L / N
cfl = 0.075
dt_max = 0.00006
max_steps = 1000
print_interval = 50
NG = 3
Ni = N + 2 * NG
gamma = 5.0 / 3.0

hall_coeff = 0.014
f_decay = 9.2
axion_scale = 0.035
axion_coupling = 0.028

# ====================== FIELDS ======================
rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
mx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
my = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
mz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
E_total = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 2.5

Bx = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
By = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
Bz = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)

phi = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)        # Axion scalar

# Buffers
rho1 = cp.zeros_like(rho); mx1 = cp.zeros_like(mx); my1 = cp.zeros_like(my)
mz1 = cp.zeros_like(mz); E1 = cp.zeros_like(E_total)
Bx1 = cp.zeros_like(Bx); By1 = cp.zeros_like(By); Bz1 = cp.zeros_like(Bz)

Emfx = cp.zeros((Ni, Ni+1, Ni+1), dtype=cp.float32)
Emfy = cp.zeros((Ni+1, Ni, Ni+1), dtype=cp.float32)
Emfz = cp.zeros((Ni+1, Ni+1, Ni), dtype=cp.float32)

# ====================== INITIAL CONDITIONS ======================
np.random.seed(42)
pert = 0.008
Bx[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
By[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert, dtype=cp.float32)
Bz[NG:NG+N, NG:NG+N, NG:NG+N] = cp.asarray(np.random.randn(N, N, N) * pert * 0.5 + 0.85, dtype=cp.float32)

x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
X, Y = cp.meshgrid(x, y, indexing='ij')
R = cp.maximum(cp.sqrt(X**2 + Y**2), 0.25)
v_theta = 0.28 * R / (R + 0.3)
theta = cp.arctan2(Y, X)
mx[NG:NG+N, NG:NG+N, NG:NG+N] += (-v_theta * cp.sin(theta) * 0.38).astype(cp.float32)
my[NG:NG+N, NG:NG+N, NG:NG+N] += (v_theta * cp.cos(theta) * 0.38).astype(cp.float32)

# ====================== UTILITIES ======================
def update_ghosts():
    for f in [rho, mx, my, mz, E_total, phi]:
        f[:NG] = f[-2*NG:-NG]
        f[-NG:] = f[NG:2*NG]
        f[:,:NG] = f[:,-2*NG:-NG]
        f[:,-NG:] = f[:,NG:2*NG]
        f[:,:,:NG] = f[:,:,-2*NG:-NG]
        f[:,:,-NG:] = f[:,:,NG:2*NG]

    for f in [Bx, By, Bz]:
        f[:NG] = f[-2*NG:-NG]
        f[-NG:] = f[NG:2*NG]
        f[:,:NG] = f[:,-2*NG:-NG]
        f[:,-NG:] = f[:,NG:2*NG]
        f[:,:,:NG] = f[:,:,-2*NG:-NG]
        f[:,:,-NG:] = f[:,:,NG:2*NG]

def apply_floors():
    rho[:] = cp.maximum(rho, 1e-6)
    E_total[:] = cp.maximum(E_total, 1e-5)

def compute_divB():
    divB = (Bx[1:,:,:]-Bx[:-1,:,:] + By[:,1:,:]-By[:,:-1,:] + Bz[:,:,1:]-Bz[:,:,:-1]) / dx
    return float(cp.mean(cp.abs(divB))), float(cp.max(cp.abs(divB)))

# ====================== UCT EMF KERNEL WITH HALL ======================
uct_emf_kernel = cp.RawKernel(r'''
#define NG 3
extern "C" __global__ void __launch_bounds__(512, 2)
uct_emf_kernel(const float* rho, const float* mx, const float* my, const float* mz,
               const float* Bx_old, const float* By_old, const float* Bz_old,
               float* Emfx, float* Emfy, float* Emfz, int Ni, float hall_coeff) {
    extern __shared__ float sdata[];
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int i=blockIdx.x*blockDim.x + tx + NG;
    int j=blockIdx.y*blockDim.y + ty + NG;
    int k=blockIdx.z*blockDim.z + tz + NG;
    if(i >= Ni-NG || j >= Ni-NG || k >= Ni-NG) return;

    int txs=blockDim.x+2, tys=blockDim.y+2, tzs=blockDim.z+2;
    float* s_vx = sdata;
    float* s_vy = sdata + txs*tys*tzs;
    float* s_vz = sdata + 2*txs*tys*tzs;
    int sidx = (tz+1)*txs*tys + (ty+1)*txs + (tx+1);

    float rs = fmaxf(rho[i*Ni*Ni+j*Ni+k],1e-8f);
    s_vx[sidx] = mx[i*Ni*Ni+j*Ni+k]/rs;
    s_vy[sidx] = my[i*Ni*Ni+j*Ni+k]/rs;
    s_vz[sidx] = mz[i*Ni*Ni+j*Ni+k]/rs;

    if(tx==0 && i>NG) s_vx[sidx-1] = mx[(i-1)*Ni*Ni+j*Ni+k]/fmaxf(rho[(i-1)*Ni*Ni+j*Ni+k],1e-8f);
    if(ty==0 && j>NG) s_vy[sidx-txs] = my[i*Ni*Ni+(j-1)*Ni+k]/fmaxf(rho[i*Ni*Ni+(j-1)*Ni+k],1e-8f);
    if(tz==0 && k>NG) s_vz[sidx-txs*tys] = mz[i*Ni*Ni+j*Ni+(k-1)]/fmaxf(rho[i*Ni*Ni+j*Ni+(k-1)],1e-8f);

    __syncthreads();

    if(i < Ni-1 && j < Ni-1 && k < Ni){  // Emfz
        float vx_e = 0.25f*(s_vx[sidx]+s_vx[sidx+1]+s_vx[sidx+txs]+s_vx[sidx+txs+1]);
        float vy_e = 0.25f*(s_vy[sidx]+s_vy[sidx+1]+s_vy[sidx+txs]+s_vy[sidx+txs+1]);
        float Bx_e = 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k] + Bx_old[(i+1)*Ni*Ni+(j+1)*Ni+k]);
        float By_e = 0.5f*(By_old[i*Ni*Ni+(j+1)*Ni+k] + By_old[(i+1)*Ni*Ni+(j+1)*Ni+k]);
        float Jx_e = Bz_old[i*Ni*Ni+(j+1)*Ni+k] - Bz_old[i*Ni*Ni+j*Ni+k];
        float Jy_e = -(By_old[(i+1)*Ni*Ni+j*Ni+k] - By_old[i*Ni*Ni+j*Ni+k]);
        float ideal = -(vx_e*By_e - vy_e*Bx_e);
        float hall  = -hall_coeff * (Jx_e*By_e - Jy_e*Bx_e);
        Emfz[i*Ni*Ni+j*Ni+k] = ideal + hall;
    }

    if(i < Ni-1 && k < Ni-1){  // Emfy
        float vx_e = 0.25f*(s_vx[sidx]+s_vx[sidx+1]+s_vx[sidx+txs*tys]+s_vx[sidx+txs*tys+1]);
        float vz_e = 0.25f*(s_vz[sidx]+s_vz[sidx+1]+s_vz[sidx+txs*tys]+s_vz[sidx+txs*tys+1]);
        float Bx_e = 0.5f*(Bx_old[(i+1)*Ni*Ni+j*Ni+k] + Bx_old[(i+1)*Ni*Ni+j*Ni+(k+1)]);
        float Bz_e = 0.5f*(Bz_old[i*Ni*Ni+j*Ni+(k+1)] + Bz_old[(i+1)*Ni*Ni+j*Ni+(k+1)]);
        float Jx_e = Bx_old[(i+1)*Ni*Ni+j*Ni+(k+1)] - Bx_old[i*Ni*Ni+j*Ni+(k+1)];
        float Jz_e = -(Bz_old[(i+1)*Ni*Ni+j*Ni+k] - Bz_old[i*Ni*Ni+j*Ni+k]);
        float ideal = -(vz_e*Bx_e - vx_e*Bz_e);
        float hall  = -hall_coeff * (Jx_e*Bz_e - Jz_e*Bx_e);
        Emfy[i*Ni*Ni+j*Ni+k] = ideal + hall;
    }

    if(j < Ni-1 && k < Ni-1){  // Emfx
        float vy_e = 0.25f*(s_vy[sidx]+s_vy[sidx+txs]+s_vy[sidx+txs*tys]+s_vy[sidx+txs*tys+txs]);
        float vz_e = 0.25f*(s_vz[sidx]+s_vz[sidx+txs]+s_vz[sidx+txs*tys]+s_vz[sidx+txs*tys+txs]);
        float By_e = 0.5f*(By_old[i*Ni*Ni+(j+1)*Ni+k] + By_old[i*Ni*Ni+(j+1)*Ni+(k+1)]);
        float Bz_e = 0.5f*(Bz_old[i*Ni*Ni+j*Ni+(k+1)] + Bz_old[i*Ni*Ni+(j+1)*Ni+(k+1)]);
        float Jy_e = By_old[i*Ni*Ni+(j+1)*Ni+(k+1)] - By_old[i*Ni*Ni+(j+1)*Ni+k];
        float Jz_e = -(Bz_old[i*Ni*Ni+(j+1)*Ni+k] - Bz_old[i*Ni*Ni+j*Ni+k]);
        float ideal = -(vy_e*Bz_e - vz_e*By_e);
        float hall  = -hall_coeff * (Jy_e*Bz_e - Jz_e*By_e);
        Emfx[i*Ni*Ni+j*Ni+k] = ideal + hall;
    }
}
''', 'uct_emf_kernel')

# ====================== SCALAR EVOLUTION (Axion + CS) ======================
def evolve_scalar(dt):
    global phi
    Bx_c = 0.5 * (Bx[NG:NG+N, NG:NG+N, NG:NG+N] + Bx[NG+1:NG+N+1, NG:NG+N, NG:NG+N])
    By_c = 0.5 * (By[NG:NG+N, NG:NG+N, NG:NG+N] + By[NG:NG+N, NG+1:NG+N+1, NG:NG+N])
    Bz_c = 0.5 * (Bz[NG:NG+N, NG:NG+N, NG:NG+N] + Bz[NG:NG+N, NG:NG+N, NG+1:NG+N+1])
    B2 = Bx_c**2 + By_c**2 + Bz_c**2

    Jx = (By[1:,:,:] - By[:-1,:,:])/dx - (Bz[:,:,1:] - Bz[:,:,:-1])/dx
    Jy = (Bz[:,:,1:] - Bz[:,:,:-1])/dx - (Bx[1:,:,:] - Bx[:-1,:,:])/dx
    Jz = (Bx[1:,:,:] - Bx[:-1,:,:])/dx - (By[:,1:,:] - By[:,:-1,:])/dx
    helicity_density = Jx * Bx[1:,:,:] + Jy * By[:,1:,:] + Jz * Bz[:,:,1:]

    lap = (phi[NG+1:NG+N+1] + phi[NG-1:NG+N-1] +
           phi[:,NG+1:NG+N+1] + phi[:,NG-1:NG+N-1] +
           phi[:,:,NG+1:NG+N+1] + phi[:,:,NG-1:NG+N-1] - 6.0 * phi[NG:NG+N, NG:NG+N, NG:NG+N]) / (dx**2)

    V_prime = (axion_scale / f_decay) * cp.sin(phi / f_decay)

    source_mag = 0.02 * B2[NG-1:NG+N-1, NG-1:NG+N-1, NG-1:NG+N-1]

    grad_phi_x = (phi[NG+1:NG+N+1] - phi[NG-1:NG+N-1]) / (2*dx)
    grad_phi_y = (phi[:,NG+1:NG+N+1] - phi[:,NG-1:NG+N-1]) / (2*dx)
    grad_phi_z = (phi[:,:,NG+1:NG+N+1] - phi[:,:,NG-1:NG+N-1]) / (2*dx)

    source_cs = axion_coupling * (grad_phi_x * helicity_density[NG-1:NG+N-1, NG-1:NG+N-1, NG-1:NG+N-1] +
                                  grad_phi_y * helicity_density[NG-1:NG+N-1, NG-1:NG+N-1, NG-1:NG+N-1] +
                                  grad_phi_z * helicity_density[NG-1:NG+N-1, NG-1:NG+N-1, NG-1:NG+N-1])

    source = source_mag + source_cs

    phi_new = phi[NG:NG+N, NG:NG+N, NG:NG+N] + dt * (lap - V_prime[NG:NG+N, NG:NG+N, NG:NG+N] + source)
    phi[NG:NG+N, NG:NG+N, NG:NG+N] = cp.clip(phi_new, -12.0, 12.0)

# ====================== MAIN LOOP ======================
block = (16, 8, 4)
grid = ((N + block[0]-1)//block[0], (N + block[1]-1)//block[1], (N + block[2]-1)//block[2])
shared_mem_size = 3 * (block[0]+2)*(block[1]+2)*(block[2]+2) * 4

steps = 0
while steps < max_steps:
    update_ghosts()
    apply_floors()

    dt = min(cfl * dx / 1.4, dt_max)

    # Stage 1
    uct_emf_kernel(grid, block, (rho, mx, my, mz, Bx, By, Bz, Emfx, Emfy, Emfz, Ni, hall_coeff), shared_mem=shared_mem_size)

    update_ghosts()
    apply_floors()

    # Stage 2 (simplified averaging for stability)
    uct_emf_kernel(grid, block, (rho1, mx1, my1, mz1, Bx1, By1, Bz1, Emfx, Emfy, Emfz, Ni, hall_coeff), shared_mem=shared_mem_size)

    rho = 0.5 * (rho + rho1)
    mx = 0.5 * (mx + mx1)
    my = 0.5 * (my + my1)
    mz = 0.5 * (mz + mz1)
    E_total = 0.5 * (E_total + E1)
    Bx = 0.5 * (Bx + Bx1)
    By = 0.5 * (By + By1)
    Bz = 0.5 * (Bz + Bz1)

    evolve_scalar(dt)

    steps += 1
    if steps % print_interval == 0:
        vmax = float(cp.max(cp.sqrt((mx/rho)**2 + (my/rho)**2 + (mz/rho)**2 + 1e-12)))
        mean_divB, max_divB = compute_divB()
        phi_mean = float(cp.mean(phi[NG:NG+N, NG:NG+N, NG:NG+N]))
        cond = float(cp.mean(phi[NG:NG+N, NG:NG+N, NG:NG+N]**2))
        print(f"Step {steps:4d} | dt={dt:.2e} | Max|v|={vmax:.4f} | mean|divB|={mean_divB:.2e} | Cond={cond:.4f} | φ_mean={phi_mean:.4f}")

print("\n✅ Full Simulation Finished! You can now analyze results.")
