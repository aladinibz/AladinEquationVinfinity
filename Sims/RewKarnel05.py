import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v43.1 — FIXED SHAPES + Char Recon + JxB + Pressure")

# ====================== PARAMETERS ======================
N = 64
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
gamma = 5.0 / 3.0
CFL = 0.12
steps = 400
p_floor = 1e-4

v_phi_factor = 0.12
c_h_factor = 12.0
kappa_factor = 15.0

# ====================== NFW ======================
M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== MINMOD ======================
def minmod(a, b):
    return cp.sign(a) * cp.minimum(cp.abs(a), cp.abs(b)) * (cp.sign(a) == cp.sign(b))

# ====================== PLM RECONSTRUCTION ======================
def reconstruct_plm(q, axis=0):
    dq_right = cp.roll(q, -1, axis=axis) - q
    dq_left = q - cp.roll(q, 1, axis=axis)
    slope = minmod(dq_left, dq_right)
    qL = q + 0.5 * slope
    qR = cp.roll(q, -1, axis=axis) - 0.5 * cp.roll(slope, -1, axis=axis)
    return qL, qR

# ====================== KERNEL (Upwind CT) ======================
kernel_code = r'''
extern "C" __global__
void ct_emf_kernel(float* vx, float* vy, float* vz,
                   float* Bx, float* By, float* Bz,
                   float dt, float dx, int N)
{
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    if (i < N-1 && j < N-1 && k < N-1) {
        float vx_avg = 0.25f * (vx[(i*N+j)*N+k] + vx[(i*N+j+1)*N+k] + vx[(i*N+j)*N+k+1] + vx[(i*N+j+1)*N+k+1]);
        float vy_avg = 0.25f * (vy[(i*N+j)*N+k] + vy[(i*N+j+1)*N+k] + vy[(i*N+j)*N+k+1] + vy[(i*N+j+1)*N+k+1]);
        float Bx_avg = 0.25f * (Bx[i*N+j] + Bx[(i+1)*N+j] + Bx[i*N+j+1] + Bx[(i+1)*N+j+1]);
        float By_avg = 0.25f * (By[i*N+j] + By[(i+1)*N+j] + By[i*N+j+1] + By[(i+1)*N+j+1]);

        float sign_v = (vx_avg * vy_avg > 0.0f) ? 1.0f : -1.0f;
        float Ez_val = - (vx_avg * By_avg - vy_avg * Bx_avg) * (1.0f + 0.08f * sign_v);

        int bz_idx = (i * N + j) * (N + 1) + k;
        Bz[bz_idx] += (dt / dx) * Ez_val;
    }
}
'''

kernel = cp.RawKernel(kernel_code, 'ct_emf_kernel')

# ====================== FIELDS ======================
rho = cp.ones((N, N, N), dtype=cp.float32) * 1e-3
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)
psi = cp.zeros((N, N, N), dtype=cp.float32)

r_cyl = cp.sqrt(X**2 + Y**2)
rho *= cp.exp(-r_cyl / 8.0) * cp.exp(-Z**2 / 2.25)

r3d = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
M_dm = nfw_enclosed_mass(r3d)
g_r = -G * M_dm / r3d**2
v_phi = v_phi_factor * cp.sqrt(cp.maximum(r_cyl * cp.abs(g_r), 0.0))

vx = -v_phi * (Y / (r_cyl + 1e-8))
vy =  v_phi * (X / (r_cyl + 1e-8))
vz = cp.zeros_like(vx)

mx = rho * vx
my = rho * vy
mz = rho * vz

B0 = 1.5
Bphi = 0.8
Bx[1:,:,:] = -Bphi * (Y[0:N,:,:] / (r_cyl[0:N,:,:] + 1e-8))
By[:,1:,:] =  Bphi * (X[:,0:N,:] / (r_cyl[:,0:N,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,0:N]**2 + Y[:,:,0:N]**2 + Z[:,:,0:N]**2) / 250.0)

c_h = c_h_factor * 120.0
kappa = kappa_factor / dx

def compute_divB():
    div = cp.zeros((N, N, N), dtype=cp.float32)
    div += (Bx[1:,:,:] - Bx[:-1,:,:]) / dx
    div += (By[:,1:,:] - By[:,:-1,:]) / dx
    div += (Bz[:,:,1:] - Bz[:,:,:-1]) / dx
    return div

def compute_staggered_JxB():
    Bx_c = 0.5 * (Bx[1:,:,:] + Bx[:-1,:,:])
    By_c = 0.5 * (By[:,1:,:] + By[:,:-1,:])
    Bz_c = 0.5 * (Bz[:,:,1:] + Bz[:,:,:-1])

    Jx = (cp.roll(Bz_c, -1, axis=1) - cp.roll(Bz_c, 1, axis=1)) / (2*dx) - (cp.roll(By_c, -1, axis=2) - cp.roll(By_c, 1, axis=2)) / (2*dx)
    Jy = (cp.roll(Bx_c, -1, axis=2) - cp.roll(Bx_c, 1, axis=2)) / (2*dx) - (cp.roll(Bz_c, -1, axis=0) - cp.roll(Bz_c, 1, axis=0)) / (2*dx)
    Jz = (cp.roll(By_c, -1, axis=0) - cp.roll(By_c, 1, axis=0)) / (2*dx) - (cp.roll(Bx_c, -1, axis=1) - cp.roll(Bx_c, 1, axis=1)) / (2*dx)

    fx = Jy * Bz_c - Jz * By_c
    fy = Jz * Bx_c - Jx * Bz_c
    fz = Jx * By_c - Jy * Bx_c
    return fx, fy, fz

def compute_pressure_gradient(p):
    px = (p[1:,:,:] - p[:-1,:,:]) / dx
    py = (p[:,1:,:] - p[:,:-1,:]) / dx
    pz = (p[:,:,1:] - p[:,:,:-1]) / dx
    # Trim to (N,N,N)
    return -px[:-1,:,:], -py[:,:-1,:], -pz[:,:,:-1]

print("Starting v43.1 with Fixed Shapes...")

block = (8, 8, 8)
grid = ((N + 7)//8, (N + 7)//8, (N + 7)//8)

for step in range(steps):
    dt = CFL * dx / 220.0

    # CT EMF kernel
    kernel(grid, block, (vx, vy, vz, Bx, By, Bz, dt, dx, N), shared_mem=6*1000*4)

    # Dedner
    divB = compute_divB()
    psi -= dt * c_h**2 * divB - dt * kappa * psi
    psi_x = (psi[1:,:,:] - psi[:-1,:,:]) / dx
    psi_y = (psi[:,1:,:] - psi[:,:-1,:]) / dx
    psi_z = (psi[:,:,1:] - psi[:,:,:-1]) / dx
    Bx[1:-1,:,:] -= dt * psi_x
    By[:,1:-1,:] -= dt * psi_y
    Bz[:,:,1:-1] -= dt * psi_z

    # Pressure gradient (fixed shape)
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - 0.5*rho*(vx**2 + vy**2 + vz**2) - 0.5*(0.25*(Bx[1:,:,:]**2 + Bx[:-1,:,:]**2) + 0.25*(By[:,1:,:]**2 + By[:,:-1,:]**2) + 0.25*(Bz[:,:,1:]**2 + Bz[:,:,:-1]**2))), p_floor)
    px, py, pz = compute_pressure_gradient(p_thermal)
    mx += dt * px
    my += dt * py
    mz += dt * pz

    # J×B (fixed shape)
    Jx, Jy, Jz = compute_staggered_JxB()
    mx += dt * Jx
    my += dt * Jy
    mz += dt * Jz

    # Update velocity
    vx = mx / rho
    vy = my / rho
    vz = mz / rho

    if step % 50 == 0:
        div_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bx_c = 0.5 * (Bx[1:,:,:] + Bx[:-1,:,:])
        By_c = 0.5 * (By[:,1:,:] + By[:,:-1,:])
        Bz_c = 0.5 * (Bz[:,:,1:] + Bz[:,:,:-1])
        Bmag = cp.sqrt(Bx_c**2 + By_c**2 + Bz_c**2)
        Bmax = float(cp.nanmax(Bmag))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB_max = {div_max:.2e}")

print("\n✅ v43.1 Finished with Fixed Shapes!")
Jx_final, Jy_final, Jz_final = compute_staggered_JxB()
print(f"Final avg |J×B| = {float(cp.mean(cp.sqrt(Jx_final**2 + Jy_final**2 + Jz_final**2))):.2e}")
