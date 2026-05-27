import numpy as np

# ====================== PARAMETERS ======================
N = 128
L = 1.0
dx = L / N
gamma = 5.0 / 3.0
CFL = 0.28

# Grid
x = np.linspace(dx/2, L - dx/2, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# ====================== VARIABLES ======================
rho = np.ones((N, N, N)) * 1.0
mx = np.zeros((N, N, N))
my = np.zeros((N, N, N))
mz = np.zeros((N, N, N))
E_total = np.ones((N, N, N)) * 3.0

# Staggered B (True Yee)
Bx = np.zeros((N+1, N, N))
By = np.zeros((N, N+1, N))
Bz = np.zeros((N, N, N+1))

np.random.seed(42)
pert = 0.08
Bx[1:-1] = np.random.randn(N-1, N, N) * pert
By[:,1:-1] = np.random.randn(N, N-1, N) * pert
Bz[:-1,:-1] = np.random.randn(N, N, N) * pert * 0.5 + 0.5

# ====================== FULL VECTORIZED EMF KERNEL (All 3 Edges) ======================
def compute_emfs(vx, vy, vz, Bx, By, Bz, dx):
    """Complete vectorized EMF calculation for all three edge types"""
    N = vx.shape[0]
    
    Emfx = np.zeros((N, N+1, N+1))   # x-parallel edges (yz planes)
    Emfy = np.zeros((N+1, N, N+1))   # y-parallel edges (xz planes)
    Emfz = np.zeros((N+1, N+1, N))   # z-parallel edges (xy planes)
    
    # ==================== EMF_z (xy-edges) ====================
    vx_c = 0.25 * (vx[:-1, :-1, :] + vx[1:, :-1, :] + vx[:-1, 1:, :] + vx[1:, 1:, :])
    vy_c = 0.25 * (vy[:-1, :-1, :] + vy[1:, :-1, :] + vy[:-1, 1:, :] + vy[1:, 1:, :])
    Bx_e = 0.5 * (Bx[1:, :-1, :] + Bx[1:, 1:, :])
    By_e = 0.5 * (By[:-1, 1:, :] + By[1:, 1:, :])
    Emfz[:-1, :-1, :] = -(vx_c * By_e - vy_c * Bx_e)
    
    # ==================== EMF_y (xz-edges) ====================
    vx_c = 0.25 * (vx[:-1, :, :-1] + vx[1:, :, :-1] + vx[:-1, :, 1:] + vx[1:, :, 1:])
    vz_c = 0.25 * (vz[:-1, :, :-1] + vz[1:, :, :-1] + vz[:-1, :, 1:] + vz[1:, :, 1:])
    Bx_e = 0.5 * (Bx[1:, :, :-1] + Bx[1:, :, 1:])
    Bz_e = 0.5 * (Bz[:-1, :, 1:] + Bz[1:, :, 1:])
    Emfy[:-1, :, :-1] = -(vz_c * Bx_e - vx_c * Bz_e)
    
    # ==================== EMF_x (yz-edges) ====================
    vy_c = 0.25 * (vy[:, :-1, :-1] + vy[:, 1:, :-1] + vy[:, :-1, 1:] + vy[:, 1:, 1:])
    vz_c = 0.25 * (vz[:, :-1, :-1] + vz[:, 1:, :-1] + vz[:, :-1, 1:] + vz[:, 1:, 1:])
    By_e = 0.5 * (By[:, 1:, :-1] + By[:, 1:, 1:])
    Bz_e = 0.5 * (Bz[:, :-1, 1:] + Bz[:, 1:, 1:])
    Emfx[:, :-1, :-1] = -(vy_c * Bz_e - vz_c * By_e)
    
    return Emfx, Emfy, Emfz


# ====================== J x B (Fixed) ======================
def compute_jxb(Bx, By, Bz, dx):
    N = Bx.shape[1]
    jxbx = np.zeros((N, N, N))
    jxby = np.zeros((N, N, N))
    jxbz = np.zeros((N, N, N))
    
    jx = (By[:, :, 1:] - By[:, :, :-1]) / dx - (Bz[1:, :, :] - Bz[:-1, :, :]) / dx
    jy = (Bz[1:, :, :] - Bz[:-1, :, :]) / dx - (Bx[:, :, 1:] - Bx[:, :, :-1]) / dx
    jz = (Bx[:, 1:, :] - Bx[:, :-1, :]) / dx - (By[1:, :, :] - By[:-1, :, :]) / dx
    
    bx = 0.5 * (Bx[1:, :, :] + Bx[:-1, :, :])
    by = 0.5 * (By[:, 1:, :] + By[:, :-1, :])
    bz = 0.5 * (Bz[:, :, 1:] + Bz[:, :, :-1])
    
    jxbx = jy * bz - jz * by
    jxby = jz * bx - jx * bz
    jxbz = jx * by - jy * bx
    
    return jxbx, jxby, jxbz


# ====================== MAIN LOOP ======================
steps = 0
max_steps = 400
dt = 0.001

print("True CT Yee with Full Vectorized EMF (All 3 Edges) Running...\n")

while steps < max_steps:
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    p = (gamma - 1.0) * (E_total - kin)
    
    # J × B
    jxbx, jxby, jxbz = compute_jxb(Bx, By, Bz, dx)
    mx += dt * jxbx
    my += dt * jxby
    mz += dt * jxbz
    
    # Full vectorized EMF at all cell edges
    Emfx, Emfy, Emfz = compute_emfs(vx, vy, vz, Bx, By, Bz, dx)
    
    # True CT Update
    Bx[1:-1] -= (dt / dx) * (Emfz[1:-1,1:,:] - Emfz[1:-1,:-1,:] - Emfy[1:-1,:,1:] + Emfy[1:-1,:,:-1])
    By[:,1:-1] -= (dt / dx) * (Emfx[:,1:-1,1:] - Emfx[:,1:-1,:-1] - Emfz[1:,1:-1,:] + Emfz[:-1,1:-1,:])
    Bz[:-1,:-1] -= (dt / dx) * (Emfy[1:,:-1,:-1] - Emfy[:-1,:-1,:-1] - Emfx[:-1,1:,:-1] + Emfx[:-1,:-1,:-1])
    
    steps += 1
    if steps % 50 == 0:
        vmax = np.sqrt(vx**2 + vy**2 + vz**2).max()
        bmag = np.sqrt(Bx.mean()**2 + By.mean()**2 + Bz.mean()**2)
        print(f"Step {steps:4d} | dt={dt:.5f} | Max|v|={vmax:.4f} | <B>={bmag:.4f}")

print("\nTrue CT Yee with complete vectorized EMF kernel finished.")
