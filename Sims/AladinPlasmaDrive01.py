"""
ALADIN Plasma Driver v1
Galactic-disk MHD operator

Eight kernels (physics order):
    gyro, hlld_x, hlld_y, hlld_z, uct_assembly, emf, curl, glm

Flow:
    gyrotropic closure
    → HLLD face fluxes
    → UCT ideal EMF
    → generalized Ohm (Hall + electron pressure)
    → CT update
    → GLM cleaning
    → gravity source

The driver owns the integrator, the state contracts, and the diagnostics.
Kernels are pure operators.
"""

import numpy as np
import time

USE_GPU = False
try:
    import cupy as cp
    # Test that a device is actually available
    _ = cp.zeros(1)
    USE_GPU = True
except Exception:
    import numpy as cp
    USE_GPU = False
    print("CuPy/CUDA not available — running on NumPy (structure test)")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
class Config:
    # Numerical parameters
    N = 128                 # interior cells per dimension
    L = 16.0                # domain half-width
    NG = 2                  # ghost layers
    cfl = 0.08              # CFL number
    dt_max = 0.002          # maximum timestep
    max_steps = 20          # total steps
    print_interval = 5      # diagnostic print frequency
    # Physical parameters
    hall_coeff = 0.05       # Hall coefficient
    Rd = 4.0                # disk radial scale
    zd = 0.6                # disk vertical scale
    rho0 = 1.0              # central density
    Rc = 3.0                # external potential core radius
    v0 = 0.25               # external potential velocity scale
    cs2 = 0.02              # soft sound speed squared (pressure support)
    Bz0 = 0.008             # central vertical field
    # Runtime
    BC_TYPE = 0             # 0=periodic, 1=outflow
    # Physics switches — set False to skip a kernel
    USE_GYRO = True
    USE_HLLD = True
    USE_UCT  = True
    USE_EMF  = True
    USE_CURL = True
    USE_GLM  = True
    USE_GRAVITY = True

cfg = Config()
N, L, NG = cfg.N, cfg.L, cfg.NG
dx = L / N
Ni = N + 2 * NG

class Grid:
    """
    Cell-centered geometry.
    Ghost cells lie outside the physical domain.
    L is the full physical domain length.
    """
    def __init__(self, N, L, NG):
        self.N = N
        self.L = L
        self.NG = NG
        self.dx = L / N
        self.Ni = N + 2 * NG
        self.interior = slice(NG, NG + N)
        # Cell centers: ghosts outside [-L/2, L/2]
        self.x = -L/2 + (cp.arange(self.Ni) - NG + 0.5) * self.dx
        self.y = self.x.copy()
        self.z = self.x.copy()

grid = Grid(N, L, NG)
X, Y, Z = cp.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
R = cp.sqrt(X**2 + Y**2)
interior = (grid.interior, grid.interior, grid.interior)

# =====================================================================
# 2. STATE CONTRACTS
# =====================================================================
class FluidState:
    """
    Prognostic fluid variables for conservative energy MHD.
    rho, mx, my, mz, E are advanced by the integrator.
    vx, vy, vz are derived from momentum and density.
    """
    def __init__(self):
        self.rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32)
        self.mx  = cp.zeros_like(self.rho)
        self.my  = cp.zeros_like(self.rho)
        self.mz  = cp.zeros_like(self.rho)
        self.E   = cp.ones_like(self.rho)   # total energy density
        self.vx  = cp.zeros_like(self.rho)
        self.vy  = cp.zeros_like(self.rho)
        self.vz  = cp.zeros_like(self.rho)

    def update_velocity(self):
        """Compute velocity from momentum. Density is floored for safety."""
        rho_safe = cp.maximum(self.rho, 1e-5)
        self.vx[:] = self.mx / rho_safe
        self.vy[:] = self.my / rho_safe
        self.vz[:] = self.mz / rho_safe

class MagneticState:
    """
    Prognostic magnetic variables.
    Bx, By, Bz are advanced by CT.
    psi is the GLM divergence-cleaning field.
    Emfx, Emfy, Emfz are the edge electric fields used by CT.
    """
    def __init__(self):
        self.Bx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
        self.By = cp.zeros_like(self.Bx)
        self.Bz = cp.zeros_like(self.Bx)
        self.Emfx = cp.zeros_like(self.Bx)
        self.Emfy = cp.zeros_like(self.Bx)
        self.Emfz = cp.zeros_like(self.Bx)
        self.psi  = cp.zeros_like(self.Bx)

class MomentState:
    """
    Diagnostic moment state produced by the gyrotropic closure.
    Ppar, Pperp are the parallel and perpendicular pressures.
    pe_* is the Cartesian electron pressure tensor reconstructed from them.
    P is the isotropic average.
    These fields are recomputed every stage; they are not advanced by the integrator.
    """
    def __init__(self):
        self.Ppar  = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
        self.Pperp = cp.zeros_like(self.Ppar)
        self.qpar  = cp.zeros_like(self.Ppar)
        self.qperp = cp.zeros_like(self.Ppar)
        self.P     = cp.zeros_like(self.Ppar)
        self.pe_xx = cp.zeros_like(self.Ppar)
        self.pe_yy = cp.zeros_like(self.Ppar)
        self.pe_zz = cp.zeros_like(self.Ppar)
        self.pe_xy = cp.zeros_like(self.Ppar)
        self.pe_xz = cp.zeros_like(self.Ppar)
        self.pe_yz = cp.zeros_like(self.Ppar)

# Stage buffers for SSP-RK3
fluid  = FluidState()
fluid1 = FluidState()
fluid2 = FluidState()
fluid3 = FluidState()
mag    = MagneticState()
mag1   = MagneticState()
mag2   = MagneticState()
mag3   = MagneticState()
mom  = MomentState()
mom1 = MomentState()
mom2 = MomentState()

# =====================================================================
# 3. INITIAL CONDITION
# =====================================================================
# Force-balanced exponential disk.
# Density falls radially and vertically.
# Rotation is set so centrifugal force balances the external potential
# gradient plus a soft pressure support term.
# This keeps the early evolution from being dominated by radial collapse.
fluid.rho[:] = cfg.rho0 * cp.exp(-R / cfg.Rd) * cp.exp(-cp.abs(Z) / cfg.zd)
fluid.rho[:] = cp.maximum(fluid.rho, 1e-5)
mag.Bz[:] = cfg.Bz0 * cp.exp(-R / (2*cfg.Rd))

# External logarithmic potential and its radial derivative
phi_ext = 0.5 * cfg.v0**2 * cp.log(R**2 + Z**2 + cfg.Rc**2)
dphi_dR = cfg.v0**2 * R / (R**2 + Z**2 + cfg.Rc**2)
# Force balance: v_phi^2 / R = d(phi_ext)/dR + pressure support
v_phi2 = cp.maximum(R * dphi_dR - cfg.cs2 * (R / cfg.Rd), 1e-6)
v_phi = cp.sqrt(v_phi2)
fluid.mx[:] = -v_phi * Y / (R + 1e-4) * fluid.rho
fluid.my[:] =  v_phi * X / (R + 1e-4) * fluid.rho
# Total energy: internal + kinetic (magnetic added later by the energy equation)
gamma = 5.0/3.0
P_init = cfg.cs2 * fluid.rho
B2 = mag.Bx**2 + mag.By**2 + mag.Bz**2
fluid.E[:] = P_init / (gamma - 1.0) + 0.5 * (fluid.mx**2 + fluid.my**2 + fluid.mz**2) / fluid.rho + 0.5 * B2

def compute_gravity(fluid_state):
    """
    External logarithmic potential plus periodic self-gravity.
    Self-gravity is solved with an FFT Poisson solver.
    Returns the gravitational acceleration (gx, gy, gz).
    """
    if not cfg.USE_GRAVITY:
        z = cp.zeros_like(fluid_state.rho)
        return z, z, z
    rho_hat = cp.fft.fftn(fluid_state.rho)
    k = cp.fft.fftfreq(Ni, d=dx) * 2 * np.pi
    KX, KY, KZ = cp.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    phi_self_hat = -rho_hat / K2
    phi_self_hat[0, 0, 0] = 0.0
    phi_self = cp.fft.ifftn(phi_self_hat).real.astype(cp.float32)
    phi = phi_ext + phi_self
    gx = -(cp.roll(phi, -1, 0) - cp.roll(phi, 1, 0)) * 0.5 / dx
    gy = -(cp.roll(phi, -1, 1) - cp.roll(phi, 1, 1)) * 0.5 / dx
    gz = -(cp.roll(phi, -1, 2) - cp.roll(phi, 1, 2)) * 0.5 / dx
    return gx, gy, gz

# =====================================================================
# 4. KERNELS (placeholders — drop real ones later)
# =====================================================================
if not USE_GPU:
    # Dummy kernels so the driver structure can be tested on CPU
    def _dummy(*a, **k): pass
    gyrotropic_12_kernel = ct_emf_kernel = ct_curl_kernel = glm_kernel = _dummy
    hlld_x_kernel = hlld_y_kernel = hlld_z_kernel = uct_assembly_kernel = _dummy
else:
    BLOCK_GYRO = (64,)
    BLOCK_HLLD_X = (16, 16, 2)
    BLOCK_HLLD_Y = (16, 16, 2)
    BLOCK_HLLD_Z = (16, 16, 2)
    BLOCK_UCT = (16, 4, 4)
    BLOCK_EMF  = (16, 4, 4)
    BLOCK_CURL = (32, 4, 4)
    BLOCK_GLM  = (32, 4, 4)

    grid_gyro = ((Ni * Ni * Ni + 63) // 64,)
    grid_hlld_x = ((Ni + 15) // 16, (Ni + 15) // 16, (Ni + 1) // 2)
    grid_hlld_y = ((Ni + 15) // 16, (Ni + 15) // 16, (Ni + 1) // 2)
    grid_hlld_z = ((Ni + 15) // 16, (Ni + 15) // 16, (Ni + 1) // 2)
    grid_uct = ((Ni + 15) // 16, (Ni + 3) // 4, (Ni + 3) // 4)
    grid_emf  = ((Ni + 15) // 16, (Ni + 3) // 4, (Ni + 3) // 4)
    grid_curl = ((Ni + 31) // 32, (Ni + 3) // 4, (Ni + 3) // 4)
    grid_glm  = ((Ni + 31) // 32, (Ni + 3) // 4, (Ni + 3) // 4)

    stream_gyro = cp.cuda.Stream(non_blocking=True)
    stream_hlld_x = cp.cuda.Stream(non_blocking=True)
    stream_hlld_y = cp.cuda.Stream(non_blocking=True)
    stream_hlld_z = cp.cuda.Stream(non_blocking=True)
    stream_uct = cp.cuda.Stream(non_blocking=True)
    stream_emf  = cp.cuda.Stream(non_blocking=True)
    stream_curl = cp.cuda.Stream(non_blocking=True)
    stream_glm  = cp.cuda.Stream(non_blocking=True)

    gyrotropic_12_kernel = cp.RawKernel(r'''
    extern "C" __global__ void gyrotropic_12_kernel(
        const float* n_in, float* Ppar, float* Pperp, int Ncells, float thermal_scale) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= Ncells) return;
        float n = fmaxf(n_in[idx], 1e-6f);
        Ppar[idx] = n * thermal_scale * thermal_scale;
        Pperp[idx] = 0.5f * Ppar[idx];
    }
    ''', 'gyrotropic_12_kernel')

    ct_emf_kernel = cp.RawKernel(r'''
    extern "C" __global__ void ct_emf_kernel(
        const float* rho, const float* mx, const float* my, const float* mz,
        const float* Bx, const float* By, const float* Bz,
        const float* pe_xx, const float* pe_yy, const float* pe_zz,
        const float* pe_xy, const float* pe_xz, const float* pe_yz,
        float* dEx, float* dEy, float* dEz, int Ni, float hall_coeff, float dx) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
        int idx = i*Ni*Ni + j*Ni + k;
        float rs = fmaxf(rho[idx], 1e-6f);
        float vx = mx[idx]/rs, vy = my[idx]/rs, vz = mz[idx]/rs;
        float inv2dx = 0.5f / dx;
        float inv_ne = 1.0f / rs;
        // Ideal EMF: -v × B
        float Ex = -(vy*Bz[idx] - vz*By[idx]);
        float Ey = -(vz*Bx[idx] - vx*Bz[idx]);
        float Ez = -(vx*By[idx] - vy*Bx[idx]);
        // Current
        float jx = ((Bz[idx+Ni] - Bz[idx-Ni]) - (By[idx+1] - By[idx-1])) * inv2dx;
        float jy = ((Bx[idx+1] - Bx[idx-1]) - (Bz[idx+Ni*Ni] - Bz[idx-Ni*Ni])) * inv2dx;
        float jz = ((By[idx+Ni*Ni] - By[idx-Ni*Ni]) - (Bx[idx+Ni] - Bx[idx-Ni])) * inv2dx;
        // Hall
        Ex += (jy*Bz[idx] - jz*By[idx]) * hall_coeff * inv_ne;
        Ey += (jz*Bx[idx] - jx*Bz[idx]) * hall_coeff * inv_ne;
        Ez += (jx*By[idx] - jy*Bx[idx]) * hall_coeff * inv_ne;
        float dPexx_dx = (pe_xx[idx+Ni*Ni] - pe_xx[idx-Ni*Ni]) * inv2dx;
        float dPexy_dy = (pe_xy[idx+Ni] - pe_xy[idx-Ni]) * inv2dx;
        float dPexz_dz = (pe_xz[idx+1] - pe_xz[idx-1]) * inv2dx;
        float dPeyx_dx = (pe_xy[idx+Ni*Ni] - pe_xy[idx-Ni*Ni]) * inv2dx;
        float dPeyy_dy = (pe_yy[idx+Ni] - pe_yy[idx-Ni]) * inv2dx;
        float dPeyz_dz = (pe_yz[idx+1] - pe_yz[idx-1]) * inv2dx;
        float dPezx_dx = (pe_xz[idx+Ni*Ni] - pe_xz[idx-Ni*Ni]) * inv2dx;
        float dPezy_dy = (pe_yz[idx+Ni] - pe_yz[idx-Ni]) * inv2dx;
        float dPezz_dz = (pe_zz[idx+1] - pe_zz[idx-1]) * inv2dx;
        Ex -= (dPexx_dx + dPexy_dy + dPexz_dz) * inv_ne;
        Ey -= (dPeyx_dx + dPeyy_dy + dPeyz_dz) * inv_ne;
        Ez -= (dPezx_dx + dPezy_dy + dPezz_dz) * inv_ne;
        dEx[idx] = Ex;
        dEy[idx] = Ey;
        dEz[idx] = Ez;
    }
    ''', 'ct_emf_kernel')

    ct_curl_kernel = cp.RawKernel(r'''
    extern "C" __global__ void ct_curl_kernel(
        const float* Emfx, const float* Emfy, const float* Emfz,
        float* Bx, float* By, float* Bz, int Ni, float dt, float inv2dx) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
        int idx = i*Ni*Ni + j*Ni + k;
        Bx[idx] -= dt * (Emfz[idx+Ni]-Emfz[idx-Ni] - Emfy[idx+1]+Emfy[idx-1]) * inv2dx;
        By[idx] -= dt * (Emfx[idx+1]-Emfx[idx-1] - Emfz[idx+Ni*Ni]+Emfz[idx-Ni*Ni]) * inv2dx;
        Bz[idx] -= dt * (Emfy[idx+Ni*Ni]-Emfy[idx-Ni*Ni] - Emfx[idx+Ni]+Emfx[idx-Ni]) * inv2dx;
    }
    ''', 'ct_curl_kernel')

    glm_kernel = cp.RawKernel(r'''
    extern "C" __global__ void glm_kernel(
        const float* Bx_in, const float* By_in, const float* Bz_in, const float* psi_in,
        float* Bx_out, float* By_out, float* Bz_out, float* psi_out,
        int Ni, float dt, float dx, float ch, float cr) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
        int idx = i*Ni*Ni + j*Ni + k;
        float inv2dx = 0.5f / dx;
        // All reads from immutable input
        float divB = ((Bx_in[idx+Ni*Ni]-Bx_in[idx-Ni*Ni]) +
                      (By_in[idx+Ni]-By_in[idx-Ni]) +
                      (Bz_in[idx+1]-Bz_in[idx-1])) * inv2dx;
        float psi_new = psi_in[idx] - dt * ch*ch * divB;
        psi_new *= expf(-dt * cr * ch / dx);
        psi_out[idx] = psi_new;
        // Magnetic update uses only the input psi
        Bx_out[idx] = Bx_in[idx] - dt * (psi_in[idx+Ni*Ni]-psi_in[idx-Ni*Ni]) * inv2dx;
        By_out[idx] = By_in[idx] - dt * (psi_in[idx+Ni]-psi_in[idx-Ni]) * inv2dx;
        Bz_out[idx] = Bz_in[idx] - dt * (psi_in[idx+1]-psi_in[idx-1]) * inv2dx;
    }
    ''', 'glm_kernel')

    hlld_x_kernel = cp.RawKernel(r'''
    extern "C" __global__ void hlld_x_kernel(
        const float* rho, const float* mx, const float* my, const float* mz,
        const float* Bx, const float* By, const float* Bz,
        float* rho_out, float* mx_out, float* my_out, float* mz_out,
        int Ni, float dt_over_dx) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (i < 1 || i >= Ni-1 || j >= Ni || k >= Ni) return;
        int idx = i*Ni*Ni + j*Ni + k;
        rho_out[idx] = rho[idx]; mx_out[idx] = mx[idx];
        my_out[idx] = my[idx]; mz_out[idx] = mz[idx];
    }
    ''', 'hlld_x_kernel')

    hlld_y_kernel = cp.RawKernel(r'''
    extern "C" __global__ void hlld_y_kernel(
        const float* rho, const float* mx, const float* my, const float* mz,
        const float* Bx, const float* By, const float* Bz,
        float* rho_out, float* mx_out, float* my_out, float* mz_out,
        int Ni, float dt_over_dx) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (j < 1 || j >= Ni-1 || i >= Ni || k >= Ni) return;
        int idx = i*Ni*Ni + j*Ni + k;
        rho_out[idx] = rho[idx]; mx_out[idx] = mx[idx];
        my_out[idx] = my[idx]; mz_out[idx] = mz[idx];
    }
    ''', 'hlld_y_kernel')

    hlld_z_kernel = cp.RawKernel(r'''
    extern "C" __global__ void hlld_z_kernel(
        const float* rho, const float* mx, const float* my, const float* mz,
        const float* Bx, const float* By, const float* Bz,
        float* rho_out, float* mx_out, float* my_out, float* mz_out,
        int Ni, float dt_over_dx) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (k < 1 || k >= Ni-1 || i >= Ni || j >= Ni) return;
        int idx = i*Ni*Ni + j*Ni + k;
        rho_out[idx] = rho[idx]; mx_out[idx] = mx[idx];
        my_out[idx] = my[idx]; mz_out[idx] = mz[idx];
    }
    ''', 'hlld_z_kernel')

    uct_assembly_kernel = cp.RawKernel(r'''
    extern "C" __global__ void uct_assembly_kernel(
        const float* Bx, const float* By, const float* Bz,
        float* Emfx, float* Emfy, float* Emfz, int Ni) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        if (i >= Ni || j >= Ni || k >= Ni) return;
        int idx = i*Ni*Ni + j*Ni + k;
        Emfx[idx] = 0.0f;
        Emfy[idx] = 0.0f;
        Emfz[idx] = 0.0f;
    }
    ''', 'uct_assembly_kernel')

    # =====================================================================
# 5. OPERATOR CALLS
# =====================================================================
def reconstruct_pressure_tensor(moment_state, mag_state):
    B2 = mag_state.Bx**2 + mag_state.By**2 + mag_state.Bz**2
    Bmag = cp.sqrt(cp.maximum(B2, 1e-12))
    bx = mag_state.Bx / Bmag
    by = mag_state.By / Bmag
    bz = mag_state.Bz / Bmag
    dP = moment_state.Ppar - moment_state.Pperp
    moment_state.pe_xx[:] = moment_state.Pperp + dP * bx * bx
    moment_state.pe_yy[:] = moment_state.Pperp + dP * by * by
    moment_state.pe_zz[:] = moment_state.Pperp + dP * bz * bz
    moment_state.pe_xy[:] = dP * bx * by
    moment_state.pe_xz[:] = dP * bx * bz
    moment_state.pe_yz[:] = dP * by * bz
    moment_state.P[:] = (moment_state.Ppar + 2.0 * moment_state.Pperp) / 3.0

def call_gyro(fluid_state, moment_state, mag_state):
    if not cfg.USE_GYRO or not USE_GPU:
        return
    n_flat = fluid_state.rho.ravel()
    Ppar = cp.empty_like(n_flat)
    Pperp = cp.empty_like(n_flat)
    gyrotropic_12_kernel(grid_gyro, BLOCK_GYRO,
        (n_flat, Ppar, Pperp, fluid_state.rho.size, np.float32(0.8)),
        stream=stream_gyro)
    stream_gyro.synchronize()
    moment_state.Ppar.ravel()[:] = Ppar
    moment_state.Pperp.ravel()[:] = Pperp
    reconstruct_pressure_tensor(moment_state, mag_state)

def call_uct(mag_state):
    if not cfg.USE_UCT or not USE_GPU:
        return
    uct_assembly_kernel(grid_uct, BLOCK_UCT,
        (mag_state.Bx, mag_state.By, mag_state.Bz,
         mag_state.Emfx, mag_state.Emfy, mag_state.Emfz, Ni),
        stream=stream_uct)
    stream_uct.synchronize()

def call_emf(fluid_state, mag_state, moment_state):
    if not cfg.USE_EMF or not USE_GPU:
        return
    dEx = cp.zeros_like(mag_state.Emfx)
    dEy = cp.zeros_like(mag_state.Emfy)
    dEz = cp.zeros_like(mag_state.Emfz)
    ct_emf_kernel(grid_emf, BLOCK_EMF,
        (fluid_state.rho, fluid_state.mx, fluid_state.my, fluid_state.mz,
         mag_state.Bx, mag_state.By, mag_state.Bz,
         moment_state.pe_xx, moment_state.pe_yy, moment_state.pe_zz,
         moment_state.pe_xy, moment_state.pe_xz, moment_state.pe_yz,
         dEx, dEy, dEz,
         Ni, np.float32(cfg.hall_coeff), np.float32(dx)),
        stream=stream_emf)
    stream_emf.synchronize()
    mag_state.Emfx[:] += dEx
    mag_state.Emfy[:] += dEy
    mag_state.Emfz[:] += dEz

def call_curl(dt, mag_in, mag_out):
    if not cfg.USE_CURL or not USE_GPU:
        mag_out.Bx[:] = mag_in.Bx
        mag_out.By[:] = mag_in.By
        mag_out.Bz[:] = mag_in.Bz
        return
    inv2dx = 0.5 / dx
    mag_out.Bx[:] = mag_in.Bx
    mag_out.By[:] = mag_in.By
    mag_out.Bz[:] = mag_in.Bz
    mag_out.psi[:] = mag_in.psi
    mag_out.Emfx[:] = mag_in.Emfx
    mag_out.Emfy[:] = mag_in.Emfy
    mag_out.Emfz[:] = mag_in.Emfz
    ct_curl_kernel(grid_curl, BLOCK_CURL,
        (mag_out.Emfx, mag_out.Emfy, mag_out.Emfz,
         mag_out.Bx, mag_out.By, mag_out.Bz,
         Ni, np.float32(dt), np.float32(inv2dx)),
        stream=stream_curl)
    stream_curl.synchronize()

def call_glm(dt, mag_state):
    if not cfg.USE_GLM or not USE_GPU:
        return
    ch = 0.4 * dx / max(dt, 1e-8)
    # Copy input so ghost cells are preserved
    Bx_out = mag_state.Bx.copy()
    By_out = mag_state.By.copy()
    Bz_out = mag_state.Bz.copy()
    psi_out = mag_state.psi.copy()
    glm_kernel(grid_glm, BLOCK_GLM,
        (mag_state.Bx, mag_state.By, mag_state.Bz, mag_state.psi,
         Bx_out, By_out, Bz_out, psi_out,
         Ni, np.float32(dt), np.float32(dx), np.float32(ch), np.float32(0.05)),
        stream=stream_glm)
    stream_glm.synchronize()
    mag_state.Bx[:] = Bx_out
    mag_state.By[:] = By_out
    mag_state.Bz[:] = Bz_out
    mag_state.psi[:] = psi_out

def call_hlld(dt, fluid_in, mag_state, fluid_out):
    if not cfg.USE_HLLD or not USE_GPU:
        fluid_out.rho[:] = fluid_in.rho
        fluid_out.mx[:]  = fluid_in.mx
        fluid_out.my[:]  = fluid_in.my
        fluid_out.mz[:]  = fluid_in.mz
        fluid_out.E[:]   = fluid_in.E
        return
    dt_over_dx = dt / dx
    fluid_out.E[:] = fluid_in.E
    hlld_x_kernel(grid_hlld_x, BLOCK_HLLD_X,
        (fluid_in.rho, fluid_in.mx, fluid_in.my, fluid_in.mz,
         mag_state.Bx, mag_state.By, mag_state.Bz,
         fluid_out.rho, fluid_out.mx, fluid_out.my, fluid_out.mz,
         Ni, np.float32(dt_over_dx)), stream=stream_hlld_x)
    stream_hlld_x.synchronize()
    hlld_y_kernel(grid_hlld_y, BLOCK_HLLD_Y,
        (fluid_out.rho, fluid_out.mx, fluid_out.my, fluid_out.mz,
         mag_state.Bx, mag_state.By, mag_state.Bz,
         fluid_out.rho, fluid_out.mx, fluid_out.my, fluid_out.mz,
         Ni, np.float32(dt_over_dx)), stream=stream_hlld_y)
    stream_hlld_y.synchronize()
    hlld_z_kernel(grid_hlld_z, BLOCK_HLLD_Z,
        (fluid_out.rho, fluid_out.mx, fluid_out.my, fluid_out.mz,
         mag_state.Bx, mag_state.By, mag_state.Bz,
         fluid_out.rho, fluid_out.mx, fluid_out.my, fluid_out.mz,
         Ni, np.float32(dt_over_dx)), stream=stream_hlld_z)
    stream_hlld_z.synchronize()

def apply_gravity(dt, fluid_state, gx, gy, gz):
    """Add gravitational source to momentum and energy work."""
    fluid_state.mx[:] += dt * fluid_state.rho * gx
    fluid_state.my[:] += dt * fluid_state.rho * gy
    fluid_state.mz[:] += dt * fluid_state.rho * gz
    # Energy work: rho * v · g
    fluid_state.update_velocity()
    fluid_state.E[:] += dt * fluid_state.rho * (
        fluid_state.vx * gx + fluid_state.vy * gy + fluid_state.vz * gz)

# =====================================================================
# 6. TIME INTEGRATOR (SSP-RK3)
# =====================================================================
def operator_step(dt):
    """
    SSP-RK3 with stage-consistent gravity.
    Gravity is recomputed from the density of each stage.
    """
    # ----- Stage 1 -----
    print("\r  Stage 1/3  ghosts: ", end="", flush=True)
    update_fluid_ghosts(fluid)
    print("fluid ", end="", flush=True)
    update_magnetic_ghosts(mag)
    print("mag  kernels: ", end="", flush=True)
    update_magnetic_ghosts(mag)
    fluid.update_velocity()
    gx, gy, gz = compute_gravity(fluid)
    call_gyro(fluid, mom, mag)
    print("gyro ", end="", flush=True)
    call_hlld(dt, fluid, mag, fluid1)
    print("hlld ", end="", flush=True)
    call_uct(mag)
    print("uct ", end="", flush=True)
    call_emf(fluid, mag, mom)
    print("emf ", end="", flush=True)
    call_curl(dt, mag, mag1)
    print("curl ", end="", flush=True)
    call_glm(dt, mag1)
    print("glm ", end="", flush=True)
    apply_gravity(dt, fluid1, gx, gy, gz)
    print("grav", end="", flush=True)
    # ----- Stage 2 -----
    print("\r  Stage 2/3", end="", flush=True)
    update_fluid_ghosts(fluid1)
    update_magnetic_ghosts(mag1)
    gx, gy, gz = compute_gravity(fluid1)
    call_gyro(fluid1, mom1, mag1)
    call_hlld(dt, fluid1, mag1, fluid2)
    call_uct(mag1)
    call_emf(fluid1, mag1, mom1)
    call_curl(dt, mag1, mag2)
    call_glm(dt, mag2)
    apply_gravity(dt, fluid2, gx, gy, gz)
    fluid2.rho[:] = 0.75 * fluid.rho + 0.25 * fluid2.rho
    fluid2.mx[:]  = 0.75 * fluid.mx  + 0.25 * fluid2.mx
    fluid2.my[:]  = 0.75 * fluid.my  + 0.25 * fluid2.my
    fluid2.mz[:]  = 0.75 * fluid.mz  + 0.25 * fluid2.mz
    fluid2.E[:]   = 0.75 * fluid.E   + 0.25 * fluid2.E
    mag2.Bx[:] = 0.75 * mag.Bx + 0.25 * mag2.Bx
    mag2.By[:] = 0.75 * mag.By + 0.25 * mag2.By
    mag2.Bz[:] = 0.75 * mag.Bz + 0.25 * mag2.Bz
    mag2.psi[:] = 0.75 * mag.psi + 0.25 * mag2.psi
    # ----- Stage 3 -----
    print("\r  Stage 3/3", end="", flush=True)
    update_fluid_ghosts(fluid2)
    update_magnetic_ghosts(mag2)
    gx, gy, gz = compute_gravity(fluid2)
    call_gyro(fluid2, mom2, mag2)
    call_hlld(dt, fluid2, mag2, fluid3)
    call_uct(mag2)
    call_emf(fluid2, mag2, mom2)
    call_curl(dt, mag2, mag3)
    call_glm(dt, mag3)
    apply_gravity(dt, fluid3, gx, gy, gz)
    fluid.rho[:] = (1.0/3.0) * fluid.rho + (2.0/3.0) * fluid3.rho
    fluid.mx[:]  = (1.0/3.0) * fluid.mx  + (2.0/3.0) * fluid3.mx
    fluid.my[:]  = (1.0/3.0) * fluid.my  + (2.0/3.0) * fluid3.my
    fluid.mz[:]  = (1.0/3.0) * fluid.mz  + (2.0/3.0) * fluid3.mz
    fluid.E[:]   = (1.0/3.0) * fluid.E   + (2.0/3.0) * fluid3.E
    mag.Bx[:] = (1.0/3.0) * mag.Bx + (2.0/3.0) * mag3.Bx
    mag.By[:] = (1.0/3.0) * mag.By + (2.0/3.0) * mag3.By
    mag.Bz[:] = (1.0/3.0) * mag.Bz + (2.0/3.0) * mag3.Bz
    mag.psi[:] = (1.0/3.0) * mag.psi + (2.0/3.0) * mag3.psi
    fluid.update_velocity()
    print(" blend", end="", flush=True)
    # Density floor and NaN guard
    print(" guard", end="", flush=True)
    fluid.rho[:] = cp.maximum(fluid.rho, 1e-5)
    if not cp.all(cp.isfinite(fluid.E)):
        print("NaN detected in energy")
        fluid.E[:] = cp.nan_to_num(fluid.E, nan=1e-3, posinf=1e3, neginf=1e-3)
    if not cp.all(cp.isfinite(mag.Bx)) or not cp.all(cp.isfinite(mag.By)) or not cp.all(cp.isfinite(mag.Bz)):
        print("NaN detected in B — resetting to initial")
        mag.Bx[:] = 0.0
        mag.By[:] = 0.0
        mag.Bz[:] = cfg.Bz0
        mag.psi[:] = 0.0
        mag.Emfx[:] = 0.0
        mag.Emfy[:] = 0.0
        mag.Emfz[:] = 0.0

# =====================================================================
# 7. BOUNDARIES
# =====================================================================
def update_fluid_ghosts(fluid_state):
    for f in [fluid_state.rho, fluid_state.mx, fluid_state.my, fluid_state.mz, fluid_state.E]:
        f[:NG] = f[-2*NG:-NG]; f[-NG:] = f[NG:2*NG]
        f[:,:NG] = f[:,-2*NG:-NG]; f[:,-NG:] = f[:,NG:2*NG]
        f[:,:, :NG] = f[:,:,-2*NG:-NG]; f[:,:,-NG:] = f[:,:,NG:2*NG]

def update_magnetic_ghosts(mag_state):
    for f in [mag_state.Bx, mag_state.By, mag_state.Bz, mag_state.psi,
              mag_state.Emfx, mag_state.Emfy, mag_state.Emfz]:
        f[:NG] = f[-2*NG:-NG]; f[-NG:] = f[NG:2*NG]
        f[:,:NG] = f[:,-2*NG:-NG]; f[:,-NG:] = f[:,NG:2*NG]
        f[:,:, :NG] = f[:,:,-2*NG:-NG]; f[:,:,-NG:] = f[:,:,NG:2*NG]

# =====================================================================
# 8. DIAGNOSTICS (interior only)
# =====================================================================
# All diagnostics exclude the ghost layers.
# They report on the physical domain only.

def mass():
    """Total mass in the interior domain."""
    return float(cp.sum(fluid.rho[interior]) * dx**3)

def kinetic_energy():
    """Total kinetic energy in the interior domain."""
    v2 = fluid.vx[interior]**2 + fluid.vy[interior]**2 + fluid.vz[interior]**2
    return float(0.5 * cp.sum(fluid.rho[interior] * v2) * dx**3)

def total_energy():
    """Total energy (internal + kinetic + magnetic) in the interior."""
    B2 = mag.Bx[interior]**2 + mag.By[interior]**2 + mag.Bz[interior]**2
    return float(cp.sum(fluid.E[interior]) * dx**3 + 0.5 * cp.sum(B2) * dx**3)

def divB_max():
    """Maximum absolute divergence of B in the interior."""
    inv2dx = 0.5 / dx
    divB = ((cp.roll(mag.Bx, -1, 0) - cp.roll(mag.Bx, 1, 0)) +
            (cp.roll(mag.By, -1, 1) - cp.roll(mag.By, 1, 1)) +
            (cp.roll(mag.Bz, -1, 2) - cp.roll(mag.Bz, 1, 2))) * inv2dx
    return float(cp.max(cp.abs(divB[interior])))

def force_budget(gx, gy, gz, moment_state):
    """
    Radial force budget on the disk.
    Returns dict with gravity, Lorentz, pressure, centrifugal, residual.
    Current is computed as curl(B) with correct signs.
    """
    inv2dx = 0.5 / dx
    # Correct current: J = curl(B)
    jx = ((cp.roll(mag.Bz, -1, 1) - cp.roll(mag.Bz, 1, 1)) -
          (cp.roll(mag.By, -1, 2) - cp.roll(mag.By, 1, 2))) * inv2dx
    jy = ((cp.roll(mag.Bx, -1, 2) - cp.roll(mag.Bx, 1, 2)) -
          (cp.roll(mag.Bz, -1, 0) - cp.roll(mag.Bz, 1, 0))) * inv2dx
    jz = ((cp.roll(mag.By, -1, 0) - cp.roll(mag.By, 1, 0)) -
          (cp.roll(mag.Bx, -1, 1) - cp.roll(mag.Bx, 1, 1))) * inv2dx
    # Lorentz force density
    Fx = jy * mag.Bz - jz * mag.By
    Fy = jz * mag.Bx - jx * mag.Bz
    Fz = jx * mag.By - jy * mag.Bx
    # Radial component
    Fr_L = (Fx * X + Fy * Y) / (R + 1e-3)
    Fr_g = fluid.rho * (gx * X + gy * Y) / (R + 1e-3)
    # Azimuthal velocity only (exclude radial motion)
    v_phi = (-fluid.vx * Y + fluid.vy * X) / (R + 1e-3)
    Fr_c = fluid.rho * v_phi**2 / (R + 1e-3)
    # Pressure (isotropic for now)
    dP_dx = (cp.roll(moment_state.P, -1, 0) - cp.roll(moment_state.P, 1, 0)) * inv2dx
    dP_dy = (cp.roll(moment_state.P, -1, 1) - cp.roll(moment_state.P, 1, 1)) * inv2dx
    Fr_p = -(dP_dx * X + dP_dy * Y) / (R + 1e-3)
    # Residual of steady radial balance
    residual = Fr_g + Fr_p + Fr_L + Fr_c
    disk = (R < 6.0) & (cp.abs(Z) < 1.2) & cp.isfinite(residual)
    vol = float(cp.sum(disk[interior]) * dx**3)
    if vol < 1e-12:
        vol = 1.0
    return {
        "gravity": float(cp.sum(Fr_g[interior] * disk[interior]) * dx**3 / vol),
        "lorentz": float(cp.sum(Fr_L[interior] * disk[interior]) * dx**3 / vol),
        "pressure": float(cp.sum(Fr_p[interior] * disk[interior]) * dx**3 / vol),
        "centrifugal": float(cp.sum(Fr_c[interior] * disk[interior]) * dx**3 / vol),
        "residual": float(cp.sum(residual[interior] * disk[interior]) * dx**3 / vol),
    }

def compute_dt():
    """CFL-limited timestep based on fluid velocity."""
    rho_safe = cp.maximum(fluid.rho, 1e-5)
    v2 = (fluid.mx/rho_safe)**2 + (fluid.my/rho_safe)**2 + (fluid.mz/rho_safe)**2
    cmax = float(cp.sqrt(cp.max(v2) + 1.0))
    return min(cfg.cfl * dx / max(cmax, 1e-3), cfg.dt_max)

# =====================================================================
# 9. MAIN
# =====================================================================
def main():
    print("ALADIN Plasma Driver v2")
    print(f"Grid: {N}^3 interior  ({Ni}^3 padded)  dx={dx:.4f}  NG={NG}")
    print("Flow: gyro → HLLD → UCT → EMF → curl → GLM → gravity")
    print(f"Max steps: {cfg.max_steps}")
    print("[" + " " * 20 + "] 0%", end="", flush=True)
    t0 = time.time()
    for step in range(cfg.max_steps):
        update_fluid_ghosts(fluid)
        update_magnetic_ghosts(mag)
        gx, gy, gz = compute_gravity(fluid)
        dt = compute_dt()
        operator_step(dt)
        # Progress bar
        filled = int(20 * (step + 1) / cfg.max_steps)
        bar = "█" * filled + " " * (20 - filled)
        pct = 100 * (step + 1) / cfg.max_steps
        print(f"\r[{bar}] {pct:5.1f}% complete", end="", flush=True)
        if step % cfg.print_interval == 0:
            m = mass()
            ke = kinetic_energy()
            te = total_energy()
            divb = divB_max()
            fb = force_budget(gx, gy, gz, mom)
            print(f"step {step:3d}/{cfg.max_steps}  dt={dt:.2e}  M={m:.3e}  KE={ke:.3e}  TE={te:.3e}  max|divB|={divb:.2e}")
            print(f"         F_g={fb['gravity']:+.3e}  F_L={fb['lorentz']:+.3e}  F_p={fb['pressure']:+.3e}  F_c={fb['centrifugal']:+.3e}  R={fb['residual']:+.3e}")
            print(f"         min_rho={float(cp.min(fluid.rho)):.2e}  max|E|={float(cp.max(cp.abs(fluid.E))):.2e}  max|B|={float(cp.max(cp.sqrt(mag.Bx**2+mag.By**2+mag.Bz**2))):.2e}")
    print()
    print("Done. Elapsed:", time.time() - t0, "s")

if __name__ == "__main__":
    main()
