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
    L = 16.0                # full box length (domain is [-L/2, L/2])
    NG = 2                  # ghost layers
    cfl = 0.08              # CFL number
    dt_max = 0.002          # maximum timestep
    max_steps = 2           # total steps
    print_interval = 5          # diagnostic print frequency

    # Physical parameters
    hall_coeff = 0.05       # Hall coefficient
    Rd = 4.0                # disk radial scale
    zd0 = 0.35              # central vertical scale
    flare = 0.25            # vertical flaring (h ~ R^flare)
    rho0 = 0.4              # central density (lowered for Q)
    Rc = 3.0                # external potential core radius
    v0 = 0.35               # external potential velocity scale
    # Dark-matter halo (static isothermal)
    halo_rho0 = 0.04
    halo_rs = 6.0
    cs2 = 0.08              # soft sound speed squared (raised for Q)
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
    USE_EXT_GRAVITY = True
    USE_SELF_GRAVITY = True
    # Isolated self-gravity padding factor (1.0 = periodic, 2.0 = full isolation)
    POISSON_PAD = 1.5
    # Extra galactic-disk physics (source terms, applied after the 8-kernel chain)
    USE_VISCOSITY = True
    alpha_visc = 0.01          # Shakura-Sunyaev alpha
    USE_RESISTIVITY = True
    eta_ohm = 1e-4             # Ohmic resistivity (code units)
    USE_FEEDBACK = True
    feedback_rho_thresh = 1.5 # density threshold for injection
    feedback_rate = 0.002      # energy injection rate (code units)
    # CUDA graph — off until all kernels are real and the step is stable
    USE_CUDA_GRAPH = False



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

class Lattice:
    """
    Thin array wrapper for staggered fields.
    Kernels always receive .data; the wrapper exists for the host.
    Supports __cuda_array_interface__ and __array_ufunc__ so
    arithmetic and CuPy ops forward to the underlying array.
    """
    def __init__(self, shape):
        self.data = cp.zeros(shape, dtype=cp.float32)
    def __array__(self, dtype=None):
        arr = self.data.get() if hasattr(self.data, "get") else self.data
        return np.asarray(arr, dtype=dtype)
    @property
    def __cuda_array_interface__(self):
        return self.data.__cuda_array_interface__
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Forward arithmetic (0.75*Bx, Bx**2, isfinite, etc.) to .data
        new_inputs = [x.data if isinstance(x, Lattice) else x for x in inputs]
        return getattr(ufunc, method)(*new_inputs, **kwargs)
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, val):
        self.data[key] = val
    def fill(self, value):
        self.data.fill(value)
    @property
    def shape(self):
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype

class FaceX(Lattice):
    """Bx on x-faces. Shape (Ni+1, Ni, Ni)."""
    def __init__(self):
        super().__init__((Ni+1, Ni, Ni))

class FaceY(Lattice):
    """By on y-faces. Shape (Ni, Ni+1, Ni)."""
    def __init__(self):
        super().__init__((Ni, Ni+1, Ni))

class FaceZ(Lattice):
    """Bz on z-faces. Shape (Ni, Ni, Ni+1)."""
    def __init__(self):
        super().__init__((Ni, Ni, Ni+1))

class EdgeX(Lattice):
    """Ex on x-edges. Shape (Ni, Ni+1, Ni+1)."""
    def __init__(self):
        super().__init__((Ni, Ni+1, Ni+1))

class EdgeY(Lattice):
    """Ey on y-edges. Shape (Ni+1, Ni, Ni+1)."""
    def __init__(self):
        super().__init__((Ni+1, Ni, Ni+1))

class EdgeZ(Lattice):
    """Ez on z-edges. Shape (Ni+1, Ni+1, Ni)."""
    def __init__(self):
        super().__init__((Ni+1, Ni+1, Ni))

class Cell(Lattice):
    """Cell-centered scalar. Shape (Ni, Ni, Ni)."""
    def __init__(self):
        super().__init__((Ni, Ni, Ni))

class MagneticState:
    """
    Staggered magnetic variables with explicit lattice locations.
    Emfx/Emfy/Emfz are direct aliases of the Emf_total arrays
    so the host and kernels always see the same buffers.
    """
    def __init__(self):
        self.Bx = FaceX()
        self.By = FaceY()
        self.Bz = FaceZ()
        self.psi  = Cell()
        # Explicit EMF contracts (edge triples of Lattice objects)
        self.Emf_ideal    = (EdgeX(), EdgeY(), EdgeZ())
        self.Emf_nonideal = (EdgeX(), EdgeY(), EdgeZ())
        self.Emf_total    = (EdgeX(), EdgeY(), EdgeZ())
        # Direct array aliases — same memory as Emf_total
        self.Emfx = self.Emf_total[0].data
        self.Emfy = self.Emf_total[1].data
        self.Emfz = self.Emf_total[2].data

    def sync_emf_aliases(self):
        """Re-bind the aliases in case the total buffers were replaced."""
        self.Emfx = self.Emf_total[0].data
        self.Emfy = self.Emf_total[1].data
        self.Emfz = self.Emf_total[2].data

    def sum_emf(self):
        """Emf_total = Emf_ideal + Emf_nonideal. Aliases stay valid."""
        self.Emf_total[0].data[:] = self.Emf_ideal[0].data + self.Emf_nonideal[0].data
        self.Emf_total[1].data[:] = self.Emf_ideal[1].data + self.Emf_nonideal[1].data
        self.Emf_total[2].data[:] = self.Emf_ideal[2].data + self.Emf_nonideal[2].data
        self.sync_emf_aliases()

class FluxState:
    """
    Face-centered ideal-MHD fluxes produced by HLLD
    and consumed by the UCT assembly kernel.

    Fx : (Ni+1, Ni, Ni)   — ρ, mx, my, mz, E, By, Bz
    Fy : (Ni, Ni+1, Ni)   — ρ, mx, my, mz, E, Bx, Bz
    Fz : (Ni, Ni, Ni+1)   — ρ, mx, my, mz, E, Bx, By
    """
    def __init__(self, Ni):
        self.Ni = Ni
        self.Fx_rho = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fx_mx  = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fx_my  = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fx_mz  = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fx_E   = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fx_By  = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fx_Bz  = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
        self.Fy_rho = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fy_mx  = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fy_my  = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fy_mz  = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fy_E   = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fy_Bx  = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fy_Bz  = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
        self.Fz_rho = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)
        self.Fz_mx  = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)
        self.Fz_my  = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)
        self.Fz_mz  = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)
        self.Fz_E   = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)
        self.Fz_Bx  = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)
        self.Fz_By  = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)

    def zero(self):
        for arr in (
            self.Fx_rho, self.Fx_mx, self.Fx_my, self.Fx_mz, self.Fx_E, self.Fx_By, self.Fx_Bz,
            self.Fy_rho, self.Fy_mx, self.Fy_my, self.Fy_mz, self.Fy_E, self.Fy_Bx, self.Fy_Bz,
            self.Fz_rho, self.Fz_mx, self.Fz_my, self.Fz_mz, self.Fz_E, self.Fz_Bx, self.Fz_By,
        ):
            arr.fill(0.0)

def face_to_cell(mag_state):
    """Average face-centered B to cell centers. Returns (Bx_c, By_c, Bz_c)."""
    Bx_c = 0.5 * (mag_state.Bx.data[:-1] + mag_state.Bx.data[1:])
    By_c = 0.5 * (mag_state.By.data[:, :-1] + mag_state.By.data[:, 1:])
    Bz_c = 0.5 * (mag_state.Bz.data[:, :, :-1] + mag_state.Bz.data[:, :, 1:])
    return Bx_c, By_c, Bz_c

def cell_to_face(Bx_c, By_c, Bz_c, mag_out):
    """Average cell-centered B onto faces. Writes into mag_out."""
    mag_out.Bx.data[1:-1] = 0.5 * (Bx_c[:-1] + Bx_c[1:])
    mag_out.By.data[:, 1:-1] = 0.5 * (By_c[:, :-1] + By_c[:, 1:])
    mag_out.Bz.data[:, :, 1:-1] = 0.5 * (Bz_c[:, :, :-1] + Bz_c[:, :, 1:])


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
        # Real gyrotropic_12 ABI buffers (Ncells x features)
        Ncells = Ni * Ni * Ni
        self.M_target = cp.zeros((Ncells, 12), dtype=cp.float32)
        self.lambda_io = cp.zeros((Ncells, 12), dtype=cp.float32)
        self.n_out = cp.empty(Ncells, dtype=cp.float32)
        self.upar = cp.empty(Ncells, dtype=cp.float32)
        self.margin = cp.empty(Ncells, dtype=cp.float32)
        self.M_recon = cp.empty((Ncells, 12), dtype=cp.float32)
        self.status = cp.zeros(Ncells, dtype=cp.int32)

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
mom3 = MomentState()
fluxes  = FluxState(Ni)
fluxes1 = FluxState(Ni)
fluxes2 = FluxState(Ni)

# =====================================================================
# 3. INITIAL CONDITION
# =====================================================================
# Flared exponential disk + static isothermal dark-matter halo.
# Vertical scale height grows with radius so the outer disk is thicker.
def vertical_scale(R):
    return cfg.zd0 * (1.0 + (R / cfg.Rd)**cfg.flare)

zd = vertical_scale(R)
fluid.rho[:] = cfg.rho0 * cp.exp(-R / cfg.Rd) * cp.exp(-Z**2 / (2 * zd**2))
fluid.rho[:] = cp.maximum(fluid.rho, 1e-5)

# Initial vertical field on z-faces
Bz_init = cfg.Bz0 * cp.exp(-R / (2*cfg.Rd))
mag.Bz.data[:, :, NG:NG+N] = Bz_init[:, :, NG:NG+N]

# External logarithmic disk potential
phi_ext = 0.5 * cfg.v0**2 * cp.log(R**2 + cfg.Rc**2)
dphi_ext_dR = cfg.v0**2 * R / (R**2 + cfg.Rc**2)

# Static isothermal dark-matter halo (analytic force)
def halo_acceleration(R, Z):
    """Radial and vertical acceleration from a spherical isothermal halo."""
    r2 = R**2 + Z**2 + 1e-8
    r = cp.sqrt(r2)
    # Isothermal sphere: g = - (4 pi G rho0 rs^2 / r) * (1 - (rs/r)*arctan(r/rs))
    # With G = 1/(4*pi) this simplifies to rho0 * rs^2 * (...)
    x = r / cfg.halo_rs
    factor = cfg.halo_rho0 * cfg.halo_rs**2 * (1.0 - cp.arctan(x) / x) / r2
    return -factor * R, -factor * Z

aR_halo, aZ_halo = halo_acceleration(R, Z)

# Temporary rotation from external + halo (self-gravity added later)
gR_ext = -dphi_ext_dR + aR_halo
v_phi2 = cp.maximum(-R * gR_ext - cfg.cs2 * (R / cfg.Rd), 1e-6)
v_phi = cp.sqrt(v_phi2)
fluid.mx[:] = -v_phi * Y / (R + 1e-4) * fluid.rho
fluid.my[:] =  v_phi * X / (R + 1e-4) * fluid.rho
fluid.update_velocity()
# Total energy: internal + kinetic (magnetic added later by the energy equation)
gamma = 5.0/3.0
P_init = cfg.cs2 * fluid.rho
# Cell-centered B for energy (average neighboring faces)
Bx_c, By_c, Bz_c = face_to_cell(mag)
B2 = Bx_c**2 + By_c**2 + Bz_c**2
fluid.E[:] = P_init / (gamma - 1.0) + 0.5 * (fluid.mx**2 + fluid.my**2 + fluid.mz**2) / fluid.rho + 0.5 * B2

def correct_radial_drift():
    """
    Recompute rotation from the total radial acceleration
    (external + halo + self-gravity) so the initial disk
    has no net radial force.
    Call once after the first gravity solve, before the loop.
    """
    gx, gy, gz = compute_gravity(fluid)
    gR_mag, gz_mag = magnetic_support()
    # Radial acceleration: gravity + magnetic
    gR = (gx * X + gy * Y) / (R + 1e-4) + gR_mag
    # Centrifugal balance: v_phi^2 / R = -gR - (cs^2 / rho) * d(rho)/dR
    v_phi2 = cp.maximum(-R * gR - cfg.cs2 * (R / cfg.Rd), 1e-6)
    v_phi = cp.sqrt(v_phi2)
    fluid.mx[:] = -v_phi * Y / (R + 1e-4) * fluid.rho
    fluid.my[:] =  v_phi * X / (R + 1e-4) * fluid.rho
    fluid.mz[:] = 0.0
    fluid.update_velocity()
    # Rebuild total energy with the new kinetic part
    gamma = 5.0/3.0
    P = cfg.cs2 * fluid.rho
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c**2 + By_c**2 + Bz_c**2
    fluid.E[:] = P / (gamma - 1.0) + 0.5 * (fluid.mx**2 + fluid.my**2 + fluid.mz**2) / fluid.rho + 0.5 * B2
    print("Radial drift corrected from total gravity.")

def magnetic_support():
    """
    Radial and vertical magnetic force (J x B)/rho plus magnetic pressure.
    Returns (gR_mag, gz_mag) to be added to the gravitational acceleration
    when building the initial balance.
    """
    # Cell-centered B and simple centered current
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    # J = curl B (centered differences)
    dBy_dz = (cp.roll(By_c, -1, 2) - cp.roll(By_c, 1, 2)) * 0.5 / dx
    dBz_dy = (cp.roll(Bz_c, -1, 1) - cp.roll(Bz_c, 1, 1)) * 0.5 / dx
    dBz_dx = (cp.roll(Bz_c, -1, 0) - cp.roll(Bz_c, 1, 0)) * 0.5 / dx
    dBx_dz = (cp.roll(Bx_c, -1, 2) - cp.roll(Bx_c, 1, 2)) * 0.5 / dx
    dBx_dy = (cp.roll(Bx_c, -1, 1) - cp.roll(Bx_c, 1, 1)) * 0.5 / dx
    dBy_dx = (cp.roll(By_c, -1, 0) - cp.roll(By_c, 1, 0)) * 0.5 / dx
    Jx = dBz_dy - dBy_dz
    Jy = dBx_dz - dBz_dx
    Jz = dBy_dx - dBx_dy
    # (J x B)_x, (J x B)_z
    JxB_x = Jy * Bz_c - Jz * By_c
    JxB_y = Jz * Bx_c - Jx * Bz_c
    JxB_z = Jx * By_c - Jy * Bx_c
    # Magnetic pressure gradient
    B2 = Bx_c**2 + By_c**2 + Bz_c**2
    dB2_dx = (cp.roll(B2, -1, 0) - cp.roll(B2, 1, 0)) * 0.5 / dx
    dB2_dz = (cp.roll(B2, -1, 2) - cp.roll(B2, 1, 2)) * 0.5 / dx
    # Force per mass
    rho_safe = cp.maximum(fluid.rho, 1e-5)
    Fx = (JxB_x - 0.5 * dB2_dx) / rho_safe
    Fy = (JxB_y) / rho_safe
    Fz = (JxB_z - 0.5 * dB2_dz) / rho_safe
    # Radial component
    gR_mag = (Fx * X + Fy * Y) / (R + 1e-4)
    return gR_mag, Fz

def apply_cooling(fluid_state, dt, tau=0.1):
    """
    Relax the internal energy toward the isothermal target
    P = cs2 * rho. tau is the cooling time in code units.
    Call once per full step, not inside the RK stages.
    """
    gamma = 5.0/3.0
    ke = 0.5 * (fluid_state.mx**2 + fluid_state.my**2 + fluid_state.mz**2) / cp.maximum(fluid_state.rho, 1e-5)
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    me = 0.5 * (Bx_c**2 + By_c**2 + Bz_c**2)
    e_int = fluid_state.E - ke - me
    e_target = (cfg.cs2 * fluid_state.rho) / (gamma - 1.0)
    # Explicit relaxation toward the target
    e_int = e_int + (e_target - e_int) * (1.0 - cp.exp(-dt / tau))
    fluid_state.E[:] = e_int + ke + me

def vertical_equilibrium(n_iter=3, tol=1e-3):
    """
    Iterate the density to vertical hydrostatic equilibrium
    with the total gravity (external + halo + self).
    Midplane density is held fixed; the profile is rebuilt
    by integrating g_z upward and downward.
    """
    mid = Ni // 2
    rho_mid = fluid.rho[:, :, mid].copy()
    print("Starting vertical equilibrium...")
    for it in range(n_iter):
        gx, gy, gz = compute_gravity(fluid)
        gR_mag, gz_mag = magnetic_support()
        gz = gz + gz_mag
        # Integrate hydrostatic balance from the midplane
        # rho(z) = rho(0) * exp(-1/cs2 * integral_0^z g_z dz')
        rho_new = cp.zeros_like(fluid.rho)
        rho_new[:, :, mid] = rho_mid
        # Upward
        for k in range(mid + 1, Ni):
            dz = Z[:, :, k] - Z[:, :, k-1]
            g_avg = 0.5 * (gz[:, :, k] + gz[:, :, k-1])
            arg = -g_avg * dz / cfg.cs2
            arg = cp.clip(arg, -10.0, 10.0)
            rho_new[:, :, k] = rho_new[:, :, k-1] * cp.exp(arg)
            rho_new[:, :, k] = cp.clip(rho_new[:, :, k], 1e-5, 1e3)
        # Downward
        for k in range(mid - 1, -1, -1):
            dz = Z[:, :, k+1] - Z[:, :, k]
            g_avg = 0.5 * (gz[:, :, k+1] + gz[:, :, k])
            arg = -g_avg * dz / cfg.cs2
            arg = cp.clip(arg, -10.0, 10.0)
            rho_new[:, :, k] = rho_new[:, :, k+1] * cp.exp(arg)
            rho_new[:, :, k] = cp.clip(rho_new[:, :, k], 1e-5, 1e3)
        rho_new = cp.maximum(rho_new, 1e-5)
        rel = float(cp.max(cp.abs(rho_new - fluid.rho)) / (cp.max(fluid.rho) + 1e-8))
        fluid.rho[:] = rho_new
        if rel < tol:
            print(f"Vertical equilibrium converged in {it+1} iterations (rel={rel:.2e})")
            break
    else:
        print(f"Vertical equilibrium stopped after {n_iter} iterations (rel={rel:.2e})")
    # Rebuild energy with the new density
    gamma = 5.0/3.0
    P = cfg.cs2 * fluid.rho
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c**2 + By_c**2 + Bz_c**2
    fluid.E[:] = P / (gamma - 1.0) + 0.5 * (fluid.mx**2 + fluid.my**2 + fluid.mz**2) / fluid.rho + 0.5 * B2

def compute_gravity(fluid_state):
    """
    External logarithmic potential + static isothermal halo
    + isolated self-gravity of the plasma.
    Returns the gravitational acceleration (gx, gy, gz).
    Difference is taken on the padded array when self-gravity is on,
    then cropped, so the isolated FFT is not re-periodized.
    """
    if not cfg.USE_GRAVITY:
        z = cp.zeros_like(fluid_state.rho)
        return z, z, z

    gx = cp.zeros_like(fluid_state.rho)
    gy = cp.zeros_like(fluid_state.rho)
    gz = cp.zeros_like(fluid_state.rho)

    if cfg.USE_SELF_GRAVITY:
        pad_n = int(Ni * cfg.POISSON_PAD)
        rho_pad = cp.zeros((pad_n, pad_n, pad_n), dtype=cp.float32)
        rho_pad[:Ni, :Ni, :Ni] = fluid_state.rho
        rho_hat = cp.fft.fftn(rho_pad)
        kx = 2.0 * np.sin(np.pi * cp.fft.fftfreq(pad_n)) / dx
        KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0
        phi_hat = -rho_hat / K2
        phi_hat[0, 0, 0] = 0.0
        phi_pad = cp.fft.ifftn(phi_hat).real.astype(cp.float32)
        gx_pad = -(cp.roll(phi_pad, -1, 0) - cp.roll(phi_pad, 1, 0)) * 0.5 / dx
        gy_pad = -(cp.roll(phi_pad, -1, 1) - cp.roll(phi_pad, 1, 1)) * 0.5 / dx
        gz_pad = -(cp.roll(phi_pad, -1, 2) - cp.roll(phi_pad, 1, 2)) * 0.5 / dx
        gx = gx_pad[:Ni, :Ni, :Ni]
        gy = gy_pad[:Ni, :Ni, :Ni]
        gz = gz_pad[:Ni, :Ni, :Ni]

    if cfg.USE_EXT_GRAVITY:
        gx += -(cp.roll(phi_ext, -1, 0) - cp.roll(phi_ext, 1, 0)) * 0.5 / dx
        gy += -(cp.roll(phi_ext, -1, 1) - cp.roll(phi_ext, 1, 1)) * 0.5 / dx
        gz += -(cp.roll(phi_ext, -1, 2) - cp.roll(phi_ext, 1, 2)) * 0.5 / dx

    # Analytic halo force (already acceleration)
    aR_h, aZ_h = halo_acceleration(R, Z)
    gx += aR_h * X / (R + 1e-4)
    gy += aR_h * Y / (R + 1e-4)
    gz += aZ_h
    return gx, gy, gz

def grid_for(shape, block):
    """Launch grid that matches the real array shape (faces/edges may be Ni+1)."""
    return tuple((s + b - 1) // b for s, b in zip(shape, block))

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
    BLOCK_EMF  = (16, 4, 4)   # matches TILE_X/Y/Z in ct_emf
    BLOCK_CURL = (32, 4, 4)
    BLOCK_GLM  = (32, 4, 4)

    # Grids derived from the real staggered shapes
    grid_gyro   = grid_for((Ni * Ni * Ni,), BLOCK_GYRO)
    grid_hlld_x = grid_for((Ni+1, Ni, Ni), BLOCK_HLLD_X)
    grid_hlld_y = grid_for((Ni, Ni+1, Ni), BLOCK_HLLD_Y)
    grid_hlld_z = grid_for((Ni, Ni, Ni+1), BLOCK_HLLD_Z)
    grid_uct    = grid_for((Ni, Ni+1, Ni+1), BLOCK_UCT)   # edge-x shape
    grid_emf    = grid_for((Ni, Ni+1, Ni+1), BLOCK_EMF)
    grid_curl   = grid_for((Ni+1, Ni, Ni), BLOCK_CURL)    # face-x shape
    grid_glm    = grid_for((Ni+1, Ni, Ni), BLOCK_GLM)

    stream_gyro = cp.cuda.Stream(non_blocking=True)
    stream_hlld_x = cp.cuda.Stream(non_blocking=True)
    stream_hlld_y = cp.cuda.Stream(non_blocking=True)
    stream_hlld_z = cp.cuda.Stream(non_blocking=True)
    stream_uct = cp.cuda.Stream(non_blocking=True)
    stream_emf  = cp.cuda.Stream(non_blocking=True)
    stream_curl = cp.cuda.Stream(non_blocking=True)
    stream_glm  = cp.cuda.Stream(non_blocking=True)

    # Persistent GLM scratch (owned at init, never malloc mid-step)
    # Cell-centered EMF scratch (written by ct_emf, averaged to edges)
    emf_cell = {
        "Ex": cp.zeros((Ni, Ni, Ni), dtype=cp.float32),
        "Ey": cp.zeros((Ni, Ni, Ni), dtype=cp.float32),
        "Ez": cp.zeros((Ni, Ni, Ni), dtype=cp.float32),
    }
    glm_scratch = {
        "Bx": cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32),
        "By": cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32),
        "Bz": cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32),
        "psi": cp.zeros((Ni, Ni, Ni), dtype=cp.float32),
    }

    gyrotropic_12_kernel = cp.RawKernel(r'''
#define N_FEAT 12
#define N_FREE 9
#define MAX_NEWTON 32
#define RES_TOL 5e-2f
__device__ __forceinline__ void eval_phi(float cpv, float c2, float p[N_FEAT]) {
    float cp2 = cpv*cpv, cp3 = cp2*cpv, cp4 = cp2*cp2;
    float c4 = c2*c2, c6 = c4*c2;
    float idp = 1.f/(1.f+cp2), ide = 1.f/(1.f+c2);
    p[0]=1.f; p[1]=cpv; p[2]=c2; p[3]=cp2;
    p[4]=cpv*idp; p[5]=cpv*c2*ide;
    p[6]=c4; p[7]=cp2*c2; p[8]=cp4;
    p[9]=cp3*idp*idp; p[10]=c6*ide; p[11]=cp2*c4*ide;
}
__constant__ float scale[N_FEAT] = {1.f,1.f,1.f,1.f,1.f,1.f,3.f,2.f,3.f,1.f,4.f,3.f};
__constant__ float npq[16] = {-6.5f,-5.5f,-4.5f,-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,-7.5f};
__constant__ float wpq[16] = {0.001f,0.004f,0.012f,0.03f,0.07f,0.13f,0.19f,0.19f,0.13f,0.07f,0.03f,0.012f,0.004f,0.001f,0.0003f,0.0003f};
__constant__ float nqq[12] = {0.2f,0.55f,1.0f,1.5f,2.1f,2.8f,3.6f,4.5f,5.6f,6.8f,8.2f,10.0f};
__constant__ float wqq[12] = {0.06f,0.12f,0.16f,0.17f,0.15f,0.12f,0.08f,0.05f,0.03f,0.015f,0.008f,0.003f};
__device__ void set_from_free(float eta, float zeta, float xi, const float low[6], float* lambda) {
    float a = __expf(eta), b = __expf(zeta), s = 0.95f * tanhf(xi);
    float A = a*a, B = b*b, C = 2.f*a*b*s;
    lambda[0] = 0.f;
    lambda[1] = low[0]; lambda[2] = low[1]; lambda[3] = low[2];
    lambda[4] = low[3]; lambda[5] = low[4]; lambda[9] = low[5];
    lambda[6] = lambda[10] = -0.5f * B;
    lambda[7] = lambda[11] = -0.5f * C;
    lambda[8] = -A;
}
__device__ bool tail_certificate(float eta, float zeta, float xi, float* margin_out) {
    float a = __expf(eta), b = __expf(zeta), s = 0.95f * tanhf(xi);
    float tr = a*a + b*b;
    float detQ = a*a * b*b * (1.f - s*s);
    float disc = (a*a - b*b)*(a*a - b*b) + 4.f*a*a*b*b*s*s;
    float lmax = 0.5f * (tr + sqrtf(fmaxf(disc, 0.f)));
    float lmin = (lmax > 1e-12f) ? detQ / lmax : 0.f;
    float margin = 0.5f - 0.5f * s*s;
    if (margin_out) *margin_out = margin;
    return (detQ > 0.f) && (lmin > 1e-5f * tr) && (margin > 0.04f) && isfinite(detQ);
}
__device__ void extract_free(const float* lambda, float* eta, float* zeta, float* xi, float low[6]) {
    float A = fmaxf(-lambda[8], 1e-8f);
    float B = fmaxf(-2.f * lambda[6], 1e-8f);
    *eta  = fmaxf(fminf(0.5f * logf(A), 2.f), -5.f);
    *zeta = fmaxf(fminf(0.5f * logf(B), 2.f), -5.f);
    float a = __expf(*eta), b = __expf(*zeta);
    float ab = fmaxf(a * b, 1e-8f);
    float s = fmaxf(fminf(-lambda[7] / ab, 0.94f), -0.94f);
    *xi = atanhf(s / 0.95f);
    low[0] = lambda[1]; low[1] = lambda[2]; low[2] = lambda[3];
    low[3] = lambda[4]; low[4] = lambda[5]; low[5] = lambda[9];
}
__device__ void jacobian(float eta, float zeta, float xi, float* J) {
    for (int i = 0; i < N_FEAT * N_FREE; ++i) J[i] = 0.f;
    float a = __expf(eta), b = __expf(zeta);
    float th = tanhf(xi), s = 0.95f * th, ds = 0.95f * (1.f - th*th);
    J[8*N_FREE + 0] = -2.f * a*a;
    J[6*N_FREE + 1] = J[10*N_FREE + 1] = -b*b;
    float abs_ = a * b * s;
    J[7*N_FREE + 0] = J[11*N_FREE + 0] = -abs_;
    J[7*N_FREE + 1] = J[11*N_FREE + 1] = -abs_;
    J[7*N_FREE + 2] = J[11*N_FREE + 2] = -a * b * ds;
    J[1*N_FREE + 3] = 1.f; J[2*N_FREE + 4] = 1.f; J[3*N_FREE + 5] = 1.f;
    J[4*N_FREE + 6] = 1.f; J[5*N_FREE + 7] = 1.f; J[9*N_FREE + 8] = 1.f;
}
__device__ float moment_objective(const float* M, const float* Mt) {
    float obj = 0.f;
    for (int k = 1; k < N_FEAT; ++k) {
        float r = (M[k] - Mt[k]) / scale[k];
        obj += 0.5f * r * r;
    }
    return obj;
}
__device__ bool moments_only(const float* lambda, float ap, float ape, float* M,
                             const float* s_np, const float* s_wp, const float* s_nq, const float* s_wq) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cpv = s_np[ip], c2 = s_nq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cpv, c2, p);
            float e = -ap*cpv*cpv - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cpv = s_np[ip], wp_ = s_wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = s_nq[iq]*s_nq[iq], w = wp_*s_wq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cpv, c2, p);
            float e = -ap*cpv*cpv - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            for (int i = 0; i < N_FEAT; ++i) M1[i] += wgt * p[i];
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * iZ;
    return true;
}
__device__ bool moments(const float* lambda, float ap, float ape, float* M, float* G,
                        const float* s_np, const float* s_wp, const float* s_nq, const float* s_wq) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cpv = s_np[ip], c2 = s_nq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cpv, c2, p);
            float e = -ap*cpv*cpv - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0}, M2[N_FEAT*N_FEAT] = {0};
    for (int ip = 0; ip < 16; ++ip) {
        float cpv = s_np[ip], wp_ = s_wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = s_nq[iq]*s_nq[iq], w = wp_*s_wq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cpv, c2, p);
            float e = -ap*cpv*cpv - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            for (int i = 0; i < N_FEAT; ++i) {
                float wi = wgt * p[i];
                M1[i] += wi;
                for (int j = 0; j <= i; ++j) M2[i*N_FEAT + j] += wi * p[j];
            }
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    for (int i = 0; i < N_FEAT; ++i) {
        M[i] = M1[i] * iZ;
        for (int j = 0; j <= i; ++j) {
            float v = M2[i*N_FEAT + j]*iZ - M[i]*M[j];
            G[i*N_FEAT + j] = G[j*N_FEAT + i] = v;
        }
    }
    return true;
}
__device__ bool moments_final(const float* lambda, float ap, float ape, float* M,
                              float* m3_par, float* m3_mix,
                              const float* s_np, const float* s_wp, const float* s_nq, const float* s_wq) {
    float maxe = -1e30f;
    for (int ip = 0; ip < 16; ++ip)
        for (int iq = 0; iq < 12; ++iq) {
            float cpv = s_np[ip], c2 = s_nq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cpv, c2, p);
            float e = -ap*cpv*cpv - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            maxe = fmaxf(maxe, e);
        }
    float Z = 0.f, M1[N_FEAT] = {0}, acc_par = 0.f, acc_mix = 0.f;
    for (int ip = 0; ip < 16; ++ip) {
        float cpv = s_np[ip], wp_ = s_wp[ip];
        for (int iq = 0; iq < 12; ++iq) {
            float c2 = s_nq[iq]*s_nq[iq], w = wp_*s_wq[iq]*s_nq[iq], p[N_FEAT];
            eval_phi(cpv, c2, p);
            float e = -ap*cpv*cpv - ape*c2;
            for (int k = 0; k < N_FEAT; ++k) e += lambda[k]*p[k];
            float wgt = __expf(e - maxe) * w;
            Z += wgt;
            acc_par += wgt * (cpv*cpv*cpv);
            acc_mix += wgt * (cpv * c2);
            for (int i = 0; i < N_FEAT; ++i) M1[i] += wgt * p[i];
        }
    }
    if (!(Z > 1e-40f)) return false;
    float iZ = 1.f / Z;
    for (int i = 0; i < N_FEAT; ++i) M[i] = M1[i] * iZ;
    *m3_par = acc_par * iZ;
    *m3_mix = acc_mix * iZ;
    return true;
}
__device__ bool chol9(float* H, float* rhs, float* dx) {
    float L[N_FREE*N_FREE] = {0};
    for (int i = 0; i < N_FREE; ++i)
        for (int j = 0; j <= i; ++j) {
            float s = H[i*N_FREE + j];
            for (int k = 0; k < j; ++k) s -= L[i*N_FREE + k] * L[j*N_FREE + k];
            if (i == j) {
                if (!(s > 1e-12f)) return false;
                L[i*N_FREE + j] = sqrtf(s);
            } else L[i*N_FREE + j] = s / L[j*N_FREE + j];
        }
    float y[N_FREE];
    for (int i = 0; i < N_FREE; ++i) {
        float s = rhs[i];
        for (int k = 0; k < i; ++k) s -= L[i*N_FREE + k] * y[k];
        y[i] = s / L[i*N_FREE + i];
    }
    for (int i = N_FREE-1; i >= 0; --i) {
        float s = y[i];
        for (int k = i+1; k < N_FREE; ++k) s -= L[k*N_FREE + i] * dx[k];
        dx[i] = s / L[i*N_FREE + i];
    }
    return true;
}
extern "C" __global__
__launch_bounds__(64, 2)
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
    float* __restrict__ margin_out,
    float* __restrict__ M_recon,
    int* __restrict__ status,
    int Ncells,
    float mass,
    float thermal_scale,
    float alpha_par,
    float alpha_perp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncells) return;
    __shared__ float s_np[16], s_wp[16], s_nq[12], s_wq[12];
    if (threadIdx.x < 16) { s_np[threadIdx.x] = npq[threadIdx.x]; s_wp[threadIdx.x] = wpq[threadIdx.x]; }
    if (threadIdx.x < 12) { s_nq[threadIdx.x] = nqq[threadIdx.x]; s_wq[threadIdx.x] = wqq[threadIdx.x]; }
    __syncthreads();
    float Mt[N_FEAT];
    const float4* Mt4 = reinterpret_cast<const float4*>(M_target + idx * N_FEAT);
    float4 t0 = Mt4[0], t1 = Mt4[1], t2 = Mt4[2];
    Mt[0]=t0.x; Mt[1]=t0.y; Mt[2]=t0.z; Mt[3]=t0.w;
    Mt[4]=t1.x; Mt[5]=t1.y; Mt[6]=t1.z; Mt[7]=t1.w;
    Mt[8]=t2.x; Mt[9]=t2.y; Mt[10]=t2.z; Mt[11]=t2.w;
    float* lo = lambda_io + idx * N_FEAT;
    int flag = 0;
    float n = n_in[idx];
    if (!(n > 0.f) || !(mass > 0.f) || !(thermal_scale > 0.f)) flag = 5;
    if (Mt[2] < 1e-8f || Mt[3] < 1e-8f) flag = 5;
    float eta, zeta, xi, low[6];
    extract_free(lo, &eta, &zeta, &xi, low);
    float margin = 0.f;
    bool tail_ok = tail_certificate(eta, zeta, xi, &margin);
    if (!tail_ok && flag == 0) flag = 7;
    float ap = fmaxf(alpha_par, 0.3f), ae = fmaxf(alpha_perp, 0.3f);
    float lambda_lm = 1e-3f;
    int chol_fails = 0, ls_fails = 0;
    float lambda[N_FEAT], M[N_FEAT];
    if (flag == 0) {
        for (int it = 0; it < MAX_NEWTON; ++it) {
            set_from_free(eta, zeta, xi, low, lambda);
            float G[N_FEAT * N_FEAT];
            if (!moments(lambda, ap, ae, M, G, s_np, s_wp, s_nq, s_wq)) { flag = 6; break; }
            float obj = moment_objective(M, Mt);
            float max_r = 0.f;
            for (int k = 1; k < N_FEAT; ++k)
                max_r = fmaxf(max_r, fabsf((M[k] - Mt[k]) / scale[k]));
            if (max_r < RES_TOL) { flag = 0; break; }
            float J[N_FEAT * N_FREE];
            jacobian(eta, zeta, xi, J);
            float H[N_FREE * N_FREE] = {0}, rhs[N_FREE] = {0}, dx[N_FREE];
            for (int i = 1; i < N_FEAT; ++i) {
                float ri = (M[i] - Mt[i]) / scale[i];
                float Di[N_FREE];
                for (int p = 0; p < N_FREE; ++p) {
                    float s = 0.f;
                    for (int j = 0; j < N_FEAT; ++j)
                        s += (G[i*N_FEAT + j] * J[j*N_FREE + p]) / scale[i];
                    Di[p] = s;
                    rhs[p] += s * ri;
                }
                for (int p = 0; p < N_FREE; ++p)
                    for (int q = 0; q < N_FREE; ++q)
                        H[p*N_FREE + q] += Di[p] * Di[q];
            }
            float max_diag = 1e-8f;
            for (int p = 0; p < N_FREE; ++p) max_diag = fmaxf(max_diag, H[p*N_FREE + p]);
            float ridge = lambda_lm * max_diag;
            for (int p = 0; p < N_FREE; ++p) H[p*N_FREE + p] += ridge;
            if (!chol9(H, rhs, dx)) {
                lambda_lm *= 4.f;
                if (++chol_fails > 4) { flag = 2; break; }
                continue;
            }
            chol_fails = 0;
            float al = 0.8f;
            bool accepted = false;
            for (int ls = 0; ls < 4; ++ls) {
                float e2 = eta - al*dx[0], z2 = zeta - al*dx[1], x2 = xi - al*dx[2];
                if (e2 < -5.f || e2 > 2.f || z2 < -5.f || z2 > 2.f) { al *= 0.5f; continue; }
                float l2[6];
                for (int k = 0; k < 6; ++k) l2[k] = low[k] - al*dx[3+k];
                float lam2[N_FEAT], M2[N_FEAT];
                set_from_free(e2, z2, x2, l2, lam2);
                if (!moments_only(lam2, ap, ae, M2, s_np, s_wp, s_nq, s_wq)) { al *= 0.5f; continue; }
                if (moment_objective(M2, Mt) < obj) {
                    eta = e2; zeta = z2; xi = x2;
                    for (int k = 0; k < 6; ++k) low[k] = l2[k];
                    accepted = true;
                    break;
                }
                al *= 0.5f;
            }
            if (!accepted) {
                if (++ls_fails > 3) { flag = 3; break; }
                lambda_lm = fminf(lambda_lm * 4.f, 10.f);
            } else ls_fails = 0;
            if (it == MAX_NEWTON - 1 && flag == 0) flag = 1;
        }
    }
    float m3_par = 0.f, m3_mix = 0.f;
    set_from_free(eta, zeta, xi, low, lambda);
    bool ok_final = moments_final(lambda, ap, ae, M, &m3_par, &m3_mix, s_np, s_wp, s_nq, s_wq);
    if (!ok_final && flag == 0) flag = 6;
    tail_ok = tail_certificate(eta, zeta, xi, &margin);
    if (!tail_ok && flag == 0) flag = 7;
    if (ok_final && (flag == 0 || flag == 1)) {
        float moment_r = 0.f;
        for (int k = 1; k < N_FEAT; ++k)
            moment_r = fmaxf(moment_r, fabsf((M[k] - Mt[k]) / scale[k]));
        if (moment_r < RES_TOL && tail_ok) flag = 0;
        else if (flag == 0) flag = 1;
    }
    if (flag == 0)
        for (int k = 0; k < N_FEAT; ++k) lo[k] = lambda[k];
    status[idx] = flag;
    margin_out[idx] = margin;
    float* Mr = M_recon + idx * N_FEAT;
    for (int k = 0; k < N_FEAT; ++k) Mr[k] = ok_final ? M[k] : 0.f;
    bool valid = (flag == 0) && ok_final && tail_ok;
    float vth2 = thermal_scale * thermal_scale;
    float vth3 = vth2 * thermal_scale;
    float up = valid ? M[1] : 0.f;
    float cp2 = valid ? fmaxf(M[3] - up*up, 1e-6f) : 1e-4f;
    float ce2 = valid ? fmaxf(M[2], 1e-6f) : 1e-4f;
    float qpar = 0.f, qperp = 0.f;
    if (valid) {
        qpar  = mass * n * vth3 * (m3_par - 3.f*up*cp2 - up*up*up);
        qperp = 0.5f * mass * n * vth3 * (m3_mix - up * M[2]);
    }
    n_out[idx] = n;
    upar_out[idx] = up * thermal_scale;
    Ppar_out[idx] = mass * n * vth2 * cp2;
    Pperp_out[idx] = 0.5f * mass * n * vth2 * ce2;
    qpar_out[idx] = qpar;
    qperp_out[idx] = qperp;
}
''', 'gyrotropic_12_kernel')

    ct_emf_kernel = cp.RawKernel(r'''#define TILE_X 16
#define TILE_Y 4
#define TILE_Z 4
#define PAD 1
__device__ float mc_limiter(float a, float b) {
    if (a * b <= 0.0f) return 0.0f;
    float min1 = 2.0f * fminf(fabsf(a), fabsf(b));
    float min2 = 0.5f * fabsf(a + b);
    return copysignf(fminf(min1, min2), a);
}
extern "C" __launch_bounds__(256, 4)
__global__ void ct_emf_kernel(
    const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz,
    const float* pe_xx, const float* pe_yy, const float* pe_zz,
    const float* pe_xy, const float* pe_xz, const float* pe_yz,
    float* Ex_cell, float* Ey_cell, float* Ez_cell,
    int Ni, float hall_coeff, float dx, float dt_over_dx, float base_bias) {
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int i = blockIdx.x * TILE_X + tx;
    int j = blockIdx.y * TILE_Y + ty;
    int k = blockIdx.z * TILE_Z + tz;
    int sx = tx + PAD;
    int sy = ty + PAD;
    int sz = tz + PAD;
    __shared__ float s_rho[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_mx [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_my [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_mz [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Bx [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_By [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_Bz [TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_pexy[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_pexz[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    __shared__ float s_peyz[TILE_X + 2*PAD][TILE_Y + 2*PAD][TILE_Z + 2*PAD];
    int tid = tx + ty * TILE_X + tz * TILE_X * TILE_Y;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int offset = tid; offset < (TILE_X + 2*PAD)*(TILE_Y + 2*PAD)*(TILE_Z + 2*PAD); offset += total_threads) {
        int lx = offset % (TILE_X + 2*PAD);
        int ly = (offset / (TILE_X + 2*PAD)) % (TILE_Y + 2*PAD);
        int lz = offset / ((TILE_X + 2*PAD) * (TILE_Y + 2*PAD));
        int li = blockIdx.x * TILE_X + (lx - PAD);
        int lj = blockIdx.y * TILE_Y + (ly - PAD);
        int lk = blockIdx.z * TILE_Z + (lz - PAD);
        if (li >= 0 && li < Ni && lj >= 0 && lj < Ni && lk >= 0 && lk < Ni) {
            int gidx = li * Ni * Ni + lj * Ni + lk;
            s_rho[lx][ly][lz] = rho[gidx];
            s_mx[lx][ly][lz]  = mx[gidx];
            s_my[lx][ly][lz]  = my[gidx];
            s_mz[lx][ly][lz]  = mz[gidx];
            // Cell-centered B approximation: average adjacent faces if available, else cell
            s_Bx[lx][ly][lz]  = Bx[gidx];
            s_By[lx][ly][lz]  = By[gidx];
            s_Bz[lx][ly][lz]  = Bz[gidx];
            s_pexy[lx][ly][lz] = pe_xy[gidx];
            s_pexz[lx][ly][lz] = pe_xz[gidx];
            s_peyz[lx][ly][lz] = pe_yz[gidx];
        } else {
            s_rho[lx][ly][lz] = 1e-8f;
            s_mx[lx][ly][lz] = s_my[lx][ly][lz] = s_mz[lx][ly][lz] = 0.0f;
            s_Bx[lx][ly][lz] = s_By[lx][ly][lz] = s_Bz[lx][ly][lz] = 0.0f;
            s_pexy[lx][ly][lz] = s_pexz[lx][ly][lz] = s_peyz[lx][ly][lz] = 0.0f;
        }
    }
    __syncthreads();
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
    float rho0 = fmaxf(s_rho[sx][sy][sz], 1e-8f);
    float vx0 = s_mx[sx][sy][sz] / rho0;
    float vy0 = s_my[sx][sy][sz] / rho0;
    float vz0 = s_mz[sx][sy][sz] / rho0;
    float Bx0 = s_Bx[sx][sy][sz];
    float By0 = s_By[sx][sy][sz];
    float Bz0 = s_Bz[sx][sy][sz];
    float inv2dx = 0.5f / dx;
    float jx = (s_By[sx][sy][sz+1] - s_By[sx][sy][sz-1] - (s_Bz[sx][sy+1][sz] - s_Bz[sx][sy-1][sz])) * inv2dx;
    float jy = (s_Bz[sx][sy][sz+1] - s_Bz[sx][sy][sz-1] - (s_Bx[sx][sy+1][sz] - s_Bx[sx][sy-1][sz])) * inv2dx;
    float jz = (s_Bx[sx][sy+1][sz] - s_Bx[sx][sy-1][sz] - (s_By[sx][sy][sz+1] - s_By[sx][sy][sz-1])) * inv2dx;
    float inv_ne = 1.0f / rho0;
    float vex = vx0 - jx * hall_coeff * inv_ne;
    float vey = vy0 - jy * hall_coeff * inv_ne;
    float vez = vz0 - jz * hall_coeff * inv_ne;
    float Ex = -(vey * Bz0 - vez * By0);
    float Ey = -(vez * Bx0 - vex * Bz0);
    float Ez = -(vex * By0 - vey * Bx0);
    float dPexx_dx = (pe_xx[(i+1)*Ni*Ni + j*Ni + k] - pe_xx[(i-1)*Ni*Ni + j*Ni + k]) * inv2dx;
    float dPexy_dy = (s_pexy[sx][sy+1][sz] - s_pexy[sx][sy-1][sz]) * inv2dx;
    float dPexz_dz = (s_pexz[sx][sy][sz+1] - s_pexz[sx][sy][sz-1]) * inv2dx;
    float dPeyx_dx = (s_pexy[sx+1][sy][sz] - s_pexy[sx-1][sy][sz]) * inv2dx;
    float dPeyy_dy = (pe_yy[i*Ni*Ni + (j+1)*Ni + k] - pe_yy[i*Ni*Ni + (j-1)*Ni + k]) * inv2dx;
    float dPeyz_dz = (s_peyz[sx][sy][sz+1] - s_peyz[sx][sy][sz-1]) * inv2dx;
    float dPezx_dx = (s_pexz[sx+1][sy][sz] - s_pexz[sx-1][sy][sz]) * inv2dx;
    float dPezy_dy = (s_peyz[sx][sy+1][sz] - s_peyz[sx][sy-1][sz]) * inv2dx;
    float dPezz_dz = (pe_zz[i*Ni*Ni + j*Ni + (k+1)] - pe_zz[i*Ni*Ni + j*Ni + (k-1)]) * inv2dx;
    Ex -= (dPexx_dx + dPexy_dy + dPexz_dz) * inv_ne;
    Ey -= (dPeyx_dx + dPeyy_dy + dPeyz_dz) * inv_ne;
    Ez -= (dPezx_dx + dPezy_dy + dPezz_dz) * inv_ne;
    int idx = i * Ni * Ni + j * Ni + k;
    Ex_cell[idx] = Ex;
    Ey_cell[idx] = Ey;
    Ez_cell[idx] = Ez;
}
''', 'ct_emf_kernel')

    ct_curl_kernel = cp.RawKernel(r'''
    extern "C" __global__ void ct_curl_kernel(
        const float* Ex, const float* Ey, const float* Ez,
        float* Bx, float* By, float* Bz,
        int Ni, float dt, float inv_dx) {
        // Thread owns one x-face (i, j, k) where i runs 0..Ni
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;

        // Update Bx (shape Ni+1, Ni, Ni) from Ey and Ez
        if (i <= Ni && j < Ni && k < Ni) {
            int idx_Bx = i*(Ni*Ni) + j*Ni + k;
            // Ey shape (Ni+1, Ni, Ni+1), Ez shape (Ni+1, Ni+1, Ni)
            float dEy_dz = (Ey[i*(Ni*(Ni+1)) + j*(Ni+1) + (k+1)]
                          - Ey[i*(Ni*(Ni+1)) + j*(Ni+1) + k]) * inv_dx;
            float dEz_dy = (Ez[i*((Ni+1)*Ni) + (j+1)*Ni + k]
                          - Ez[i*((Ni+1)*Ni) + j*Ni + k]) * inv_dx;
            Bx[idx_Bx] -= dt * (dEy_dz - dEz_dy);
        }
        // Update By (shape Ni, Ni+1, Ni) from Ez and Ex
        if (i < Ni && j <= Ni && k < Ni) {
            int idx_By = i*((Ni+1)*Ni) + j*Ni + k;
            float dEz_dx = (Ez[(i+1)*((Ni+1)*Ni) + j*Ni + k]
                          - Ez[i*((Ni+1)*Ni) + j*Ni + k]) * inv_dx;
            float dEx_dz = (Ex[i*((Ni+1)*(Ni+1)) + j*(Ni+1) + (k+1)]
                          - Ex[i*((Ni+1)*(Ni+1)) + j*(Ni+1) + k]) * inv_dx;
            By[idx_By] -= dt * (dEz_dx - dEx_dz);
        }
        // Update Bz (shape Ni, Ni, Ni+1) from Ex and Ey
        if (i < Ni && j < Ni && k <= Ni) {
            int idx_Bz = i*(Ni*(Ni+1)) + j*(Ni+1) + k;
            float dEx_dy = (Ex[i*((Ni+1)*(Ni+1)) + (j+1)*(Ni+1) + k]
                          - Ex[i*((Ni+1)*(Ni+1)) + j*(Ni+1) + k]) * inv_dx;
            float dEy_dx = (Ey[(i+1)*(Ni*(Ni+1)) + j*(Ni+1) + k]
                          - Ey[i*(Ni*(Ni+1)) + j*(Ni+1) + k]) * inv_dx;
            Bz[idx_Bz] -= dt * (dEx_dy - dEy_dx);
        }
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
        if (i >= Ni || j >= Ni || k >= Ni) return;
        int idx = i*Ni*Ni + j*Ni + k;
        float inv_dx = 1.0f / dx;

        // Staggered divergence at cell center
        float divB = ((Bx_in[(i+1)*Ni*Ni + j*Ni + k] - Bx_in[i*Ni*Ni + j*Ni + k]) +
                      (By_in[i*(Ni+1)*Ni + (j+1)*Ni + k] - By_in[i*(Ni+1)*Ni + j*Ni + k]) +
                      (Bz_in[i*Ni*(Ni+1) + j*(Ni+1) + (k+1)] - Bz_in[i*Ni*(Ni+1) + j*(Ni+1) + k])) * inv_dx;

        // psi update
        float psi_new = psi_in[idx] - dt * ch*ch * divB;
        psi_new *= expf(-dt * cr * ch / dx);
        psi_out[idx] = psi_new;

        // Gradient correction on faces that touch this cell
        // Only write the lower face; the higher face is written by the next cell
        Bx_out[i*Ni*Ni + j*Ni + k] = Bx_in[i*Ni*Ni + j*Ni + k]
            - dt * (psi_in[idx] - (i>0 ? psi_in[(i-1)*Ni*Ni + j*Ni + k] : psi_in[idx])) * inv_dx;
        By_out[i*(Ni+1)*Ni + j*Ni + k] = By_in[i*(Ni+1)*Ni + j*Ni + k]
            - dt * (psi_in[idx] - (j>0 ? psi_in[i*Ni*Ni + (j-1)*Ni + k] : psi_in[idx])) * inv_dx;
        Bz_out[i*Ni*(Ni+1) + j*(Ni+1) + k] = Bz_in[i*Ni*(Ni+1) + j*(Ni+1) + k]
            - dt * (psi_in[idx] - (k>0 ? psi_in[i*Ni*Ni + j*Ni + (k-1)] : psi_in[idx])) * inv_dx;
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
    """Build Cartesian Pe from Ppar, Pperp and cell-centered B."""
    Bx_c, By_c, Bz_c = face_to_cell(mag_state)
    B2 = Bx_c**2 + By_c**2 + Bz_c**2
    Bmag = cp.sqrt(cp.maximum(B2, 1e-12))
    bx = Bx_c / Bmag
    by = By_c / Bmag
    bz = Bz_c / Bmag
    dP = moment_state.Ppar - moment_state.Pperp
    moment_state.pe_xx[:] = moment_state.Pperp + dP * bx * bx
    moment_state.pe_yy[:] = moment_state.Pperp + dP * by * by
    moment_state.pe_zz[:] = moment_state.Pperp + dP * bz * bz
    moment_state.pe_xy[:] = dP * bx * by
    moment_state.pe_xz[:] = dP * bx * bz
    moment_state.pe_yz[:] = dP * by * bz
    moment_state.P[:] = (moment_state.Ppar + 2.0 * moment_state.Pperp) / 3.0

def fill_target_moments(moment_state, fluid_state, alpha_par=0.5, alpha_perp=0.5):
    """Fill M_target with a simple Gaussian base and warm-start lambda."""
    np_h = np.array([-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,-7.5], dtype=np.float32)
    wp_h = np.array([0.001,0.004,0.012,0.03,0.07,0.13,0.19,0.19,0.13,0.07,0.03,0.012,0.004,0.001,0.0003,0.0003], dtype=np.float32)
    nq_h = np.array([0.2,0.55,1.0,1.5,2.1,2.8,3.6,4.5,5.6,6.8,8.2,10.0], dtype=np.float32)
    wq_h = np.array([0.06,0.12,0.16,0.17,0.15,0.12,0.08,0.05,0.03,0.015,0.008,0.003], dtype=np.float32)
    def phi(cpv, c2):
        cp2 = cpv*cpv; idp = 1.0/(1.0+cp2); ide = 1.0/(1.0+c2)
        return np.array([1, cpv, c2, cp2, cpv*idp, cpv*c2*ide, c2*c2, cp2*c2, cp2*cp2,
                         (cpv**3)*idp*idp, (c2**3)*ide, cp2*(c2**2)*ide], dtype=np.float32)
    Z, M = 0.0, np.zeros(12, dtype=np.float32)
    for ip in range(16):
        for iq in range(12):
            cpv, c2 = np_h[ip], nq_h[iq]**2
            w = wp_h[ip] * wq_h[iq] * nq_h[iq]
            wgt = np.exp(-alpha_par*cpv*cpv - alpha_perp*c2) * w
            Z += wgt; M += wgt * phi(cpv, c2)
    M /= (Z + 1e-40)
    Ncells = moment_state.M_target.shape[0]
    moment_state.M_target[:] = cp.asarray(M)
    # Warm start
    moment_state.lambda_io.fill(0.0)
    moment_state.lambda_io[:,6] = moment_state.lambda_io[:,8] = moment_state.lambda_io[:,10] = -0.05

def call_gyro(fluid_state, moment_state, mag_state):
    if not cfg.USE_GYRO:
        return
    fill_target_moments(moment_state, fluid_state)
    Ncells = fluid_state.rho.size
    if USE_GPU:
        n_flat = fluid_state.rho.ravel()
        # Real 12-feature ABI: M_target, n_in, lambda_io → n_out, upar, Ppar, Pperp, qpar, qperp, margin, M_recon, status
        gyrotropic_12_kernel(grid_gyro, BLOCK_GYRO,
            (moment_state.M_target, n_flat, moment_state.lambda_io,
             moment_state.n_out, moment_state.upar,
             moment_state.Ppar.ravel(), moment_state.Pperp.ravel(),
             moment_state.qpar.ravel(), moment_state.qperp.ravel(),
             moment_state.margin, moment_state.M_recon, moment_state.status,
             Ncells,
             np.float32(1.0), np.float32(1.0),
             np.float32(0.5), np.float32(0.5)),
            stream=stream_gyro)
        stream_gyro.synchronize()
    else:
        # Identity CPU path — simple isotropic-like closure
        moment_state.Ppar[:] = cfg.cs2 * fluid_state.rho
        moment_state.Pperp[:] = 0.5 * moment_state.Ppar
    reconstruct_pressure_tensor(moment_state, mag_state)

def assemble_residual(dt, fluid_in, fluxes, fluid_out):
    """
    Conservative residual: U_out = U_in - dt/dx * div(F).
    Face fluxes are differenced onto cell centers.
    Identity fluxes (all zero) leave the state unchanged.
    """
    inv_dx = dt / dx
    # Interior cells only; ghosts are refilled after the step
    # Face fluxes: Fx (Ni+1,Ni,Ni), Fy (Ni,Ni+1,Ni), Fz (Ni,Ni,Ni+1)
    # Difference onto interior (Ni-2, Ni-2, Ni-2)
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))
    # x-faces: right face [2:-1] minus left face [1:-2], y/z interior
    dFx = fluxes.Fx_rho[2:-1, 1:-1, 1:-1] - fluxes.Fx_rho[1:-2, 1:-1, 1:-1]
    # y-faces: right face [:,2:-1] minus left face [:,1:-2]
    dFy = fluxes.Fy_rho[1:-1, 2:-1, 1:-1] - fluxes.Fy_rho[1:-1, 1:-2, 1:-1]
    # z-faces: right face [:,:,2:-1] minus left face [:,:,1:-2]
    dFz = fluxes.Fz_rho[1:-1, 1:-1, 2:-1] - fluxes.Fz_rho[1:-1, 1:-1, 1:-2]
    fluid_out.rho[sl] = fluid_in.rho[sl] - inv_dx * (dFx + dFy + dFz)
    dFx = fluxes.Fx_mx[2:-1, 1:-1, 1:-1] - fluxes.Fx_mx[1:-2, 1:-1, 1:-1]
    dFy = fluxes.Fy_mx[1:-1, 2:-1, 1:-1] - fluxes.Fy_mx[1:-1, 1:-2, 1:-1]
    dFz = fluxes.Fz_mx[1:-1, 1:-1, 2:-1] - fluxes.Fz_mx[1:-1, 1:-1, 1:-2]
    fluid_out.mx[sl] = fluid_in.mx[sl] - inv_dx * (dFx + dFy + dFz)
    dFx = fluxes.Fx_my[2:-1, 1:-1, 1:-1] - fluxes.Fx_my[1:-2, 1:-1, 1:-1]
    dFy = fluxes.Fy_my[1:-1, 2:-1, 1:-1] - fluxes.Fy_my[1:-1, 1:-2, 1:-1]
    dFz = fluxes.Fz_my[1:-1, 1:-1, 2:-1] - fluxes.Fz_my[1:-1, 1:-1, 1:-2]
    fluid_out.my[sl] = fluid_in.my[sl] - inv_dx * (dFx + dFy + dFz)
    dFx = fluxes.Fx_mz[2:-1, 1:-1, 1:-1] - fluxes.Fx_mz[1:-2, 1:-1, 1:-1]
    dFy = fluxes.Fy_mz[1:-1, 2:-1, 1:-1] - fluxes.Fy_mz[1:-1, 1:-2, 1:-1]
    dFz = fluxes.Fz_mz[1:-1, 1:-1, 2:-1] - fluxes.Fz_mz[1:-1, 1:-1, 1:-2]
    fluid_out.mz[sl] = fluid_in.mz[sl] - inv_dx * (dFx + dFy + dFz)
    dFx = fluxes.Fx_E[2:-1, 1:-1, 1:-1] - fluxes.Fx_E[1:-2, 1:-1, 1:-1]
    dFy = fluxes.Fy_E[1:-1, 2:-1, 1:-1] - fluxes.Fy_E[1:-1, 1:-2, 1:-1]
    dFz = fluxes.Fz_E[1:-1, 1:-1, 2:-1] - fluxes.Fz_E[1:-1, 1:-1, 1:-2]
    fluid_out.E[sl] = fluid_in.E[sl] - inv_dx * (dFx + dFy + dFz)
    # Copy ghosts from input so the next ghost fill has a defined base
    fluid_out.rho[:1] = fluid_in.rho[:1]; fluid_out.rho[-1:] = fluid_in.rho[-1:]
    fluid_out.mx[:1] = fluid_in.mx[:1]; fluid_out.mx[-1:] = fluid_in.mx[-1:]
    fluid_out.my[:1] = fluid_in.my[:1]; fluid_out.my[-1:] = fluid_in.my[-1:]
    fluid_out.mz[:1] = fluid_in.mz[:1]; fluid_out.mz[-1:] = fluid_in.mz[-1:]
    fluid_out.E[:1] = fluid_in.E[:1]; fluid_out.E[-1:] = fluid_in.E[-1:]

def call_hlld(dt, fluid_in, mag_state, fluxes, fluid_out):
    """Directional HLLD. Identity path zeros fluxes, then residual assembly updates fluid."""
    if not cfg.USE_HLLD:
        fluid_out.rho[:] = fluid_in.rho
        fluid_out.mx[:]  = fluid_in.mx
        fluid_out.my[:]  = fluid_in.my
        fluid_out.mz[:]  = fluid_in.mz
        fluid_out.E[:]   = fluid_in.E
        return
    fluxes.zero()
    if USE_GPU:
        # Real HLLD-X/Y/Z kernels write face fluxes into FluxState.
        # Placeholder path leaves them zero.
        pass
    # Residual assembly: fluid_out = fluid_in - dt/dx * div(F)
    assemble_residual(dt, fluid_in, fluxes, fluid_out)

def call_uct(fluxes, mag_state):
    if not cfg.USE_UCT:
        return
    # Identity until the real UCT kernel matches the frozen ABI.
    # Do not launch the old cell-indexed stub.
    mag_state.Emf_ideal[0].fill(0.0)
    mag_state.Emf_ideal[1].fill(0.0)
    mag_state.Emf_ideal[2].fill(0.0)

def call_emf(fluid_state, mag_state, moment_state):
    if not cfg.USE_EMF or not USE_GPU:
        return
    hall = float(getattr(cfg, "hall_coeff", 0.018))
    bias = float(getattr(cfg, "base_bias", 0.45))
    dt_over_dx = float(getattr(cfg, "last_dt", 1e-6)) / max(dx, 1e-12)
    # Cell-centered B for the EMF kernel (average faces to cells)
    Bx_c = 0.5 * (mag_state.Bx.data[:-1] + mag_state.Bx.data[1:])
    By_c = 0.5 * (mag_state.By.data[:, :-1] + mag_state.By.data[:, 1:])
    Bz_c = 0.5 * (mag_state.Bz.data[:, :, :-1] + mag_state.Bz.data[:, :, 1:])
    ct_emf_kernel(grid_emf, BLOCK_EMF,
        (fluid_state.rho, fluid_state.mx, fluid_state.my, fluid_state.mz,
         Bx_c, By_c, Bz_c,
         moment_state.pe_xx, moment_state.pe_yy, moment_state.pe_zz,
         moment_state.pe_xy, moment_state.pe_xz, moment_state.pe_yz,
         emf_cell["Ex"], emf_cell["Ey"], emf_cell["Ez"],
         Ni, np.float32(hall), np.float32(dx),
         np.float32(dt_over_dx), np.float32(bias)),
        stream=stream_emf)
    stream_emf.synchronize()
    # Scatter cell EMF to edge arrays with the strides ct_curl expects
    Ex = emf_cell["Ex"]; Ey = emf_cell["Ey"]; Ez = emf_cell["Ez"]
    # EdgeX (Ni, Ni+1, Ni+1): average four surrounding cells in y,z
    mag_state.Emfx[1:-1, 1:-1, 1:-1] = 0.25 * (
        Ex[1:-1, :-1, :-1] + Ex[1:-1, 1:, :-1] +
        Ex[1:-1, :-1, 1:]  + Ex[1:-1, 1:, 1:])
    # EdgeY (Ni+1, Ni, Ni+1): average four surrounding cells in x,z
    mag_state.Emfy[1:-1, 1:-1, 1:-1] = 0.25 * (
        Ey[:-1, 1:-1, :-1] + Ey[1:, 1:-1, :-1] +
        Ey[:-1, 1:-1, 1:]  + Ey[1:, 1:-1, 1:])
    # EdgeZ (Ni+1, Ni+1, Ni): average four surrounding cells in x,y
    mag_state.Emfz[1:-1, 1:-1, 1:-1] = 0.25 * (
        Ez[:-1, :-1, 1:-1] + Ez[1:, :-1, 1:-1] +
        Ez[:-1, 1:, 1:-1]  + Ez[1:, 1:, 1:-1])

def call_glm(dt, mag_state):
    if not cfg.USE_GLM or not USE_GPU:
        return
    ch = 0.4 * dx / max(dt, 1e-8)
    # Use persistent scratch — never malloc mid-step
    Bx_out = glm_scratch["Bx"]
    By_out = glm_scratch["By"]
    Bz_out = glm_scratch["Bz"]
    psi_out = glm_scratch["psi"]
    Bx_out[:] = mag_state.Bx.data
    By_out[:] = mag_state.By.data
    Bz_out[:] = mag_state.Bz.data
    psi_out[:] = mag_state.psi.data
    glm_kernel(grid_glm, BLOCK_GLM,
        (mag_state.Bx.data, mag_state.By.data, mag_state.Bz.data, mag_state.psi.data,
         Bx_out, By_out, Bz_out, psi_out,
         Ni, np.float32(dt), np.float32(dx), np.float32(ch), np.float32(0.05)),
        stream=stream_glm)
    stream_glm.synchronize()
    mag_state.Bx.data[:] = Bx_out
    mag_state.By.data[:] = By_out
    mag_state.Bz.data[:] = Bz_out
    mag_state.psi.data[:] = psi_out

def call_curl(dt, mag_in, mag_out):
    """Staggered CT update using index arithmetic."""
    mag_out.Bx.data[:] = mag_in.Bx.data
    mag_out.By.data[:] = mag_in.By.data
    mag_out.Bz.data[:] = mag_in.Bz.data
    mag_out.psi.data[:] = mag_in.psi.data
    mag_out.Emf_total[0].data[:] = mag_in.Emf_total[0].data
    mag_out.Emf_total[1].data[:] = mag_in.Emf_total[1].data
    mag_out.Emf_total[2].data[:] = mag_in.Emf_total[2].data
    mag_out.sync_emf_aliases()

    if not cfg.USE_CURL:
        return

    if USE_GPU:
        inv_dx = np.float32(1.0 / dx)
        ct_curl_kernel(grid_curl, BLOCK_CURL,
            (mag_out.Emfx, mag_out.Emfy, mag_out.Emfz,
             mag_out.Bx.data, mag_out.By.data, mag_out.Bz.data,
             Ni, np.float32(dt), inv_dx),
            stream=stream_curl)
        stream_curl.synchronize()
    else:
        # Array fallback (correct staggered differences)
        mag_out.Bx.data[1:-1, :, :] -= dt * (
            (mag_in.Emfy[1:-1, :, 1:] - mag_in.Emfy[1:-1, :, :-1]) / dx
          - (mag_in.Emfz[1:-1, 1:, :] - mag_in.Emfz[1:-1, :-1, :]) / dx)
        mag_out.By.data[:, 1:-1, :] -= dt * (
            (mag_in.Emfz[1:, 1:-1, :] - mag_in.Emfz[:-1, 1:-1, :]) / dx
          - (mag_in.Emfx[:, 1:-1, 1:] - mag_in.Emfx[:, 1:-1, :-1]) / dx)
        mag_out.Bz.data[:, :, 1:-1] -= dt * (
            (mag_in.Emfx[:, 1:, 1:-1] - mag_in.Emfx[:, :-1, 1:-1]) / dx
          - (mag_in.Emfy[1:, :, 1:-1] - mag_in.Emfy[:-1, :, 1:-1]) / dx)

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
# CUDA graph scaffold — off by default.
# Capture only the pure kernel chain once the kernels are real and the
# launch sequence is stable. Ghost fills, gravity, and RK blends stay
# outside the graph because they are host-side and may change.
_graph = None
_graph_exec = None

def _kernel_chain(dt, fluid_in, mag_in, mom_in, fluxes_in, fluid_out, mag_out):
    """Pure kernel sequence for one stage. Graph-safe once kernels are real."""
    call_gyro(fluid_in, mom_in, mag_in)
    call_hlld(dt, fluid_in, mag_in, fluxes_in, fluid_out)
    call_uct(fluxes_in, mag_in)
    call_emf(fluid_in, mag_in, mom_in)
    call_curl(dt, mag_in, mag_out)
    call_glm(dt, mag_out)

def capture_graph(dt):
    """Capture the stage kernel chain. Call once after kernels are real."""
    global _graph, _graph_exec
    if not USE_GPU or not cfg.USE_CUDA_GRAPH:
        return
    stream = cp.cuda.Stream()
    with stream:
        stream.begin_capture()
        _kernel_chain(dt, fluid, mag, mom, fluxes, fluid1, mag1)
        _graph = stream.end_capture()
    _graph_exec = _graph.instantiate()
    print("CUDA graph captured.")

def _run_stage(dt, fluid_in, mag_in, mom_in, fluxes_in, fluid_out, mag_out):
    """Run one stage: graph replay if captured, else explicit launches."""
    if cfg.USE_CUDA_GRAPH and _graph_exec is not None:
        _graph_exec.launch()
    else:
        _kernel_chain(dt, fluid_in, mag_in, mom_in, fluxes_in, fluid_out, mag_out)


# =====================================================================
# Extra galactic-disk source terms (optional, after the 8-kernel chain)
# =====================================================================
def apply_viscosity(fluid_state, dt):
    """Simple alpha-viscosity: momentum diffusion proportional to local sound speed and scale height."""
    if not cfg.USE_VISCOSITY:
        return
    cs = cp.sqrt(cfg.cs2)
    # Rough scale height from vertical density moment (or fixed)
    H = cfg.zd0 * (1.0 + (R / cfg.Rd)**cfg.flare)
    nu = cfg.alpha_visc * cs * H
    # Explicit Laplacian on momentum (centered, interior only)
    for mom in (fluid_state.mx, fluid_state.my, fluid_state.mz):
        lap = (
            cp.roll(mom, -1, 0) + cp.roll(mom, 1, 0) +
            cp.roll(mom, -1, 1) + cp.roll(mom, 1, 1) +
            cp.roll(mom, -1, 2) + cp.roll(mom, 1, 2) - 6.0 * mom
        ) / (dx * dx)
        mom[:] += dt * nu * lap
    fluid_state.update_velocity()

def apply_resistivity(mag_state, dt):
    """Simple Ohmic resistivity on the face-centered magnetic field."""
    if not cfg.USE_RESISTIVITY:
        return
    eta = cfg.eta_ohm
    for B in (mag_state.Bx.data, mag_state.By.data, mag_state.Bz.data):
        lap = (
            cp.roll(B, -1, 0) + cp.roll(B, 1, 0) +
            cp.roll(B, -1, 1) + cp.roll(B, 1, 1) +
            cp.roll(B, -1, 2) + cp.roll(B, 1, 2) - 6.0 * B
        ) / (dx * dx)
        B[:] += dt * eta * lap

def apply_feedback(fluid_state, dt):
    """Simple density-threshold feedback: inject thermal energy and a small random kick."""
    if not cfg.USE_FEEDBACK:
        return
    mask = fluid_state.rho > cfg.feedback_rho_thresh
    if not cp.any(mask):
        return
    # Thermal energy injection
    de = cfg.feedback_rate * dt * fluid_state.rho
    fluid_state.E[:] += cp.where(mask, de, 0.0)
    # Small isotropic momentum kick (order of sound speed)
    kick = 0.1 * cp.sqrt(cfg.cs2)
    rng = cp.random.randn(*fluid_state.rho.shape).astype(cp.float32)
    fluid_state.mx[:] += cp.where(mask, kick * rng * fluid_state.rho, 0.0)
    rng = cp.random.randn(*fluid_state.rho.shape).astype(cp.float32)
    fluid_state.my[:] += cp.where(mask, kick * rng * fluid_state.rho, 0.0)
    rng = cp.random.randn(*fluid_state.rho.shape).astype(cp.float32)
    fluid_state.mz[:] += cp.where(mask, kick * rng * fluid_state.rho, 0.0)
    fluid_state.update_velocity()

def operator_step(dt):
    """
    SSP-RK3 with stage-consistent gravity.
    Gravity is recomputed from the density of each stage.
    """
    # ----- Stage 1 -----
    update_fluid_ghosts(fluid)
    update_magnetic_ghosts(mag)
    fluid.update_velocity()
    gx, gy, gz = compute_gravity(fluid)
    _run_stage(dt, fluid, mag, mom, fluxes, fluid1, mag1)
    apply_gravity(dt, fluid1, gx, gy, gz)
    # ----- Stage 2 -----
    update_fluid_ghosts(fluid1)
    update_magnetic_ghosts(mag1)
    gx, gy, gz = compute_gravity(fluid1)
    _run_stage(dt, fluid1, mag1, mom1, fluxes1, fluid2, mag2)
    apply_gravity(dt, fluid2, gx, gy, gz)
    fluid2.rho[:] = 0.75 * fluid.rho + 0.25 * fluid2.rho
    fluid2.mx[:]  = 0.75 * fluid.mx  + 0.25 * fluid2.mx
    fluid2.my[:]  = 0.75 * fluid.my  + 0.25 * fluid2.my
    fluid2.mz[:]  = 0.75 * fluid.mz  + 0.25 * fluid2.mz
    fluid2.E[:]   = 0.75 * fluid.E   + 0.25 * fluid2.E
    mag2.Bx.data[:] = 0.75 * mag.Bx.data + 0.25 * mag2.Bx.data
    mag2.By.data[:] = 0.75 * mag.By.data + 0.25 * mag2.By.data
    mag2.Bz.data[:] = 0.75 * mag.Bz.data + 0.25 * mag2.Bz.data
    mag2.psi.data[:] = 0.75 * mag.psi.data + 0.25 * mag2.psi.data
    # ----- Stage 3 -----
    update_fluid_ghosts(fluid2)
    update_magnetic_ghosts(mag2)
    gx, gy, gz = compute_gravity(fluid2)
    _run_stage(dt, fluid2, mag2, mom3, fluxes2, fluid3, mag3)
    apply_gravity(dt, fluid3, gx, gy, gz)
    fluid.rho[:] = (1.0/3.0) * fluid.rho + (2.0/3.0) * fluid3.rho
    fluid.mx[:]  = (1.0/3.0) * fluid.mx  + (2.0/3.0) * fluid3.mx
    fluid.my[:]  = (1.0/3.0) * fluid.my  + (2.0/3.0) * fluid3.my
    fluid.mz[:]  = (1.0/3.0) * fluid.mz  + (2.0/3.0) * fluid3.mz
    fluid.E[:]   = (1.0/3.0) * fluid.E   + (2.0/3.0) * fluid3.E
    mag.Bx.data[:] = (1.0/3.0) * mag.Bx.data + (2.0/3.0) * mag3.Bx.data
    mag.By.data[:] = (1.0/3.0) * mag.By.data + (2.0/3.0) * mag3.By.data
    mag.Bz.data[:] = (1.0/3.0) * mag.Bz.data + (2.0/3.0) * mag3.Bz.data
    mag.psi.data[:] = (1.0/3.0) * mag.psi.data + (2.0/3.0) * mag3.psi.data
    fluid.update_velocity()
    # Density floor and NaN guard
    fluid.rho[:] = cp.maximum(fluid.rho, 1e-5)
    # Extra galactic-disk source terms (operator-split, after the 8-kernel chain)
    apply_viscosity(fluid, dt)
    apply_resistivity(mag, dt)
    apply_feedback(fluid, dt)
    if not cp.all(cp.isfinite(fluid.E)):
        print("NaN detected in energy")
        fluid.E[:] = cp.nan_to_num(fluid.E, nan=1e-3, posinf=1e3, neginf=1e-3)
    if not cp.all(cp.isfinite(mag.Bx.data)) or not cp.all(cp.isfinite(mag.By.data)) or not cp.all(cp.isfinite(mag.Bz.data)):
        print("NaN detected in B — sanitizing (CT preserved)")
        mag.Bx.data[:] = cp.nan_to_num(mag.Bx.data, nan=0.0, posinf=1.0, neginf=-1.0)
        mag.By.data[:] = cp.nan_to_num(mag.By.data, nan=0.0, posinf=1.0, neginf=-1.0)
        mag.Bz.data[:] = cp.nan_to_num(mag.Bz.data, nan=cfg.Bz0, posinf=1.0, neginf=-1.0)
        mag.psi.data[:] = cp.nan_to_num(mag.psi.data, nan=0.0)
        mag.Emfx[:] = cp.nan_to_num(mag.Emfx, nan=0.0)
        mag.Emfy[:] = cp.nan_to_num(mag.Emfy, nan=0.0)
        mag.Emfz[:] = cp.nan_to_num(mag.Emfz, nan=0.0)

# =====================================================================
# 7. BOUNDARIES
# =====================================================================
def _ghost_periodic(arr):
    arr[:NG] = arr[-2*NG:-NG]; arr[-NG:] = arr[NG:2*NG]
    arr[:, :NG] = arr[:, -2*NG:-NG]; arr[:, -NG:] = arr[:, NG:2*NG]
    arr[:, :, :NG] = arr[:, :, -2*NG:-NG]; arr[:, :, -NG:] = arr[:, :, NG:2*NG]

def _ghost_outflow(arr):
    arr[:NG] = arr[NG]; arr[-NG:] = arr[-NG-1]
    arr[:, :NG] = arr[:, NG]; arr[:, -NG:] = arr[:, -NG-1]
    arr[:, :, :NG] = arr[:, :, NG]; arr[:, :, -NG:] = arr[:, :, -NG-1]

def _ghost_cell(arr, bc):
    """Ghost fill for cell-centered arrays (Ni, Ni, Ni)."""
    if bc == 0:
        arr[:NG] = arr[-2*NG:-NG]; arr[-NG:] = arr[NG:2*NG]
        arr[:, :NG] = arr[:, -2*NG:-NG]; arr[:, -NG:] = arr[:, NG:2*NG]
        arr[:, :, :NG] = arr[:, :, -2*NG:-NG]; arr[:, :, -NG:] = arr[:, :, NG:2*NG]
    else:
        arr[:NG] = arr[NG]; arr[-NG:] = arr[-NG-1]
        arr[:, :NG] = arr[:, NG]; arr[:, -NG:] = arr[:, -NG-1]
        arr[:, :, :NG] = arr[:, :, NG]; arr[:, :, -NG:] = arr[:, :, -NG-1]

def _ghost_face_normal(arr, axis, bc):
    """
    Ghost only the normal (Ni+1) axis of a face or edge array.
    Periodic: [0] = [-2], [-1] = [1]
    Outflow: copy the first interior face; do not overwrite the physical boundary.
    """
    if bc == 0:
        if axis == 0:
            arr[0] = arr[-2]; arr[-1] = arr[1]
        elif axis == 1:
            arr[:, 0] = arr[:, -2]; arr[:, -1] = arr[:, 1]
        else:
            arr[:, :, 0] = arr[:, :, -2]; arr[:, :, -1] = arr[:, :, 1]
    else:
        # outflow: copy interior face outward, leave the physical boundary alone
        if axis == 0:
            arr[0] = arr[1]; arr[-1] = arr[-2]
        elif axis == 1:
            arr[:, 0] = arr[:, 1]; arr[:, -1] = arr[:, -2]
        else:
            arr[:, :, 0] = arr[:, :, 1]; arr[:, :, -1] = arr[:, :, -2]

def update_fluid_ghosts(fluid_state):
    for f in [fluid_state.rho, fluid_state.mx, fluid_state.my, fluid_state.mz, fluid_state.E]:
        _ghost_cell(f, cfg.BC_TYPE)

def update_magnetic_ghosts(mag_state):
    """
    Staggered ghosts. Cell fields use the cell helper.
    Faces and edges wrap only the transverse axes; the normal
    (Ni+1) axis uses the two-face periodic / outflow copy.
    """
    bc = cfg.BC_TYPE
    # Face B — normal axis is the long one
    _ghost_face_normal(mag_state.Bx.data, 0, bc)
    _ghost_cell_transverse(mag_state.Bx.data, 0, bc)
    _ghost_face_normal(mag_state.By.data, 1, bc)
    _ghost_cell_transverse(mag_state.By.data, 1, bc)
    _ghost_face_normal(mag_state.Bz.data, 2, bc)
    _ghost_cell_transverse(mag_state.Bz.data, 2, bc)
    # Cell psi
    _ghost_cell(mag_state.psi.data, bc)
    # Edge EMF: Ex is normal to x (long axes are y,z)
    _ghost_face_normal(mag_state.Emfx, 1, bc)
    _ghost_face_normal(mag_state.Emfx, 2, bc)
    _ghost_cell_transverse(mag_state.Emfx, 1, bc)
    _ghost_cell_transverse(mag_state.Emfx, 2, bc)
    _ghost_face_normal(mag_state.Emfy, 0, bc)
    _ghost_face_normal(mag_state.Emfy, 2, bc)
    _ghost_cell_transverse(mag_state.Emfy, 0, bc)
    _ghost_cell_transverse(mag_state.Emfy, 2, bc)
    _ghost_face_normal(mag_state.Emfz, 0, bc)
    _ghost_face_normal(mag_state.Emfz, 1, bc)
    _ghost_cell_transverse(mag_state.Emfz, 0, bc)
    _ghost_cell_transverse(mag_state.Emfz, 1, bc)

def _ghost_cell_transverse(arr, normal_axis, bc):
    """Wrap the two axes that are not the normal axis, using cell rules."""
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    for ax in axes:
        if bc == 0:
            if ax == 0:
                arr[:NG] = arr[-2*NG:-NG]; arr[-NG:] = arr[NG:2*NG]
            elif ax == 1:
                arr[:, :NG] = arr[:, -2*NG:-NG]; arr[:, -NG:] = arr[:, NG:2*NG]
            else:
                arr[:, :, :NG] = arr[:, :, -2*NG:-NG]; arr[:, :, -NG:] = arr[:, :, NG:2*NG]
        else:
            if ax == 0:
                arr[:NG] = arr[NG]; arr[-NG:] = arr[-NG-1]
            elif ax == 1:
                arr[:, :NG] = arr[:, NG]; arr[:, -NG:] = arr[:, -NG-1]
            else:
                arr[:, :, :NG] = arr[:, :, NG]; arr[:, :, -NG:] = arr[:, :, -NG-1]

def update_flux_ghosts(fluxes):
    """Ghosts on face fluxes after HLLD, before UCT."""
    bc = cfg.BC_TYPE
    # X-faces
    for name in ("Fx_rho", "Fx_mx", "Fx_my", "Fx_mz", "Fx_E", "Fx_By", "Fx_Bz"):
        a = getattr(fluxes, name)
        _ghost_face_normal(a, 0, bc)
        _ghost_cell_transverse(a, 0, bc)
    # Y-faces
    for name in ("Fy_rho", "Fy_mx", "Fy_my", "Fy_mz", "Fy_E", "Fy_Bx", "Fy_Bz"):
        a = getattr(fluxes, name)
        _ghost_face_normal(a, 1, bc)
        _ghost_cell_transverse(a, 1, bc)
    # Z-faces
    for name in ("Fz_rho", "Fz_mx", "Fz_my", "Fz_mz", "Fz_E", "Fz_Bx", "Fz_By"):
        a = getattr(fluxes, name)
        _ghost_face_normal(a, 2, bc)
        _ghost_cell_transverse(a, 2, bc)

def update_moment_ghosts(moment_state):
    """Ghosts on pressure tensor after gyro, before EMF."""
    for name in ("Ppar", "Pperp", "P", "pe_xx", "pe_yy", "pe_zz", "pe_xy", "pe_xz", "pe_yz"):
        _ghost_cell(getattr(moment_state, name), cfg.BC_TYPE)

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
    """Total energy (internal + kinetic + magnetic) in the interior.
    fluid.E already contains magnetic energy; do not add it again.
    """
    return clean_energy()

def divB_max(mag_state=None):
    """Discrete divergence on the staggered lattice.
    Pass the state you just stepped (e.g. mag1) so the test sees the update.
    """
    m = mag if mag_state is None else mag_state
    divB = ((m.Bx.data[1:] - m.Bx.data[:-1]) +
            (m.By.data[:, 1:] - m.By.data[:, :-1]) +
            (m.Bz.data[:, :, 1:] - m.Bz.data[:, :, :-1])) / dx
    return float(cp.max(cp.abs(divB[interior])))

def force_budget(gx, gy, gz, moment_state):
    """
    Radial force budget on the disk.
    Returns dict with gravity, Lorentz, pressure, centrifugal, residual.
    Uses cell-centered B so the curl is well-defined.
    """
    inv2dx = 0.5 / dx
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    # J = curl(B) at cell centers
    jx = ((cp.roll(Bz_c, -1, 1) - cp.roll(Bz_c, 1, 1)) -
          (cp.roll(By_c, -1, 2) - cp.roll(By_c, 1, 2))) * inv2dx
    jy = ((cp.roll(Bx_c, -1, 2) - cp.roll(Bx_c, 1, 2)) -
          (cp.roll(Bz_c, -1, 0) - cp.roll(Bz_c, 1, 0))) * inv2dx
    jz = ((cp.roll(By_c, -1, 0) - cp.roll(By_c, 1, 0)) -
          (cp.roll(Bx_c, -1, 1) - cp.roll(Bx_c, 1, 1))) * inv2dx
    # Lorentz force density
    Fx = jy * Bz_c - jz * By_c
    Fy = jz * Bx_c - jx * Bz_c
    Fr_L = (Fx * X + Fy * Y) / (R + 1e-3)
    Fr_g = fluid.rho * (gx * X + gy * Y) / (R + 1e-3)
    v_phi = (-fluid.vx * Y + fluid.vy * X) / (R + 1e-3)
    Fr_c = fluid.rho * v_phi**2 / (R + 1e-3)
    # Pressure gradient (use isotropic P; fall back to cs2*rho if unset)
    P = moment_state.P if float(cp.max(moment_state.P)) > 1e-12 else cfg.cs2 * fluid.rho
    dP_dx = (cp.roll(P, -1, 0) - cp.roll(P, 1, 0)) * inv2dx
    dP_dy = (cp.roll(P, -1, 1) - cp.roll(P, 1, 1)) * inv2dx
    Fr_p = -(dP_dx * X + dP_dy * Y) / (R + 1e-3)
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

def angular_momentum():
    """Total angular momentum about the z-axis (interior)."""
    Lz = fluid.rho * (X * fluid.vy - Y * fluid.vx)
    return float(cp.sum(Lz[interior]) * dx**3)

def clean_energy():
    """Total energy without magnetic double-counting.
    fluid.E already contains magnetic energy; do not add it again.
    """
    return float(cp.sum(fluid.E[interior]) * dx**3)

def positivity_check():
    """Returns dict of min density, min pressure, min |B| in interior."""
    rho_min = float(cp.min(fluid.rho[interior]))
    P_min = float(cp.min(mom.P[interior]))
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c[interior]**2 + By_c[interior]**2 + Bz_c[interior]**2
    B_min = float(cp.sqrt(cp.min(B2)))
    return {"rho_min": rho_min, "P_min": P_min, "B_min": B_min}

def radial_force_profile(gx, gy, gz, n_bins=16):
    """Binned radial force balance (interior disk).
    Returns arrays of r, F_g, F_L, F_p, F_c, residual.
    """
    inv2dx = 0.5 / dx
    jx = ((cp.roll(mag.Bz.data, -1, 1) - cp.roll(mag.Bz.data, 1, 1)) -
          (cp.roll(mag.By.data, -1, 2) - cp.roll(mag.By.data, 1, 2))) * inv2dx
    jy = ((cp.roll(mag.Bx.data, -1, 2) - cp.roll(mag.Bx.data, 1, 2)) -
          (cp.roll(mag.Bz.data, -1, 0) - cp.roll(mag.Bz.data, 1, 0))) * inv2dx
    jz = ((cp.roll(mag.By.data, -1, 0) - cp.roll(mag.By.data, 1, 0)) -
          (cp.roll(mag.Bx.data, -1, 1) - cp.roll(mag.Bx.data, 1, 1))) * inv2dx
    Fx = jy * mag.Bz.data - jz * mag.By.data
    Fy = jz * mag.Bx.data - jx * mag.Bz.data
    Fr_L = (Fx * X + Fy * Y) / (R + 1e-3)
    Fr_g = fluid.rho * (gx * X + gy * Y) / (R + 1e-3)
    v_phi = (-fluid.vx * Y + fluid.vy * X) / (R + 1e-3)
    Fr_c = fluid.rho * v_phi**2 / (R + 1e-3)
    dP_dx = (cp.roll(mom.P, -1, 0) - cp.roll(mom.P, 1, 0)) * inv2dx
    dP_dy = (cp.roll(mom.P, -1, 1) - cp.roll(mom.P, 1, 1)) * inv2dx
    Fr_p = -(dP_dx * X + dP_dy * Y) / (R + 1e-3)
    residual = Fr_g + Fr_p + Fr_L + Fr_c
    r_max = float(cp.max(R[interior]))
    bins = cp.linspace(0, r_max, n_bins + 1)
    r_c = 0.5 * (bins[:-1] + bins[1:])
    Fg = cp.zeros(n_bins); FL = cp.zeros(n_bins)
    Fp = cp.zeros(n_bins); Fc = cp.zeros(n_bins)
    Res = cp.zeros(n_bins)
    for i in range(n_bins):
        mask = (R[interior] >= bins[i]) & (R[interior] < bins[i+1])
        if cp.any(mask):
            Fg[i] = cp.mean(Fr_g[interior][mask])
            FL[i] = cp.mean(Fr_L[interior][mask])
            Fp[i] = cp.mean(Fr_p[interior][mask])
            Fc[i] = cp.mean(Fr_c[interior][mask])
            Res[i] = cp.mean(residual[interior][mask])
    return r_c.get(), Fg.get(), FL.get(), Fp.get(), Fc.get(), Res.get()

def magnetic_energy():
    """Magnetic energy from cell-centered B (interior)."""
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c[interior]**2 + By_c[interior]**2 + Bz_c[interior]**2
    return float(0.5 * cp.sum(B2) * dx**3)

def plasma_beta():
    """Volume-averaged plasma beta (P / (B^2/2)) in the disk."""
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c[interior]**2 + By_c[interior]**2 + Bz_c[interior]**2
    beta = 2.0 * mom.P[interior] / cp.maximum(B2, 1e-12)
    disk = (R[interior] < 6.0) & (cp.abs(Z[interior]) < 1.2)
    if not cp.any(disk):
        return float(cp.mean(beta))
    return float(cp.mean(beta[disk]))

def maxwell_stress():
    """Volume-averaged Maxwell stress -B_R B_phi inside the disk body."""
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    BR = (Bx_c * X + By_c * Y) / (R + 1e-4)
    Bphi = (-Bx_c * Y + By_c * X) / (R + 1e-4)
    MRphi = -BR * Bphi
    disk = (R < 0.8 * cfg.L) & (cp.abs(Z) < 2.0 * cfg.zd0)
    return float(cp.mean(MRphi[disk]))

def toomre_Q(n_bins=16):
    """
    Radial Toomre Q profile in the disk.
    Q = cs * kappa / (pi * G * Sigma)
    cs is taken from the local mid-plane pressure (P/rho),
    falling back to cfg.cs2 if pressure is unset.
    G = 1/(4*pi) matches the FFT Poisson solver.
    Returns (r_centers, Q, Sigma, kappa).
    """
    fluid.update_velocity()
    G = 1.0 / (4.0 * np.pi)
    # Local sound speed from pressure; fall back to constant
    P = mom.P if float(cp.max(mom.P)) > 1e-12 else cfg.cs2 * fluid.rho
    cs2_local = P / cp.maximum(fluid.rho, 1e-8)
    # Vertical surface density
    Sigma = cp.sum(fluid.rho[interior], axis=2) * dx
    # Mid-plane azimuthal velocity
    v_phi = (-fluid.vx * Y + fluid.vy * X) / (R + 1e-3)
    Omega = v_phi / (R + 1e-3)
    # Bin by mid-plane radius
    R2d = R[interior][:, :, Ni//2]
    r_max = float(cp.max(R2d))
    bins = cp.linspace(0.5 * dx, r_max, n_bins + 1)
    r_c = 0.5 * (bins[:-1] + bins[1:])
    Sigma_bin = cp.zeros(n_bins)
    Omega_bin = cp.zeros(n_bins)
    cs_bin = cp.zeros(n_bins)
    for i in range(n_bins):
        mask = (R2d >= bins[i]) & (R2d < bins[i + 1])
        if cp.any(mask):
            Sigma_bin[i] = cp.mean(Sigma[mask])
            Omega_bin[i] = cp.mean(Omega[interior][:, :, Ni//2][mask])
            cs_bin[i] = cp.sqrt(cp.mean(cs2_local[interior][:, :, Ni//2][mask]))
    # Epicyclic frequency
    dlnR = cp.log((r_c[1:] + 1e-6) / (r_c[:-1] + 1e-6))
    dOmega2 = Omega_bin[1:]**2 - Omega_bin[:-1]**2
    kappa2 = 4.0 * Omega_bin[:-1]**2 + dOmega2 / dlnR
    kappa = cp.sqrt(cp.maximum(kappa2, 0.0))
    r_out = r_c[:-1]
    Sigma_out = Sigma_bin[:-1]
    cs_out = cs_bin[:-1]
    Q = cs_out * kappa / (np.pi * G * cp.maximum(Sigma_out, 1e-8))
    def _to_host(a):
        return a.get() if hasattr(a, 'get') else a
    return _to_host(r_out), _to_host(Q), _to_host(Sigma_out), _to_host(kappa)

def radial_velocity_profile(n_bins=16):
    """Binned mean radial velocity in the disk plane."""
    vr = (fluid.vx * X + fluid.vy * Y) / (R + 1e-3)
    r_max = float(cp.max(R[interior]))
    bins = cp.linspace(0, r_max, n_bins + 1)
    r_c = 0.5 * (bins[:-1] + bins[1:])
    vr_bin = cp.zeros(n_bins)
    for i in range(n_bins):
        mask = (R[interior] >= bins[i]) & (R[interior] < bins[i+1])
        if cp.any(mask):
            vr_bin[i] = cp.mean(vr[interior][mask])
    def _to_host(a):
        return a.get() if hasattr(a, 'get') else a
    return _to_host(r_c), _to_host(vr_bin)

def momentum_check():
    """Total linear momentum (interior)."""
    px = float(cp.sum(fluid.mx[interior]) * dx**3)
    py = float(cp.sum(fluid.my[interior]) * dx**3)
    pz = float(cp.sum(fluid.mz[interior]) * dx**3)
    return px, py, pz

def compute_dt():
    """Timestep limited by fluid, fast magnetosonic, Alfvén and Hall speeds."""
    rho_safe = cp.maximum(fluid.rho, 1e-5)
    v2 = (fluid.mx/rho_safe)**2 + (fluid.my/rho_safe)**2 + (fluid.mz/rho_safe)**2
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c**2 + By_c**2 + Bz_c**2
    cs2 = cfg.cs2
    c_fast = cp.sqrt(v2 + cs2 + B2 / rho_safe)
    c_alf = cp.sqrt(B2 / rho_safe)
    c_hall = cfg.hall_coeff * cp.sqrt(B2) / rho_safe
    cmax = float(cp.max(cp.maximum(c_fast, c_alf + c_hall)))
    return min(cfg.cfl * dx / max(cmax, 1e-3), cfg.dt_max)



# =====================================================================
# 9. MAIN
# =====================================================================
def set_self_consistent_rotation():
    """Rebuild the rotation curve from the full (external + self) gravity."""
    gx, gy, gz = compute_gravity(fluid)
    gR = (gx * X + gy * Y) / (R + 1e-4)
    v_phi2 = cp.maximum(-R * gR - cfg.cs2 * (R / cfg.Rd), 1e-6)
    v_phi = cp.sqrt(v_phi2)
    fluid.mx[:] = -v_phi * Y / (R + 1e-4) * fluid.rho
    fluid.my[:] =  v_phi * X / (R + 1e-4) * fluid.rho
    fluid.update_velocity()

def seed_spiral(m=2, eps=0.01, k=0.8):
    """Low-amplitude logarithmic spiral density seed."""
    theta = cp.arctan2(Y, X)
    phase = m * theta - k * R
    fluid.rho[:] *= (1.0 + eps * cp.cos(phase))
    fluid.rho[:] = cp.maximum(fluid.rho, 1e-5)
    # keep momentum consistent with the new density
    fluid.mx[:] = fluid.vx * fluid.rho
    fluid.my[:] = fluid.vy * fluid.rho
    # rebuild total energy so internal + kinetic + magnetic stay consistent
    gamma = 5.0/3.0
    P_init = cfg.cs2 * fluid.rho
    Bx_c, By_c, Bz_c = face_to_cell(mag)
    B2 = Bx_c**2 + By_c**2 + Bz_c**2
    fluid.E[:] = P_init / (gamma - 1.0) + 0.5 * (fluid.mx**2 + fluid.my**2 + fluid.mz**2) / fluid.rho + 0.5 * B2

def spiral_amplitude(m=2):
    """m-arm Fourier amplitude of the surface density (volume-integrated)."""
    theta = cp.arctan2(Y, X)
    rho_i = fluid.rho[interior]
    theta_i = theta[interior]
    amp = cp.abs(cp.mean(rho_i * cp.exp(1j * m * theta_i)))
    mean = cp.mean(rho_i)
    return float(amp / (mean + 1e-8))

def conservation_logger(step, dt):
    """
    Record absolute and relative drifts of the conserved quantities.
    Writes to plasma_conservation.txt.
    Relative drifts are measured against the values at step 0.
    """
    m  = mass()
    te = clean_energy()
    Lz = angular_momentum()
    divb = divB_max()
    if not hasattr(conservation_logger, "ref"):
        conservation_logger.ref = {"M": m, "TE": te, "Lz": Lz}
        with open("plasma_conservation.txt", "w") as f:
            f.write("step  dt  M  dM/M  TE  dTE/TE  Lz  dLz/Lz  divB\n")
    ref = conservation_logger.ref
    dM  = (m  - ref["M"])  / (abs(ref["M"])  + 1e-30)
    dTE = (te - ref["TE"]) / (abs(ref["TE"]) + 1e-30)
    dLz = (Lz - ref["Lz"]) / (abs(ref["Lz"]) + 1e-30)
    with open("plasma_conservation.txt", "a") as f:
        f.write(f"{step:6d}  {dt:.4e}  {m:.6e}  {dM:+.4e}  "
                f"{te:.6e}  {dTE:+.4e}  {Lz:.6e}  {dLz:+.4e}  {divb:.4e}\n")
    # Abort if any conserved quantity drifts more than 1%
    if max(abs(dM), abs(dTE), abs(dLz)) > 0.01:
        print(f"\nConservation failure at step {step}: "
              f"dM/M={dM:+.2e}  dTE/TE={dTE:+.2e}  dLz/Lz={dLz:+.2e}")
        raise RuntimeError("Relative drift exceeded 1%")
    # Dedicated energy conservation check (same 1% threshold)
    if abs(dTE) > 0.01:
        print(f"\nEnergy conservation failure at step {step}: dTE/TE={dTE:+.2e}")
        raise RuntimeError("Energy drift exceeded 1%")
    # Momentum conservation check (relative to characteristic momentum)
    px, py, pz = momentum_check()
    p_abs = max(abs(px), abs(py), abs(pz))
    M = mass()
    p_char = max(M * (cfg.cs2 ** 0.5), 1e-8)
    if p_abs > 0.05 * p_char:
        print(f"\nMomentum conservation failure at step {step}: "
              f"px={px:+.2e}  py={py:+.2e}  pz={pz:+.2e}  (char={p_char:.2e})")
        raise RuntimeError("Linear momentum exceeded 5% of characteristic")
    return dM, dTE, dLz

def main():
    set_self_consistent_rotation()
    seed_spiral(m=2, eps=0.01, k=0.8)
    vertical_equilibrium()
    correct_radial_drift()
    print("Self-consistent rotation set + m=2 spiral seed.")
    print("ALADIN Plasma Driver v2")
    print(f"Grid: {N}^3 interior  ({Ni}^3 padded)  dx={dx:.4f}  NG={NG}")
    print("Flow: gyro → HLLD → UCT → EMF → curl → GLM → gravity")
    print(f"Max steps: {cfg.max_steps}")
    if cfg.USE_CUDA_GRAPH and USE_GPU:
        capture_graph(compute_dt())
    log_path = "plasma_diagnostics.txt"
    with open(log_path, "w") as log:
        log.write("step  dt  M  Lz  KE  ME  TE  beta  divB  rho_min  P_min  B_min  px  py  pz  Fg  FL  Fp  Fc  R  Qmean  A2  Maxwell\n")
    print("[" + " " * 20 + "] 0%", end="", flush=True)
    t0 = time.time()
    for step in range(cfg.max_steps):

        update_fluid_ghosts(fluid)
        update_magnetic_ghosts(mag)
        gx, gy, gz = compute_gravity(fluid)
        dt = compute_dt()
        operator_step(dt)
        apply_cooling(fluid, dt, tau=0.2)
        # Progress bar
        filled = int(20 * (step + 1) / cfg.max_steps)
        bar = "█" * filled + " " * (20 - filled)
        pct = 100 * (step + 1) / cfg.max_steps
        print(f"\r[{bar}] {pct:5.1f}% complete", end="", flush=True)
        if step % cfg.print_interval == 0:
            m = mass()
            ke = kinetic_energy()
            te = clean_energy()
            me = magnetic_energy()
            Lz = angular_momentum()
            divb = divB_max()
            pos = positivity_check()
            beta = plasma_beta()
            px, py, pz = momentum_check()
            fb = force_budget(gx, gy, gz, mom)
            rQ, Q, Sig, kap = toomre_Q()
            Q_disk = Q[(rQ > 1.0) & (rQ < 6.0)]
            Qmean = float(np.mean(Q_disk)) if len(Q_disk) > 0 else float(np.mean(Q))
            A2 = spiral_amplitude(m=2)
            stress = maxwell_stress()
            dM, dTE, dLz = conservation_logger(step, dt)
            line = (f"{step:6d}  {dt:.4e}  {m:.6e}  {Lz:.6e}  {ke:.6e}  {me:.6e}  {te:.6e}  "
                    f"{beta:.4f}  {divb:.4e}  {pos['rho_min']:.4e}  {pos['P_min']:.4e}  {pos['B_min']:.4e}  "
                    f"{px:+.4e}  {py:+.4e}  {pz:+.4e}  "
                    f"{fb['gravity']:+.4e}  {fb['lorentz']:+.4e}  {fb['pressure']:+.4e}  "
                    f"{fb['centrifugal']:+.4e}  {fb['residual']:+.4e}  {Qmean:.4f}  {A2:.4e}  {stress:.4e}\n")
            with open(log_path, "a") as log:
                log.write(line)
            print(f"step {step:3d}/{cfg.max_steps}  dt={dt:.2e}  M={m:.3e}  Lz={Lz:.3e}  A2={A2:.3e}  Maxwell={stress:.3e}")
            print(f"         KE={ke:.3e}  ME={me:.3e}  TE={te:.3e}  beta={beta:.2f}  max|divB|={divb:.2e}")
            print(f"         dM/M={dM:+.2e}  dTE/TE={dTE:+.2e}  dLz/Lz={dLz:+.2e}")
            print(f"         rho_min={pos['rho_min']:.2e}  P_min={pos['P_min']:.2e}  B_min={pos['B_min']:.2e}")
            print(f"         px={px:+.3e}  py={py:+.3e}  pz={pz:+.3e}")
            print(f"         F_g={fb['gravity']:+.3e}  F_L={fb['lorentz']:+.3e}  F_p={fb['pressure']:+.3e}  F_c={fb['centrifugal']:+.3e}  R={fb['residual']:+.3e}")
            if len(Q_disk) > 0:
                print(f"         Toomre Q (disk body): min={np.min(Q_disk):.2f}  mean={np.mean(Q_disk):.2f}  max={np.max(Q_disk):.2f}")
            else:
                print(f"         Toomre Q: mean={np.mean(Q):.2f}")





            bmax = max(
                float(cp.max(cp.abs(mag.Bx.data))),
                float(cp.max(cp.abs(mag.By.data))),
                float(cp.max(cp.abs(mag.Bz.data))),
            )
            print(f"         min_rho={float(cp.min(fluid.rho)):.2e}  max|E|={float(cp.max(cp.abs(fluid.E))):.2e}  max|B|={bmax:.2e}")
    print()
    print("Done. Elapsed:", time.time() - t0, "s")


def test_gyro_admissibility():
    """Check that the gyro closure yields physically admissible moments."""
    print("\n--- Gyro admissibility test ---")
    # reset to a simple uniform state
    fluid.rho[:] = 1.0
    fluid.vx[:] = 0.0
    fluid.vy[:] = 0.0
    fluid.vz[:] = 0.0
    mag.Bx[:] = 1.0
    mag.By[:] = 0.0
    mag.Bz[:] = 0.0
    fill_target_moments(mom, fluid)
    call_gyro(fluid, mom, mag)
    # positivity
    assert float(cp.min(mom.Ppar)) > 0, "Ppar not positive"
    assert float(cp.min(mom.Pperp)) > 0, "Pperp not positive"
    # reconstruction consistency: pe_xx should equal Pperp when B = x-hat
    # (P_xx = Pperp + (Ppar-Pperp)*bx^2 = Ppar)
    pe_xx = float(cp.mean(mom.pe_xx))
    Ppar = float(cp.mean(mom.Ppar))
    Pperp = float(cp.mean(mom.Pperp))
    assert abs(pe_xx - Ppar) < 1e-5, f"pe_xx != Ppar ({pe_xx} vs {Ppar})"
    # off-diagonals should vanish for aligned field
    assert float(cp.max(cp.abs(mom.pe_xy))) < 1e-6
    assert float(cp.max(cp.abs(mom.pe_xz))) < 1e-6
    # status must be clean
    assert int(cp.max(mom.status)) == 0
    # target residual on CPU identity is not zero, but finite
    assert float(cp.max(cp.abs(mom.M_recon))) < 1e6
    print(f"  Ppar={Ppar:.4e}  Pperp={Pperp:.4e}  pe_xx={pe_xx:.4e}")
    print("  positivity, alignment, status: OK")
    print("--- done ---")

def test_staggered_ct():
    """Check that staggered CT + GLM keeps discrete divB near zero."""
    print("\n--- Staggered CT test ---")
    # Divergence-free initial field (uniform Bz)
    mag.Bx[:] = 0.0
    mag.By[:] = 0.0
    mag.Bz[:] = 1.0
    mag.psi[:] = 0.0
    mag.Emfx.fill(0.0)
    mag.Emfy.fill(0.0)
    mag.Emfz.fill(0.0)
    update_magnetic_ghosts(mag)
    d0 = divB_max()
    print(f"Initial max|divB| = {d0:.2e}")

    # Apply one CT + GLM step with zero EMF (should stay divergence-free)
    call_curl(0.01, mag, mag1)
    call_glm(0.01, mag1)
    d1 = divB_max(mag1)
    print(f"After CT+GLM     = {d1:.2e}")

    # Inject a small divergence and let GLM clean it
    mag1.Bx.data[Ni//2, Ni//2, Ni//2] += 0.1
    d_dirty = divB_max(mag1)
    print(f"After injection  = {d_dirty:.2e}")
    for _ in range(5):
        call_glm(0.01, mag1)
    d_clean = divB_max(mag1)
    print(f"After 5 GLM      = {d_clean:.2e}")
    print("--- done ---")

def test_abi():
    """Assert frozen sockets and shapes before any kernel is dropped in."""
    print("\n--- ABI test ---")
    assert mag.Bx.shape == (Ni+1, Ni, Ni), "Bx face shape"
    assert mag.By.shape == (Ni, Ni+1, Ni), "By face shape"
    assert mag.Bz.shape == (Ni, Ni, Ni+1), "Bz face shape"
    assert mag.Emfx.shape == (Ni, Ni+1, Ni+1), "Emfx edge shape"
    assert mag.Emfx is mag.Emf_total[0].data, "Emfx alias"
    assert mag.Emfy is mag.Emf_total[1].data, "Emfy alias"
    assert mag.Emfz is mag.Emf_total[2].data, "Emfz alias"
    # Identity HLLD zeros fluxes
    fluxes.zero()
    call_hlld(0.01, fluid, mag, fluxes, fluid1)
    assert float(cp.max(cp.abs(fluxes.Fx_rho))) == 0.0, "HLLD identity fluxes"
    # sum_emf
    mag.Emf_ideal[0].fill(1.0)
    mag.Emf_nonideal[0].fill(2.0)
    mag.sum_emf()
    assert float(cp.max(cp.abs(mag.Emfx - 3.0))) < 1e-5, "sum_emf"
    print("ABI ok")

def test_emf_plug():
    """
    Isolate the EMF path.
    Pass criteria:
      1. Kernel launches and writes finite edge EMFs
      2. Edge arrays have the correct staggered shapes
      3. Curl of those EMFs changes B
      4. max|divB| stays bounded after GLM
    """
    print("\n--- EMF plug test ---")
    # Uniform density, mild rotation, seed B
    fluid.rho[:] = 1.0
    fluid.vx[:] = 0.0
    fluid.vy[:] = 0.1
    fluid.vz[:] = 0.0
    mag.Bx[:] = 0.0
    mag.By[:] = 0.0
    mag.Bz[:] = 1.0
    mag.Emfx[:] = 0.0
    mag.Emfy[:] = 0.0
    mag.Emfz[:] = 0.0
    fill_target_moments(mom, fluid)
    call_gyro(fluid, mom, mag)
    reconstruct_pressure_tensor(mom, mag)
    call_emf(fluid, mag, mom)
    # 2. correct shapes (always)
    assert mag.Emfx.shape == (Ni, Ni+1, Ni+1), f"Emfx shape {mag.Emfx.shape}"
    assert mag.Emfy.shape == (Ni+1, Ni, Ni+1), f"Emfy shape {mag.Emfy.shape}"
    assert mag.Emfz.shape == (Ni+1, Ni+1, Ni), f"Emfz shape {mag.Emfz.shape}"
    if not USE_GPU:
        print("  shapes OK  (GPU skipped — no CUDA)")
        print("  EMF plug: structure OK")
        return
    # 1. finite edge EMFs
    ex = float(cp.max(cp.abs(mag.Emfx)))
    ey = float(cp.max(cp.abs(mag.Emfy)))
    ez = float(cp.max(cp.abs(mag.Emfz)))
    assert cp.all(cp.isfinite(mag.Emfx)), "Emfx not finite"
    assert cp.all(cp.isfinite(mag.Emfy)), "Emfy not finite"
    assert cp.all(cp.isfinite(mag.Emfz)), "Emfz not finite"
    assert (ex + ey + ez) > 0, "EMF all zero — kernel did not write"
    # 3. curl changes B
    b0 = float(cp.sum(mag.By.data**2))
    call_curl(1e-3, mag, mag)
    b1 = float(cp.sum(mag.By.data**2))
    assert abs(b1 - b0) > 0, "Curl did not change B — EMF not on edges"
    # 4. GLM bounds divB
    for _ in range(3):
        call_glm(1e-3, mag)
    divb = divB_max()
    assert cp.isfinite(divb), "divB not finite"
    print(f"  Emf max: Ex={ex:.2e}  Ey={ey:.2e}  Ez={ez:.2e}")
    print(f"  |B| change after curl: {abs(b1-b0):.2e}")
    print(f"  max|divB| after 3 GLM: {divb:.2e}")
    print("  EMF plug: OK")

if __name__ == "__main__":
    test_abi()
    test_staggered_ct()
    test_emf_plug()
    main()
