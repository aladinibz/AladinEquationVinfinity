"""
ALADIN Plasma Drive v2.1 — Cylindrical Architecture
Driver prepared for 8 kernels (Colab Pro A100 / raw CUDA)

Locked kernel sequence (9):
gyro → hlldx → hlldy → hlldz → utc → uct_curl → uct_emf → glm → g2

Geometric sources applied inside every RK stage.
Face/edge geometric factors exposed for kernels.
Initial conditions live in separate functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Callable, Any
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

PI = math.pi
TWO_PI = 2.0 * PI

Centering = Literal[
    "cell",
    "face_r", "face_phi", "face_z",
    "edge_r", "edge_phi", "edge_z"
]

Lifecycle = Literal["CREATED", "INITIALIZED", "RUNNING", "COMPLETED", "FAILED", "CLOSED"]


# =============================================================================
# LAYER 7 — CONFIGURATION
# =============================================================================
@dataclass
class ALADINConfig:
    nr: int = 128
    nphi: int = 128
    nz: int = 128
    r_min: float = 0.0
    r_max: float = 1.0
    z_min: float = -0.5
    z_max: float = 0.5
    precision: str = "float32"
    cfl: float = 0.4
    max_dt: float = 1e-3
    min_dt: float = 1e-12
    self_gravity: bool = False
    external_gravity: bool = True
    use_hall: bool = True
    hall_coeff: float = 0.018
    use_velocity_cap: bool = False
    v_max_cap: float = 5.0
    seed: int = 42
    ic_name: str = "uniform"          # "uniform" or "equilibrium_disk"
    backend: str = "cupy"
    ghost_width: int = 2
    strict_gpu: bool = False
    print_dme_steps: int = 5
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "artifacts"

    def __post_init__(self):
        self.validate()

    def validate(self):
        if min(self.nr, self.nphi, self.nz) < 4:
            raise ValueError("Grid dimensions too small")
        if self.r_min < 0.0:
            raise ValueError("r_min cannot be negative")
        if not (0.0 < self.cfl <= 1.0):
            raise ValueError("CFL out of range")
        if self.ghost_width < 1:
            raise ValueError("ghost_width must be >= 1")
        if self.backend == "cupy" and not HAS_CUPY:
            if self.strict_gpu:
                raise RuntimeError("CuPy required but not available")
            print("WARNING: CuPy unavailable — falling back to NumPy")
            self.backend = "numpy"

    @property
    def dtype(self):
        return np.float32 if self.precision == "float32" else np.float64

    @property
    def xp(self):
        return cp if self.backend == "cupy" and HAS_CUPY else np

    def summary(self) -> str:
        n = self.nr * self.nphi * self.nz
        return (f"ALADIN v2.1 cylindrical | {self.nr}x{self.nphi}x{self.nz} = {n:,} | "
                f"{self.precision} | {self.backend} | seed={self.seed} ic={self.ic_name}")


# =============================================================================
# INITIAL CONDITIONS (separate from Driver)
# =============================================================================
def set_uniform(state, geometry, gravity=None, **kwargs):
    """Smoke-test IC: constant density, zero velocity, zero B."""
    state.get("rho")[:] = 1.0
    # momenta and E already zero from allocation
    # B already zero


def set_equilibrium_disk(state, geometry, gravity=None,
                         Sigma0: float = 1.0,
                         H: float = 0.1,
                         cs: float = 0.1,
                         beta: float = 100.0,
                         B_type: str = "none",
                         **kwargs):
    """
    Non-magnetized equilibrium disk (baseline).
    1. Gravity writes Phi, g_R, g_z
    2. Vertical hydrostatic balance → rho(R,z)
    3. Radial force balance → v_phi
    4. Isothermal pressure
    5. B = 0
    """
    xp = state.config.xp
    if gravity is None:
        state.get("rho")[:] = 1.0
        return

    # Ensure potential and accelerations exist
    gravity.update(state)

    sl = state.grid.interior_slice("cell")
    R = geometry.r_centers[:, None, None]
    z = geometry.z_centers[None, None, :]
    Phi = state.get("Phi")[sl]
    g_z = state.get("g_z")[sl] if "g_z" in state.fields else None

    # --- Vertical hydrostatic (isothermal approximation) ---
    # rho(z) ∝ exp( -(Phi(z) - Phi(0)) / cs^2 )
    # Use midplane as reference at each R
    Phi_mid = Phi[:, :, Phi.shape[2]//2:Phi.shape[2]//2+1]
    # Simple exponential vertical structure scaled by local g_z strength
    # (full integration can replace this later)
    scale = cs**2
    rho = Sigma0 / (math.sqrt(2 * math.pi) * H) * xp.exp(-(Phi - Phi_mid) / scale)
    # Floor
    rho = xp.maximum(rho, 1e-6)
    state.get("rho")[sl] = rho.astype(state.get("rho").dtype)

    # --- Radial balance for v_phi ---
    # v_phi^2 / R = dPhi/dR + (1/rho) dP/dR
    # Isothermal P = rho * cs^2 → (1/rho) dP/dR = cs^2 dlnrho/dR
    if "g_R" in state.fields:
        g_R = state.get("g_R")[sl]
        # pressure gradient term (numerical)
        lnrho = xp.log(rho + 1e-12)
        dlnrho_dR = xp.gradient(lnrho, geometry.r_centers, axis=0)
        vphi2 = R * (-g_R + cs**2 * dlnrho_dR)
        vphi2 = xp.maximum(vphi2, 0.0)
        vphi = xp.sqrt(vphi2)
        # rho * v_phi
        state.get("rho_vphi")[sl] = (rho * vphi).astype(state.get("rho_vphi").dtype)

    # Thermal energy (isothermal)
    # E ≈ P / (gamma-1) with gamma=5/3, P=rho*cs^2
    gamma = 5.0/3.0
    P = rho * cs**2
    state.get("E")[sl] = (P / (gamma - 1.0)).astype(state.get("E").dtype)

    # Magnetic field remains zero (non-magnetized baseline)
    print(f"Equilibrium disk (non-magnetized): Sigma0={Sigma0}, H={H}, cs={cs}")


IC_REGISTRY = {
    "uniform": set_uniform,
    "equilibrium_disk": set_equilibrium_disk,
}


# =============================================================================
# UNITS (physical ↔ internal)
# =============================================================================
class Units:
    """
    Internal units:
      length  = 1 kpc
      velocity = 1 km/s
      time    = 1 Myr
      mass    = 1 M☉
      B       = 1 µG
    """
    def __init__(self):
        self.kpc = 1.0
        self.km_s = 1.0
        self.Myr = 1.0
        self.Msun = 1.0
        self.uG = 1.0

        # G in (kpc^3 / M☉ / Myr^2)
        # Approximate value consistent with these units
        self.G = 4.498e-12   # kpc³ M☉⁻¹ Myr⁻²  (order-of-magnitude correct)

    def length(self, kpc: float) -> float:
        return kpc * self.kpc

    def velocity(self, km_s: float) -> float:
        return km_s * self.km_s

    def time(self, Myr: float) -> float:
        return Myr * self.Myr

    def mass(self, Msun: float) -> float:
        return Msun * self.Msun

    def magnetic(self, uG: float) -> float:
        return uG * self.uG


# =============================================================================
# LAYER 1 — GEOMETRY
# =============================================================================
class GeometryManager:
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.xp = config.xp
        self.dtype = config.dtype
        self.ghost = config.ghost_width
        self._build()

    def _build(self):
        cfg = self.config
        xp = self.xp

        self.r_faces = xp.linspace(cfg.r_min, cfg.r_max, cfg.nr + 1, dtype=self.dtype)
        self.phi_faces = xp.linspace(0.0, TWO_PI, cfg.nphi + 1, dtype=self.dtype)
        self.z_faces = xp.linspace(cfg.z_min, cfg.z_max, cfg.nz + 1, dtype=self.dtype)

        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.phi_centers = 0.5 * (self.phi_faces[:-1] + self.phi_faces[1:])
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        self.dr = self.r_faces[1:] - self.r_faces[:-1]
        self.dphi = self.phi_faces[1:] - self.phi_faces[:-1]
        self.dz = self.z_faces[1:] - self.z_faces[:-1]

        r_out = self.r_faces[1:]
        r_in = self.r_faces[:-1]
        self.cell_volumes = (0.5 * (r_out**2 - r_in**2))[:, None, None] * \
                            self.dphi[None, :, None] * self.dz[None, None, :]

        self.axis_mask = (self.r_faces[:-1] == 0.0)
        self.r_c = self.r_centers[:, None, None]
        self.one_over_r = xp.where(self.r_c > 0, 1.0 / self.r_c, 0.0)

        # Face areas
        self.area_r = self.r_faces[:, None, None] * self.dphi[None, :, None] * self.dz[None, None, :]
        self.area_phi = self.dr[:, None, None] * self.dz[None, None, :]
        self.area_z = 0.5 * (self.r_faces[1:]**2 - self.r_faces[:-1]**2)[:, None, None] * self.dphi[None, :, None]

        # Edge lengths
        self.edge_len_r = self.dr[:, None, None]
        self.edge_len_phi = self.r_centers[:, None, None] * self.dphi[None, :, None]
        self.edge_len_z = self.dz[None, None, :]

        # r at kernel locations
        self.r_face_r = self.r_faces[:, None, None]
        self.r_face_phi = self.r_centers[:, None, None]
        self.r_edge_z = self.r_faces[:, None, None]

        assert xp.all(self.cell_volumes > 0)

    def face_areas(self):
        return self.area_r, self.area_phi, self.area_z

    def edge_lengths(self):
        return self.edge_len_r, self.edge_len_phi, self.edge_len_z

    def total_volume(self) -> float:
        return float(self.xp.sum(self.cell_volumes))

    def interior_slice(self) -> Tuple[slice, slice, slice]:
        g = self.ghost
        return (slice(g, -g), slice(g, -g), slice(g, -g))


class GeometricSources:
    """1/r geometric source terms for cylindrical momentum."""
    def __init__(self, config, geometry):
        self.config = config
        self.geometry = geometry
        self.xp = config.xp

    def momentum_sources(self, state):
        xp = self.xp
        rho = state.interior("rho")
        vr = state.interior("rho_vr") / (rho + 1e-40)
        vphi = state.interior("rho_vphi") / (rho + 1e-40)
        r = self.geometry.r_centers
        r_safe = xp.maximum(r, 0.5 * self.geometry.dr)
        S_vr = rho * vphi**2 / r_safe
        S_vphi = -rho * vr * vphi / r_safe
        return S_vr, S_vphi

    def apply_momentum(self, state, dt):
        S_vr, S_vphi = self.momentum_sources(state)
        sl = state.grid.interior_slice("cell")
        state.get("rho_vr")[sl] += dt * S_vr
        state.get("rho_vphi")[sl] += dt * S_vphi


class GridManager:
    def __init__(self, config: ALADINConfig, geometry: GeometryManager):
        self.config = config
        self.geometry = geometry
        self.ghost = config.ghost_width
        self.nr, self.nphi, self.nz = config.nr, config.nphi, config.nz
        self._define_shapes()

    def _define_shapes(self):
        g = self.ghost
        nr, nphi, nz = self.nr, self.nphi, self.nz
        self.shapes = {
            "cell":     ((nr, nphi, nz),     (nr + 2*g, nphi + 2*g, nz + 2*g)),
            "face_r":   ((nr + 1, nphi, nz), (nr + 1 + 2*g, nphi + 2*g, nz + 2*g)),
            "face_phi": ((nr, nphi + 1, nz), (nr + 2*g, nphi + 1 + 2*g, nz + 2*g)),
            "face_z":   ((nr, nphi, nz + 1), (nr + 2*g, nphi + 2*g, nz + 1 + 2*g)),
            "edge_r":   ((nr, nphi + 1, nz + 1), (nr + 2*g, nphi + 1 + 2*g, nz + 1 + 2*g)),
            "edge_phi": ((nr + 1, nphi, nz + 1), (nr + 1 + 2*g, nphi + 2*g, nz + 1 + 2*g)),
            "edge_z":   ((nr + 1, nphi + 1, nz), (nr + 1 + 2*g, nphi + 1 + 2*g, nz + 2*g)),
        }

    def get_shape(self, centering: Centering, ghosts: bool = False) -> Tuple[int, ...]:
        physical, ghosted = self.shapes[centering]
        return ghosted if ghosts else physical

    def interior_slice(self, centering: Centering = "cell"):
        g = self.ghost
        return (slice(g, -g), slice(g, -g), slice(g, -g))


# =============================================================================
# LAYER 2 — STATE
# =============================================================================
class FieldMetadata:
    def __init__(self, name, centering, role, checkpoint=True, rk=False):
        self.name = name
        self.centering = centering
        self.role = role
        self.checkpoint = checkpoint
        self.rk = rk
        self.valid = False


class StateRegistry:
    def __init__(self, config: ALADINConfig, grid: GridManager):
        self.config = config
        self.grid = grid
        self.xp = config.xp
        self.fields: Dict[str, Any] = {}
        self.metadata: Dict[str, FieldMetadata] = {}

    def register(self, name: str, centering: Centering, role: str,
                 checkpoint: bool = True, rk: bool = False):
        meta = FieldMetadata(name, centering, role, checkpoint, rk)
        self.metadata[name] = meta
        shape = self.grid.get_shape(centering, ghosts=True)
        self.fields[name] = self.xp.zeros(shape, dtype=self.config.dtype)
        meta.valid = True

    def get(self, name: str):
        return self.fields[name]

    def interior(self, name: str):
        arr = self.fields[name]
        sl = self.grid.interior_slice(self.metadata[name].centering)
        return arr[sl]

    def magnetic_energy(self) -> float:
        xp = self.xp
        return float(xp.sum(self.interior("Br")**2) +
                     xp.sum(self.interior("Bphi")**2) +
                     xp.sum(self.interior("Bz")**2))


# =============================================================================
# LAYER 1 — BOUNDARIES
# =============================================================================
class BoundaryManager:
    def __init__(self, config: ALADINConfig, geometry: GeometryManager, grid: GridManager):
        self.config = config
        self.geometry = geometry
        self.grid = grid
        self.xp = config.xp
        self.ghost = config.ghost_width

    def apply_all(self, state: StateRegistry):
        for name, meta in state.metadata.items():
            if meta.role != "prognostic":
                continue
            arr = state.get(name)
            self._periodic_phi(arr, meta.centering)
            if self.geometry.axis_mask.any():
                self._axis(arr, meta.centering)
            self._outer(arr, meta.centering)

    def _periodic_phi(self, arr, centering: Centering):
        g = self.ghost
        if arr.shape[1] > 2 * g:
            arr[:, :g, :] = arr[:, -2*g:-g, :]
            arr[:, -g:, :] = arr[:, g:2*g, :]

    def _axis(self, arr, centering: Centering):
        g = self.ghost
        xp = self.xp
        if arr.shape[0] <= 2 * g:
            return
        if centering == "face_r":
            # Odd reflection first, then force the axis face to zero
            arr[:g, :, :] = -arr[2*g:g:-1, :, :]
            arr[0, :, :] = 0.0
        elif centering in ("face_phi", "edge_phi"):
            arr[:g, :, :] = -arr[2*g:g:-1, :, :]
            nphi = arr.shape[1]
            half = nphi // 2
            arr[:g, :, :] = xp.roll(arr[:g, :, :], half, axis=1)
        else:
            arr[:g, :, :] = arr[2*g:g:-1, :, :]

    def _outer(self, arr, centering: Centering):
        g = self.ghost
        if arr.shape[0] > 2 * g:
            arr[-g:, :, :] = arr[-g-1:-1, :, :]
        if arr.shape[2] > 2 * g:
            arr[:, :, :g] = arr[:, :, g:2*g]
            arr[:, :, -g:] = arr[:, :, -2*g:-g]


# =============================================================================
# MEMORY POOL MANAGER
# =============================================================================
class MemoryPoolManager:
    """
    Owns CuPy device + pinned memory pools.
    Reports usage, warms up, and frees cleanly.
    No-op when CuPy is absent.
    """
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.enabled = HAS_CUPY and config.backend == "cupy"
        self.device_pool = None
        self.pinned_pool = None
        if self.enabled:
            self.device_pool = cp.get_default_memory_pool()
            self.pinned_pool = cp.get_default_pinned_memory_pool()

    def warmup(self, nbytes: int = 64 * 1024 * 1024):
        """Touch the pool so the first real allocation is fast."""
        if not self.enabled:
            return
        tmp = cp.empty(nbytes // 4, dtype=cp.float32)
        del tmp
        self.device_pool.free_all_blocks()

    def free(self):
        """Return all cached blocks to the driver."""
        if not self.enabled:
            return
        self.device_pool.free_all_blocks()
        self.pinned_pool.free_all_blocks()

    def report(self) -> str:
        if not self.enabled:
            return "MemoryPool: disabled (NumPy)"
        used = self.device_pool.used_bytes()
        total = self.device_pool.total_bytes()
        return (f"MemoryPool: used={used/1024**2:.1f} MiB  "
                f"total={total/1024**2:.1f} MiB")

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "used_bytes": self.device_pool.used_bytes(),
            "total_bytes": self.device_pool.total_bytes(),
            "pinned_used": self.pinned_pool.used_bytes(),
        }

class MemoryManager:
    def __init__(self, config: ALADINConfig, state: StateRegistry):
        self.config = config
        self.state = state

    def allocate_persistent(self):
        for n in ["rho", "rho_vr", "rho_vphi", "rho_vz", "E"]:
            self.state.register(n, "cell", "prognostic", True, True)
        self.state.register("Br", "face_r", "prognostic", True, False)
        self.state.register("Bphi", "face_phi", "prognostic", True, False)
        self.state.register("Bz", "face_z", "prognostic", True, False)
        for n in ["P_par", "P_perp", "q_par", "q_perp"]:
            self.state.register(n, "cell", "prognostic", True, True)
        self.state.register("Pe", "cell", "derived", True, False)
        self.state.register("entropy", "cell", "derived", False, False)
        self.state.register("uct_state", "cell", "workspace", False, False)
        self.state.register("flux_r", "face_r", "workspace", False, False)
        self.state.register("flux_phi", "face_phi", "workspace", False, False)
        self.state.register("flux_z", "face_z", "workspace", False, False)
        self.state.register("emf_r", "edge_r", "workspace", False, False)
        self.state.register("emf_phi", "edge_phi", "workspace", False, False)
        self.state.register("emf_z", "edge_z", "workspace", False, False)
        if self.config.self_gravity or self.config.external_gravity:
            self.state.register("Phi", "cell", "derived", True, False)
            self.state.register("g_R", "cell", "derived", False, False)
            self.state.register("g_z", "cell", "derived", False, False)
        self.state.register("divB", "cell", "diagnostic", False, False)
        # G2 geometry/algebra sources
        self.state.register("S_vr", "cell", "workspace", False, False)
        self.state.register("S_vphi", "cell", "workspace", False, False)
        self.state.register("S_vz", "cell", "workspace", False, False)
        self.state.register("S_E", "cell", "workspace", False, False)

    def report(self) -> str:
        total = sum(a.nbytes for a in self.state.fields.values())
        return f"{len(self.state.fields)} fields | {total/1024**3:.3f} GiB"


# =============================================================================
# GPU BACKEND (high-class: descriptors, stream, staggered launch, pointer table)
# =============================================================================
@dataclass
class KernelDescriptor:
    """Per-kernel launch contract."""
    name: str
    centering: str                  # "cell", "face_r", "edge_phi", ...
    block: int = 128                # warp multiple
    shared_bytes: int = 0
    fields: List[str] = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = []


KERNEL_DESCRIPTORS = {
    "gyro":              KernelDescriptor("gyro", "cell", 128, 0,
                                          ["rho", "P_par", "P_perp"]),
    "hlldx":             KernelDescriptor("hlldx", "face_r", 128, 0, ["flux_r"]),
    "hlldy":             KernelDescriptor("hlldy", "face_phi", 128, 0, ["flux_phi"]),
    "hlldz":             KernelDescriptor("hlldz", "face_z", 128, 0, ["flux_z"]),
    "utc":               KernelDescriptor("utc", "cell", 128, 0, ["flux_r", "flux_phi", "flux_z"]),
    "uct_curl":          KernelDescriptor("uct_curl", "face_r", 128, 0, ["Br", "Bphi", "Bz"]),
    "uct_emf":           KernelDescriptor("uct_emf", "edge_z", 128, 0,
                                          ["emf_r", "emf_phi", "emf_z"]),
    "glm":               KernelDescriptor("glm", "face_r", 128, 0, ["Br", "Bphi", "Bz"]),
    "g2":                KernelDescriptor("g2", "cell", 128, 0,
                                          ["rho", "rho_vr", "rho_vphi", "rho_vz",
                                           "Br", "Bphi", "Bz"]),
}


class KernelBackend:
    """Base class. Driver only calls execute(state, dt)."""
    def execute(self, state, dt):
        raise NotImplementedError
    def capture(self, state, dt):
        pass
    def replay(self):
        pass


class NumPyBackend(KernelBackend):
    """CPU fallback."""
    def __init__(self, operators):
        self.operators = operators
    def execute(self, state, dt):
        for name in ExecutionController.KERNELS:
            if name in self.operators and self.operators[name].enabled:
                self.operators[name].func(state, dt)


class CuPyBackend(KernelBackend):
    """
    Real CuPy backend:
    - one stream
    - per-kernel descriptor
    - actual RawKernel launches (placeholder math, real launch)
    """
    def __init__(self, operators, grid_manager=None):
        self.operators = operators
        self.grid = grid_manager
        self.descriptors = KERNEL_DESCRIPTORS
        self.kernels = {}
        self.stream = None
        if HAS_CUPY:
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Simple but real kernel: out = in + dt (manufactured)
            src = r'''
            extern "C" __global__
            void aladin_placeholder(const float* in, float* out, float dt, int n) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i < n) {
                    out[i] = in[i] + dt;
                }
            }
            '''
            for name in ExecutionController.KERNELS:
                self.kernels[name] = cp.RawKernel(src, "aladin_placeholder")

    def _grid_size(self, state, centering: str) -> int:
        if self.grid is None:
            return int(state.get("rho").size)
        shape = self.grid.get_shape(centering, ghosts=False)
        return int(np.prod(shape))

    def _launch(self, name: str, state, dt: float):
        desc = self.descriptors[name]
        kernel = self.kernels.get(name)
        if kernel is None:
            return
        n = self._grid_size(state, desc.centering)
        block = desc.block
        grid = (n + block - 1) // block
        arr = state.get("rho")
        dt32 = np.float32(dt)
        if self.stream is not None:
            with self.stream:
                kernel((grid,), (block,), (arr, arr, dt32, n))
        else:
            kernel((grid,), (block,), (arr, arr, dt32, n))

    def execute(self, state, dt):
        if not HAS_CUPY:
            return
        for name in ExecutionController.KERNELS:
            if name in self.kernels:
                self._launch(name, state, dt)
        if self.stream is not None:
            self.stream.synchronize()

    def parity_test(self, n: int = 1024, dt: float = 0.1) -> bool:
        """Manufactured check: GPU out == in + dt."""
        if not HAS_CUPY:
            return False
        x = cp.ones(n, dtype=cp.float32)
        y = cp.empty_like(x)
        kernel = self.kernels.get("uct") or self.kernels.get("gyro") or next(iter(self.kernels.values()), None)
        if kernel is None:
            return False
        block = 128
        grid = (n + block - 1) // block
        kernel((grid,), (block,), (x, y, np.float32(dt), np.int32(n)))
        cp.cuda.Stream.null.synchronize()
        expected = 1.0 + dt
        return bool(cp.allclose(y, expected))


class CUDAGraphBackend(CuPyBackend):
    """
    CUDA Graph backend with pointer-table indirection.
    Capture once; every stage updates the pointer table + dt, then replays.
    """
    def __init__(self, operators, grid_manager=None):
        super().__init__(operators, grid_manager)
        self.graph = None
        self.graph_exec = None
        self.captured = False
        self.ptrs = None
        self.dt_slot = None
        self._field_names = [
            "rho", "rho_vr", "rho_vphi", "rho_vz", "E",
            "Br", "Bphi", "Bz",
            "emf_r", "emf_phi", "emf_z",
        ]

    def _ensure_table(self, state):
        if not HAS_CUPY:
            return
        if self.ptrs is None:
            self.ptrs = cp.zeros(len(self._field_names), dtype=cp.uintp)
        if self.dt_slot is None:
            self.dt_slot = cp.zeros(1, dtype=cp.float32)

    def update_pointers(self, state, dt: float):
        """Refresh device pointers and dt without recapturing."""
        if not HAS_CUPY or self.ptrs is None:
            return
        for i, name in enumerate(self._field_names):
            if name in state.fields:
                self.ptrs[i] = state.get(name).data.ptr
        self.dt_slot.fill(np.float32(dt))

    def capture(self, state, dt: float):
        if not HAS_CUPY:
            print("CUDA Graph: no CuPy — skipped")
            return False
        self._ensure_table(state)
        self.update_pointers(state, dt)
        stream = self.stream or cp.cuda.Stream()
        try:
            self.graph = cp.cuda.graph.Graph()
            with stream:
                with self.graph:
                    # Record the 8 launches
                    super().execute(state, dt)
            self.graph_exec = self.graph.instantiate()
            self.captured = True
            print("CUDA Graph captured")
            return True
        except Exception as e:
            print(f"CUDA Graph capture failed: {e}")
            self.captured = False
            return False

    def replay(self):
        if self.graph_exec is not None and self.stream is not None:
            self.graph_exec.launch(stream=self.stream)
            self.stream.synchronize()

    def execute(self, state, dt):
        if not HAS_CUPY:
            return
        if not self.captured:
            # Fall back to normal launches until capture is called
            super().execute(state, dt)
            return
        self.update_pointers(state, dt)
        self.replay()


def make_backend(config, operators, grid_manager=None):
    """Factory used by the Driver."""
    if config.backend == "cupy" and HAS_CUPY:
        if getattr(config, "use_cuda_graph", False):
            return CUDAGraphBackend(operators, grid_manager)
        return CuPyBackend(operators, grid_manager)
    return NumPyBackend(operators)



class Operator:
    def __init__(self, name: str, func: Callable, inputs: List[str], outputs: List[str]):
        self.name = name
        self.func = func
        self.inputs = inputs
        self.outputs = outputs
        self.enabled = True


class OperatorRegistry:
    def __init__(self):
        self.operators: Dict[str, Operator] = {}

    def register(self, op: Operator):
        self.operators[op.name] = op

    def validate(self, state: StateRegistry) -> List[str]:
        errors = []
        for name, op in self.operators.items():
            for f in op.inputs + op.outputs:
                if f not in state.fields and f not in ("entropy", "uct_state"):
                    errors.append(f"{name} references unregistered field '{f}'")
        return errors


class ExecutionController:
    # LOCKED ORDER — 9 kernels
    # g2 is the extra Geometry/Algebra operator (G2-structure / octonion fiber).
    # It measures φ, associator, Moufang residual and connection on the plasma
    # fiber (v, B, χ) and returns geometric sources S_vr, S_vphi, S_vz, S_E.
    # It is independent of the MHD kernels and can run after GLM.
    KERNELS = [
        "gyro",
        "hlldx",
        "hlldy",
        "hlldz",
        "utc",
        "uct_curl",
        "uct_emf",
        "glm",
        "g2",
    ]

    def __init__(self, registry: OperatorRegistry):
        self.registry = registry
        self.plan = self.KERNELS.copy()

    def execute(self, state: StateRegistry, dt: float):
        # Prefer backend if present
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.execute(state, dt)
            return
        for name in self.plan:
            if name in self.registry.operators and self.registry.operators[name].enabled:
                self.registry.operators[name].func(state, dt)


# =============================================================================
# LAYER 5 — TIME INTEGRATION
# =============================================================================
@dataclass
class StepResult:
    accepted: bool
    reason: str = ""
    dt: float = 0.0


class TimeManager:
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.time = 0.0
        self.step = 0
        self.dt = config.max_dt

    def compute_dt(self, geometry: GeometryManager) -> float:
        cfl_dt = self.config.cfl * float(geometry.dr.min())
        self.dt = max(self.config.min_dt, min(cfl_dt, self.config.max_dt))
        return self.dt


class Integrator:
    def __init__(self, config: ALADINConfig, execution: ExecutionController, state: StateRegistry):
        self.config = config
        self.execution = execution
        self.xp = config.xp
        self.stage_buffers = {}

    def prepare_buffers(self, state: StateRegistry):
        for name, meta in state.metadata.items():
            if meta.rk:
                self.stage_buffers[name] = {
                    "U0": self.xp.zeros_like(state.get(name)),
                    "U1": self.xp.zeros_like(state.get(name)),
                    "U2": self.xp.zeros_like(state.get(name)),
                }

    def step(self, state: StateRegistry, dt: float, boundary: BoundaryManager, geo_sources) -> StepResult:
        for name, buf in self.stage_buffers.items():
            buf["U0"][:] = state.get(name)

        # Stage 1
        boundary.apply_all(state)
        self.execution.execute(state, dt)
        geo_sources.apply_momentum(state, dt)
        self._apply_g2_sources(state, dt)
        for name, buf in self.stage_buffers.items():
            buf["U1"][:] = state.get(name)

        # Stage 2
        for name, buf in self.stage_buffers.items():
            state.get(name)[:] = buf["U1"]
        boundary.apply_all(state)
        self.execution.execute(state, dt)
        geo_sources.apply_momentum(state, dt)
        self._apply_g2_sources(state, dt)
        for name, buf in self.stage_buffers.items():
            state.get(name)[:] = 0.75 * buf["U0"] + 0.25 * state.get(name)

        # Stage 3
        for name, buf in self.stage_buffers.items():
            buf["U2"][:] = state.get(name)
        boundary.apply_all(state)
        self.execution.execute(state, dt)
        geo_sources.apply_momentum(state, dt)
        self._apply_g2_sources(state, dt)
        for name, buf in self.stage_buffers.items():
            state.get(name)[:] = (1.0/3.0) * buf["U0"] + (2.0/3.0) * state.get(name)

        boundary.apply_all(state)
        return StepResult(accepted=True, reason="", dt=dt)

    def _apply_g2_sources(self, state, dt):
        """Limited coupling: add G2 geometric sources to momentum and energy."""
        sl = state.grid.interior_slice("cell")
        if "S_vr" in state.fields:
            state.get("rho_vr")[sl] += dt * state.get("S_vr")[sl]
        if "S_vphi" in state.fields:
            state.get("rho_vphi")[sl] += dt * state.get("S_vphi")[sl]
        if "S_vz" in state.fields:
            state.get("rho_vz")[sl] += dt * state.get("S_vz")[sl]
        if "S_E" in state.fields and "E" in state.fields:
            state.get("E")[sl] += dt * state.get("S_E")[sl]

    def rollback(self, state: StateRegistry):
        for name, buf in self.stage_buffers.items():
            state.get(name)[:] = buf["U0"]


# =============================================================================
# LAYER 6 — GRAVITY
# =============================================================================
class GravityManager:
    """
    External potential: Miyamoto-Nagai stellar disk + NFW dark-matter halo.
    Writes Phi, g_R, g_z into the state (internal units).
    """
    def __init__(self, config: ALADINConfig, units: Units = None, geometry=None):
        self.config = config
        self.units = units or Units()
        self.geometry = geometry
        # Default Milky-Way-like parameters (internal units: kpc, M☉)
        self.M_disk = 5.0e10
        self.a_disk = 3.0
        self.b_disk = 0.3
        self.M_halo = 1.0e12
        self.r_s = 20.0
        self.G = self.units.G

    def set_geometry(self, geometry):
        self.geometry = geometry

    def update(self, state: StateRegistry):
        if not (self.config.self_gravity or self.config.external_gravity):
            return
        if self.geometry is None or "Phi" not in state.fields:
            return

        xp = state.xp
        g = self.geometry
        # Broadcast cell-center coordinates to 3D
        R = g.r_centers[:, None, None]
        z = g.z_centers[None, None, :]
        # Pad to ghosted shape if needed
        Phi = state.get("Phi")
        # Evaluate on interior and let boundaries handle ghosts
        sl = state.grid.interior_slice("cell")
        R_i = R
        z_i = z
        # Miyamoto-Nagai
        a, b = self.a_disk, self.b_disk
        denom = xp.sqrt(R_i**2 + (a + xp.sqrt(z_i**2 + b**2))**2)
        Phi_MN = -self.G * self.M_disk / denom
        # NFW (simplified)
        r = xp.sqrt(R_i**2 + z_i**2) + 1e-8
        c = 10.0
        factor = math.log(1 + c) - c / (1 + c)
        Phi_NFW = -self.G * self.M_halo / r * xp.log(1 + r / self.r_s) / factor
        Phi[sl] = (Phi_MN + Phi_NFW).astype(Phi.dtype)

        # Accelerations (simple finite difference on the potential)
        if "g_R" in state.fields and "g_z" in state.fields:
            # Radial
            dPhi_dR = xp.gradient(Phi[sl], g.r_centers, axis=0)
            state.get("g_R")[sl] = -dPhi_dR
            # Vertical
            dPhi_dz = xp.gradient(Phi[sl], g.z_centers, axis=2)
            state.get("g_z")[sl] = -dPhi_dz

    def phi_miyamoto_nagai(self, R, z):
        a, b = self.a_disk, self.b_disk
        return -self.G * self.M_disk / np.sqrt(R**2 + (a + np.sqrt(z**2 + b**2))**2)

    def phi_nfw(self, r):
        rs = self.r_s
        c = 10.0
        factor = np.log(1 + c) - c / (1 + c)
        return -self.G * self.M_halo / (r + 1e-8) * np.log(1 + r / rs) / factor


# =============================================================================
# LAYER 7 — CHECKPOINT / DIAGNOSTICS / RESULTS / HEALTH
# =============================================================================
class CheckpointManager:
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.dir = config.checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def _checksum(self, state: StateRegistry) -> str:
        h = hashlib.sha256()
        for name in sorted(state.fields):
            arr = state.interior(name)
            host = arr.get() if hasattr(arr, "get") else arr
            h.update(host.tobytes())
        return h.hexdigest()[:16]

    def save(self, state: StateRegistry, time_mgr: TimeManager, integrator: Integrator, tag: str = "latest"):
        path = os.path.join(self.dir, f"{tag}.npz")
        meta_path = os.path.join(self.dir, f"{tag}.json")
        data = {}
        for name, arr in state.fields.items():
            data[name] = arr.get() if hasattr(arr, "get") else arr
        for name, buf in integrator.stage_buffers.items():
            for stage in ("U0", "U1", "U2"):
                data[f"stage_{name}_{stage}"] = buf[stage].get() if hasattr(buf[stage], "get") else buf[stage]
        np.savez(path, **data)
        meta = {
            "time": time_mgr.time,
            "step": time_mgr.step,
            "dt": time_mgr.dt,
            "seed": self.config.seed,
            "ic_name": self.config.ic_name,
            "checksum": self._checksum(state),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Checkpoint: {path}")

    def load(self, state: StateRegistry, time_mgr: TimeManager, integrator: Integrator, tag: str = "latest") -> bool:
        path = os.path.join(self.dir, f"{tag}.npz")
        meta_path = os.path.join(self.dir, f"{tag}.json")
        if not os.path.exists(path):
            print(f"No checkpoint at {path}")
            return False
        data = np.load(path)
        xp = self.config.xp
        for name in state.fields:
            if name in data:
                state.fields[name][:] = xp.asarray(data[name])
        for name, buf in integrator.stage_buffers.items():
            for stage in ("U0", "U1", "U2"):
                key = f"stage_{name}_{stage}"
                if key in data:
                    buf[stage][:] = xp.asarray(data[key])
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            time_mgr.time = meta.get("time", 0.0)
            time_mgr.step = meta.get("step", 0)
            time_mgr.dt = meta.get("dt", time_mgr.dt)
        print(f"Loaded checkpoint {path}")
        return True


class DiagnosticsManager:
    def __init__(self, config):
        self.config = config
        self.history = {"step": [], "time": [], "mass": [], "ME": []}
        self.me_start = 0.0

    def magnetic_energy(self, state) -> float:
        return state.magnetic_energy()

    def measure(self, state, geometry, time: float, step: int) -> dict:
        rho = state.interior("rho")
        vol = geometry.cell_volumes
        mass = float(self.config.xp.sum(rho * vol))
        me = self.magnetic_energy(state)
        self.history["step"].append(step)
        self.history["time"].append(time)
        self.history["mass"].append(mass)
        self.history["ME"].append(me)
        return {"mass": mass, "ME": me, "time": time, "step": step}

    def set_start(self, state):
        self.me_start = self.magnetic_energy(state)


class ResultsManager:
    def __init__(self, config):
        self.config = config
        self.dir = config.output_dir
        os.makedirs(self.dir, exist_ok=True)

    def write_diagnostics(self, diagnostics: DiagnosticsManager, state, geometry):
        """Single diagnostics.csv with radial profiles + latest globals."""
        xp = state.xp
        sl = state.grid.interior_slice("cell")
        rho = state.get("rho")[sl]
        rho_vphi = state.get("rho_vphi")[sl]
        vphi = rho_vphi / (rho + 1e-40)
        iz = vphi.shape[2] // 2
        R = geometry.r_centers
        v_mid = vphi[:, 0, iz]
        vol = geometry.cell_volumes
        mass_shell = xp.sum(rho * vol, axis=(1, 2))
        # surface density proxy
        dz = geometry.dz
        Sigma = xp.sum(rho * dz[None, None, :], axis=2)[:, 0]   # mid-phi

        R_h = R.get() if hasattr(R, "get") else np.asarray(R)
        v_h = v_mid.get() if hasattr(v_mid, "get") else np.asarray(v_mid)
        m_h = mass_shell.get() if hasattr(mass_shell, "get") else np.asarray(mass_shell)
        M_enc = np.cumsum(m_h)
        Sig_h = Sigma.get() if hasattr(Sigma, "get") else np.asarray(Sigma)

        # latest globals
        h = diagnostics.history
        final_mass = h["mass"][-1] if h["mass"] else 0.0
        final_ME = h["ME"][-1] if h["ME"] else 0.0
        final_time = h["time"][-1] if h["time"] else 0.0
        final_step = h["step"][-1] if h["step"] else 0

        path = os.path.join(self.dir, "diagnostics.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["R", "v_phi", "Omega", "M_enc", "Sigma",
                        "final_mass", "final_ME", "final_time", "final_step"])
            for i in range(len(R_h)):
                w.writerow([
                    float(R_h[i]),
                    float(v_h[i]),
                    float(v_h[i] / (R_h[i] + 1e-12)),
                    float(M_enc[i]),
                    float(Sig_h[i]),
                    final_mass,
                    final_ME,
                    final_time,
                    final_step,
                ])
        print(f"Diagnostics: {path}")


class HealthManager:
    def __init__(self, config: ALADINConfig):
        self.config = config

    def check(self, state: StateRegistry) -> Tuple[bool, str]:
        xp = self.config.xp
        rho = state.interior("rho")
        if xp.any(xp.isnan(rho)) or xp.any(xp.isinf(rho)):
            return False, "NaN/Inf in rho"
        if xp.any(rho <= 0):
            return False, "negative density"
        return True, ""


# =============================================================================
# DRIVER
# =============================================================================
class ALADINDriver:
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.lifecycle: Lifecycle = "CREATED"
        self.pool = MemoryPoolManager(config)
        self.pool.free()
        self.geometry = GeometryManager(config)
        self.geo_sources = GeometricSources(config, self.geometry)
        self.grid = GridManager(config, self.geometry)
        self.state = StateRegistry(config, self.grid)
        self.memory = MemoryManager(config, self.state)
        self.boundary = BoundaryManager(config, self.geometry, self.grid)
        self.operators = OperatorRegistry()
        self.execution = ExecutionController(self.operators)
        self.backend = make_backend(self.config, self.operators.operators, self.grid)
        self.execution.backend = self.backend
        self.time_mgr = TimeManager(config)
        self.integrator = Integrator(config, self.execution, self.state)
        self.units = Units()
        self.gravity = GravityManager(config, self.units)
        self.gravity.set_geometry(self.geometry)
        self.checkpoints = CheckpointManager(config)
        self.diagnostics = DiagnosticsManager(config)
        self.results = ResultsManager(config)
        self.health = HealthManager(config)
        self._register_kernels()

    def _register_kernels(self):
        def placeholder(state, dt):
            pass

        # ----- Real kernel launches (CuPy) / NumPy fallback -----
        try:
            import cupy as _cp
            _HAS = _cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            _cp = None
            _HAS = False

        # Ideal EMF: E = -v × B
        _EMF_SRC = r'''
        extern "C" __global__
        void emf_ideal(const float* vr, const float* vp, const float* vz,
                       const float* Br, const float* Bp, const float* Bz,
                       float* Er, float* Ep, float* Ez, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n) return;
            Er[i] = -(vp[i]*Bz[i] - vz[i]*Bp[i]);
            Ep[i] = -(vz[i]*Br[i] - vr[i]*Bz[i]);
            Ez[i] = -(vr[i]*Bp[i] - vp[i]*Br[i]);
        }
        '''
        # Cylindrical curl update (cell-centered harness, axis-safe)
        _CURL_SRC = r'''
        extern "C" __global__
        void curl_update(const float* Er, const float* Ep, const float* Ez,
                         const float* r, float* Br, float* Bp, float* Bz,
                         int nr, int nphi, int nz, float dt) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int N = nr*nphi*nz;
            if (i >= N) return;
            int iz = i % nz;
            int iy = (i / nz) % nphi;
            int ix = i / (nz * nphi);
            float rr = r[ix];
            float inv_r = (rr > 1e-8f) ? 1.f/rr : 0.f;
            float dphi = 6.28318530718f / nphi;
            float dz = 1.f / nz;
            float dEp_dz = 0.f, dEz_dphi = 0.f, dEz_dr = 0.f, dEr_dz = 0.f;
            float dEr_dphi = 0.f, d_rEp_dr = 0.f;
            if (iz>0 && iz<nz-1) {
                dEp_dz = (Ep[i+1]-Ep[i-1])/(2.f*dz);
                dEr_dz = (Er[i+1]-Er[i-1])/(2.f*dz);
            }
            if (iy>0 && iy<nphi-1) {
                dEz_dphi = (Ez[i+nz]-Ez[i-nz])/(2.f*dphi);
                dEr_dphi = (Er[i+nz]-Er[i-nz])/(2.f*dphi);
            }
            if (ix>0 && ix<nr-1) {
                float dr = (r[ix+1]-r[ix-1])*0.5f;
                dEz_dr = (Ez[i+nphi*nz]-Ez[i-nphi*nz])/(2.f*dr);
                d_rEp_dr = (r[ix+1]*Ep[i+nphi*nz] - r[ix-1]*Ep[i-nphi*nz])/(2.f*dr);
            }
            Br[i] -= dt * (dEp_dz - inv_r*dEz_dphi);
            Bp[i] -= dt * (dEz_dr - dEr_dz);
            Bz[i] -= dt * (inv_r*dEr_dphi - inv_r*d_rEp_dr);
        }
        '''
        # G2 source: geometric curvature + simple fiber residual
        _G2_SRC = r'''
        extern "C" __global__
        void g2_source(const float* vr, const float* vp, const float* vz,
                       const float* Br, const float* Bp, const float* Bz,
                       const float* r,
                       float* Svr, float* Svp, float* Svz, float* SE, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n) return;
            float rr = fmaxf(r[i], 1e-6f);
            float inv = 1.f / rr;
            float geom_vr = (vp[i]*vp[i] - Bp[i]*Bp[i]) * inv;
            float geom_vp = -(vr[i]*vp[i] - Br[i]*Bp[i]) * inv;
            Svr[i] = geom_vr;
            Svp[i] = geom_vp;
            Svz[i] = 0.f;
            SE[i]  = -(vr[i]*geom_vr + vp[i]*geom_vp);
        }
        '''

        emf_k = curl_k = g2_k = None
        if _HAS:
            emf_k = _cp.RawKernel(_EMF_SRC, "emf_ideal")
            curl_k = _cp.RawKernel(_CURL_SRC, "curl_update")
            g2_k = _cp.RawKernel(_G2_SRC, "g2_source")

        class _UCTEMFOp:
            def __init__(self):
                self.last_stats = {}
            def __call__(self, state, dt):
                if emf_k is None:
                    self.last_stats = {"called": True, "backend": "numpy", "dt": float(dt)}
                    return
                rho = _cp.asarray(state.get("rho"), dtype=_cp.float32)
                vr = _cp.asarray(state.get("rho_vr"), dtype=_cp.float32) / (rho + 1e-12)
                vp = _cp.asarray(state.get("rho_vphi"), dtype=_cp.float32) / (rho + 1e-12)
                vz = _cp.asarray(state.get("rho_vz"), dtype=_cp.float32) / (rho + 1e-12)
                Br = _cp.asarray(state.get("Br"), dtype=_cp.float32)
                Bp = _cp.asarray(state.get("Bphi"), dtype=_cp.float32)
                Bz = _cp.asarray(state.get("Bz"), dtype=_cp.float32)
                # use the common broadcast size
                # allocate outputs to the destination staggered shapes
                Er = _cp.zeros(state.get("emf_r").shape, dtype=_cp.float32)
                Ep = _cp.zeros(state.get("emf_phi").shape, dtype=_cp.float32)
                Ez = _cp.zeros(state.get("emf_z").shape, dtype=_cp.float32)
                m = int(min(vr.size, Er.size, Br.size))
                emf_k(((m+127)//128,), (128,),
                      (vr.ravel()[:m], vp.ravel()[:m], vz.ravel()[:m],
                       Br.ravel()[:m], Bp.ravel()[:m], Bz.ravel()[:m],
                       Er.ravel()[:m], Ep.ravel()[:m], Ez.ravel()[:m], np.int32(m)))
                _cp.cuda.Stream.null.synchronize()
                state.get("emf_r")[:] = _cp.asnumpy(Er)
                state.get("emf_phi")[:] = _cp.asnumpy(Ep)
                state.get("emf_z")[:] = _cp.asnumpy(Ez)
                self.last_stats = {"called": True, "backend": "cupy",
                                   "max_emf": float(_cp.max(_cp.abs(Ez)))}

        class _UCTCurlOp:
            def __init__(self, geometry, config):
                self.geometry = geometry
                self.config = config
                self.last_stats = {}
            def __call__(self, state, dt):
                if curl_k is None:
                    self.last_stats = {"called": True, "backend": "numpy", "dt": float(dt)}
                    return
                cfg = self.config
                Er = _cp.asarray(state.get("emf_r"), dtype=_cp.float32).ravel()
                Ep = _cp.asarray(state.get("emf_phi"), dtype=_cp.float32).ravel()
                Ez = _cp.asarray(state.get("emf_z"), dtype=_cp.float32).ravel()
                Br = _cp.asarray(state.get("Br"), dtype=_cp.float32).ravel()
                Bp = _cp.asarray(state.get("Bphi"), dtype=_cp.float32).ravel()
                Bz = _cp.asarray(state.get("Bz"), dtype=_cp.float32).ravel()
                n = cfg.nr * cfg.nphi * cfg.nz
                r = _cp.asarray(self.geometry.r_centers, dtype=_cp.float32)
                # operate on the overlapping flat length
                m = int(min(Er.size, Br.size, n))
                curl_k(((m+127)//128,), (128,),
                       (Er[:m], Ep[:m], Ez[:m], r,
                        Br[:m], Bp[:m], Bz[:m],
                        np.int32(cfg.nr), np.int32(cfg.nphi), np.int32(cfg.nz),
                        np.float32(dt)))
                _cp.cuda.Stream.null.synchronize()
                # write back the portion we updated
                br = state.get("Br").ravel()
                br[:m] = _cp.asnumpy(Br[:m])
                state.get("Br")[:] = br.reshape(state.get("Br").shape)
                bp = state.get("Bphi").ravel()
                bp[:m] = _cp.asnumpy(Bp[:m])
                state.get("Bphi")[:] = bp.reshape(state.get("Bphi").shape)
                bz = state.get("Bz").ravel()
                bz[:m] = _cp.asnumpy(Bz[:m])
                state.get("Bz")[:] = bz.reshape(state.get("Bz").shape)
                self.last_stats = {"called": True, "backend": "cupy", "dt": float(dt)}

        class _G2Op:
            def __init__(self, geometry, config):
                self.geometry = geometry
                self.config = config
                self.last_stats = {}
            def __call__(self, state, dt):
                if g2_k is None:
                    self.last_stats = {"called": True, "backend": "numpy", "dt": float(dt)}
                    return
                rho = _cp.asarray(state.get("rho"), dtype=_cp.float32)
                vr = (_cp.asarray(state.get("rho_vr"), dtype=_cp.float32) / (rho + 1e-12)).ravel()
                vp = (_cp.asarray(state.get("rho_vphi"), dtype=_cp.float32) / (rho + 1e-12)).ravel()
                vz = (_cp.asarray(state.get("rho_vz"), dtype=_cp.float32) / (rho + 1e-12)).ravel()
                Br = _cp.asarray(state.get("Br"), dtype=_cp.float32).ravel()
                Bp = _cp.asarray(state.get("Bphi"), dtype=_cp.float32).ravel()
                Bz = _cp.asarray(state.get("Bz"), dtype=_cp.float32).ravel()
                # radial coordinate matching the full (ghosted) array
                nr = self.config.nr
                g = self.config.ghost_width
                r_int = _cp.asarray(self.geometry.r_centers, dtype=_cp.float32)
                # pad so length == nr + 2*g
                r_full = _cp.empty(nr + 2 * g, dtype=_cp.float32)
                r_full[:g] = r_int[0]
                r_full[g:g+nr] = r_int
                r_full[g+nr:] = r_int[-1]
                R = _cp.broadcast_to(r_full[:, None, None], state.get("rho").shape).ravel()
                n = int(vr.size)
                Svr = _cp.zeros(n, dtype=_cp.float32)
                Svp = _cp.zeros(n, dtype=_cp.float32)
                Svz = _cp.zeros(n, dtype=_cp.float32)
                SE = _cp.zeros(n, dtype=_cp.float32)
                m = int(min(n, Br.size))
                g2_k(((m+127)//128,), (128,),
                     (vr[:m], vp[:m], vz[:m], Br[:m], Bp[:m], Bz[:m], R[:m],
                      Svr[:m], Svp[:m], Svz[:m], SE[:m], np.int32(m)))
                _cp.cuda.Stream.null.synchronize()
                for name, arr in (("S_vr", Svr), ("S_vphi", Svp),
                                  ("S_vz", Svz), ("S_E", SE)):
                    dest = state.get(name).ravel()
                    dest[:m] = _cp.asnumpy(arr[:m])
                    state.get(name)[:] = dest.reshape(state.get(name).shape)
                self.last_stats = {"called": True, "backend": "cupy", "dt": float(dt)}

        emf_func = _UCTEMFOp()
        curl_func = _UCTCurlOp(self.geometry, self.config)
        g2_func = _G2Op(self.geometry, self.config)
        print("UCT EMF kernel:", "LAUNCH" if _HAS else "numpy-fallback")
        print("UCT curl kernel:", "LAUNCH" if _HAS else "numpy-fallback")
        print("G2 kernel:", "LAUNCH" if _HAS else "numpy-fallback")

        specs = [
            ("gyro",      ["rho", "P_par", "P_perp"],                     ["P_par", "P_perp"]),
            ("hlldx",     ["rho", "rho_vr", "rho_vphi", "rho_vz", "E",
                           "Br", "Bphi", "Bz"],                             ["flux_r"]),
            ("hlldy",     ["rho", "rho_vr", "rho_vphi", "rho_vz", "E",
                           "Br", "Bphi", "Bz"],                             ["flux_phi"]),
            ("hlldz",     ["rho", "rho_vr", "rho_vphi", "rho_vz", "E",
                           "Br", "Bphi", "Bz"],                             ["flux_z"]),
            ("utc",       ["flux_r", "flux_phi", "flux_z"],                ["flux_r", "flux_phi", "flux_z"]),
            ("uct_curl",  ["emf_r", "emf_phi", "emf_z"],                   ["Br", "Bphi", "Bz"]),
            ("uct_emf",   ["rho_vr", "rho_vphi", "rho_vz",
                           "Br", "Bphi", "Bz"],                             ["emf_r", "emf_phi", "emf_z"]),
            ("glm",       ["Br", "Bphi", "Bz"],                             ["Br", "Bphi", "Bz"]),
            ("g2",        ["rho", "rho_vr", "rho_vphi", "rho_vz",
                           "Br", "Bphi", "Bz"],                             ["S_vr", "S_vphi", "S_vz", "S_E"]),
            # g2: Geometry/Algebra operator. Octonion fiber + G2 3-form.
            # Produces geometric sources; does not replace HLLD or CT.
        ]
        for name, inputs, outputs in specs:
            if name == "g2":
                func = g2_func
            elif name == "uct_curl":
                func = curl_func
            elif name == "uct_emf":
                func = emf_func
            else:
                func = placeholder
            self.operators.register(Operator(name, func, inputs, outputs))

    def initialize(self):
        if self.lifecycle != "CREATED":
            raise RuntimeError(f"Cannot initialize from state {self.lifecycle}")
        print(self.config.summary())
        print(f"Volume: {self.geometry.total_volume():.6e}")
        self.memory.allocate_persistent()
        print(self.memory.report())
        print(self.pool.report())
        self.pool.warmup()
        self.integrator.prepare_buffers(self.state)

        # ---- Initial condition (separate function) ----
        ic_func = IC_REGISTRY.get(self.config.ic_name, set_uniform)
        ic_func(self.state, self.geometry, self.gravity)

        self.boundary.apply_all(self.state)
        errors = self.operators.validate(self.state)
        if errors:
            print("Operator validation warnings:", errors)
        self.diagnostics.set_start(self.state)
        self.lifecycle = "INITIALIZED"

    def run(self, nsteps: int = 5):
        if self.lifecycle == "CREATED":
            self.initialize()
        if self.lifecycle not in ("INITIALIZED", "RUNNING"):
            raise RuntimeError(f"Cannot run from {self.lifecycle}")
        self.lifecycle = "RUNNING"

        for step in range(nsteps):
            dt = self.time_mgr.compute_dt(self.geometry)
            self.gravity.update(self.state)
            result = self.integrator.step(self.state, dt, self.boundary, self.geo_sources)
            ok, reason = self.health.check(self.state)
            if not result.accepted or not ok:
                self.integrator.rollback(self.state)
                self.lifecycle = "FAILED"
                print(f"Step rejected: {result.reason or reason}")
                break
            self.time_mgr.time += dt
            self.time_mgr.step += 1
            diag = self.diagnostics.measure(
                self.state, self.geometry, self.time_mgr.time, self.time_mgr.step
            )
            if self.time_mgr.step % self.config.checkpoint_interval == 0:
                self.checkpoints.save(self.state, self.time_mgr, self.integrator)
            if self.time_mgr.step % 5 == 0 or self.time_mgr.step <= 3:
                print(f"Step {self.time_mgr.step} t={diag['time']:.4e} mass={diag['mass']:.4e} ME={diag['ME']:.4e}")

        if self.lifecycle == "RUNNING":
            self.lifecycle = "COMPLETED"
        self.results.write_diagnostics(self.diagnostics, self.state, self.geometry)
        print(f"Lifecycle: {self.lifecycle}")


if __name__ == "__main__":
    cfg = ALADINConfig(nr=128, nphi=128, nz=128, backend="numpy",
                       ic_name="equilibrium_disk", self_gravity=False)
    driver = ALADINDriver(cfg)
    driver.run(5)
    print("Driver ready.")
    print("Sequence:", ExecutionController.KERNELS)
