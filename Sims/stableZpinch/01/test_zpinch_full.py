"""
ALADIN — Z-pinch test with CFL-limited dt and force-balance diagnostic

Requires:
  hlld_x_kernel.py
  hlld_y_kernel.py
  hlld_z_kernel.py
  residual_cylindrical.py
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

class Config:
    nr, nphi, nz = 48, 8, 8
    ghost_width = 2
    xp = cp if HAS_CUPY else np
    dtype = np.float32
    cfl = 0.3
    gamma = 5.0/3.0

class Geometry:
    def __init__(self, cfg):
        xp = cfg.xp
        g = cfg.ghost_width
        self.r_faces_phys = xp.linspace(0.0, 1.0, cfg.nr + 1, dtype=cfg.dtype)
        self.r_centers = 0.5 * (self.r_faces_phys[:-1] + self.r_faces_phys[1:])
        self.dr = self.r_faces_phys[1:] - self.r_faces_phys[:-1]
        self.dphi = xp.full(cfg.nphi, 2*np.pi / cfg.nphi, dtype=cfg.dtype)
        self.dz = xp.full(cfg.nz, 1.0 / cfg.nz, dtype=cfg.dtype)
        r_out = self.r_faces_phys[1:]
        r_in = self.r_faces_phys[:-1]
        self.cell_volumes = (0.5*(r_out**2 - r_in**2))[:, None, None] * \
                            self.dphi[None, :, None] * self.dz[None, None, :]
        self.area_r = self.r_faces_phys[:, None, None] * self.dphi[None, :, None] * self.dz[None, None, :]
        self.area_phi = (r_out - r_in)[:, None, None] * self.dz[None, None, :]
        self.area_z = 0.5*(r_out**2 - r_in**2)[:, None, None] * self.dphi[None, :, None] * xp.ones((1, 1, cfg.nz + 1), dtype=cfg.dtype)
        self.r_faces = xp.concatenate([
            xp.full(g, 0.0, dtype=cfg.dtype),
            self.r_faces_phys,
            xp.full(g, 1.0, dtype=cfg.dtype)
        ])

class State:
    def __init__(self, cfg, geometry):
        self.cfg = cfg
        xp = cfg.xp
        g = cfg.ghost_width
        nr, nphi, nz = cfg.nr, cfg.nphi, cfg.nz
        shape_c = (nr + 2*g, nphi + 2*g, nz + 2*g)
        shape_fr = (nr + 1 + 2*g, nphi + 2*g, nz + 2*g)
        shape_fp = (nr + 2*g, nphi + 1 + 2*g, nz + 2*g)
        shape_fz = (nr + 2*g, nphi + 2*g, nz + 1 + 2*g)
        self.fields = {
            "rho":      xp.ones(shape_c, dtype=cfg.dtype),
            "rho_vr":   xp.zeros(shape_c, dtype=cfg.dtype),
            "rho_vphi": xp.zeros(shape_c, dtype=cfg.dtype),
            "rho_vz":   xp.zeros(shape_c, dtype=cfg.dtype),
            "E":        xp.ones(shape_c, dtype=cfg.dtype) * 1.0,
            "Br":       xp.zeros(shape_fr, dtype=cfg.dtype),
            "Bphi":     xp.zeros(shape_c, dtype=cfg.dtype),
            "Bz":       xp.zeros(shape_c, dtype=cfg.dtype),
        }
        class _Grid:
            def interior_slice(self, centering="cell"):
                return (slice(g, -g), slice(g, -g), slice(g, -g))
        self.grid = _Grid()
        self._shapes = {"r": shape_fr, "phi": shape_fp, "z": shape_fz}

    def get(self, name):
        return self.fields[name]

def set_zpinch(state, geometry, cfg, B0=0.5, rho0=1.0, p0=0.1):
    xp = cfg.xp
    sl = state.grid.interior_slice()
    r = geometry.r_centers[:, None, None]
    rho = rho0 * xp.exp(-(r / 0.3)**2)
    state.get("rho")[sl] = rho
    Bphi = B0 * (r / 0.25) * xp.exp(-(r / 0.4)**2)
    state.get("Bphi")[sl] = Bphi
    state.get("E")[sl] = p0 / (cfg.gamma - 1.0) + 0.5 * Bphi**2

def cfl_dt(state, geometry, cfg):
    """Fast-magnetosonic CFL on the radial direction."""
    xp = cfg.xp
    sl = state.grid.interior_slice()
    rho = xp.maximum(state.get("rho")[sl], 1e-6)
    vr = state.get("rho_vr")[sl] / rho
    vphi = state.get("rho_vphi")[sl] / rho
    vz = state.get("rho_vz")[sl] / rho
    Bphi = state.get("Bphi")[sl]
    Bz = state.get("Bz")[sl]
    # rough thermal + magnetic
    p = xp.maximum((cfg.gamma - 1.0) * (state.get("E")[sl] - 0.5*rho*(vr**2+vphi**2+vz**2) - 0.5*(Bphi**2+Bz**2)), 1e-8)
    cs2 = cfg.gamma * p / rho
    ca2 = (Bphi**2 + Bz**2) / rho
    cf = xp.sqrt(xp.maximum(cs2 + ca2, 0.0))
    speed = xp.abs(vr) + cf
    dr = geometry.dr[:, None, None]
    dt_local = cfg.cfl * dr / xp.maximum(speed, 1e-6)
    return float(xp.min(dt_local))

def force_balance(state, geometry, cfg):
    """
    Radial force balance diagnostic.
    Returns arrays of r, centrifugal, Lorentz (~ -Bφ²/r), pressure gradient.
    """
    xp = cfg.xp
    sl = state.grid.interior_slice()
    r = geometry.r_centers
    rho = xp.maximum(state.get("rho")[sl][:, 0, 0], 1e-6)
    vphi = state.get("rho_vphi")[sl][:, 0, 0] / rho
    Bphi = state.get("Bphi")[sl][:, 0, 0]
    # centrifugal
    cent = vphi**2 / (r + 1e-6)
    # magnetic tension (dominant for pure Bφ)
    lorentz = -(Bphi**2) / (r + 1e-6)
    # pressure gradient (simple)
    p = xp.maximum((cfg.gamma - 1.0) * state.get("E")[sl][:, 0, 0], 1e-8)
    dp_dr = xp.gradient(p, r)
    press = -dp_dr / rho
    return r, cent, lorentz, press

def run_zpinch_full(nsteps=20):
    from hlld_x_kernel import launch_hlld_x
    from hlld_y_kernel import launch_hlld_y
    from hlld_z_kernel import launch_hlld_z
    from residual_cylindrical import residual_cylindrical

    cfg = Config()
    geom = Geometry(cfg)
    state = State(cfg, geom)
    set_zpinch(state, geom, cfg)

    keys = ["rho", "vr", "vphi", "vz", "E", "Bphi", "Bz", "Br"]
    fluxes = {
        "r":   {k: cfg.xp.zeros(state._shapes["r"],   dtype=cfg.dtype) for k in keys},
        "phi": {k: cfg.xp.zeros(state._shapes["phi"], dtype=cfg.dtype) for k in keys},
        "z":   {k: cfg.xp.zeros(state._shapes["z"],   dtype=cfg.dtype) for k in keys},
    }

    print("=== Z-PINCH TEST (HLLD x/y/z + CFL) ===")
    print(f"grid: {cfg.nr}x{cfg.nphi}x{cfg.nz}  B0=0.5  cfl={cfg.cfl}")

    def sanitize():
        xp = cfg.xp
        for name in ("rho", "rho_vr", "rho_vphi", "rho_vz", "E", "Bphi", "Bz"):
            if name in state.fields:
                arr = state.get(name)
                arr[:] = xp.where(xp.isfinite(arr), arr, 0.0)
        state.get("rho")[:] = xp.maximum(state.get("rho"), 1e-6)

    for step in range(nsteps):
        sanitize()
        dt = cfl_dt(state, geom, cfg)
        launch_hlld_x(state, geom, cfg, fluxes["r"])
        launch_hlld_y(state, cfg, fluxes["phi"])
        launch_hlld_z(state, cfg, fluxes["z"])
        residual_cylindrical(state, geom, cfg, fluxes, dt)
        sanitize()
        if step % 5 == 0 or step == nsteps-1:
            print(f"  step {step:3d}  dt={dt:.3e}")

    # final force balance
    r, cent, lorentz, press = force_balance(state, geom, cfg)
    sl = state.grid.interior_slice()
    rho = state.get("rho")[sl][:, 0, 0]
    vr  = state.get("rho_vr")[sl][:, 0, 0] / cfg.xp.maximum(rho, 1e-6)

    print("\n=== FORCE BALANCE (mid-plane) ===")
    print("  r     ρ      vr       cent      Lorentz    -dp/ρ")
    for i in range(0, cfg.nr, 6):
        print(f"  {float(r[i]):.2f}  {float(rho[i]):.3f}  {float(vr[i]):+.2e}  "
              f"{float(cent[i]):+.2e}  {float(lorentz[i]):+.2e}  {float(press[i]):+.2e}")
    print("Done.")

if __name__ == "__main__":
    run_zpinch_full()
