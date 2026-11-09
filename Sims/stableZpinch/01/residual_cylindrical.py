"""
ALADIN Plasma Drive v2.1
Cylindrical Residual Operator

Differences HLLD face fluxes onto cells.
Only processes directions present in the fluxes dict.
Geometry areas and volumes are physical (no ghosts).
Areas are broadcast to match the flux difference.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


def residual_cylindrical(state, geometry, config, fluxes, dt):
    """
    Finite-volume update from face fluxes.

    fluxes may contain any subset of "r", "phi", "z".
    Each is a dict of ghosted face arrays.
    geometry.area_* and cell_volumes are physical.
    """
    g = config.ghost_width
    sl = state.grid.interior_slice("cell")
    vol = geometry.cell_volumes   # (nr, nphi, nz)

    def _align(A, target_shape):
        """Broadcast or tile A to target_shape if needed."""
        if A.shape == target_shape:
            return A
        # squeeze trailing singleton dims then broadcast
        while A.ndim > 0 and A.shape[-1] == 1 and A.ndim > len(target_shape):
            A = A.squeeze(-1)
        return np.broadcast_to(A, target_shape) if not HAS_CUPY else cp.broadcast_to(A, target_shape)

    def _clean(arr):
        """Replace NaN/Inf with zero so one bad face does not poison the run."""
        if HAS_CUPY:
            return cp.where(cp.isfinite(arr), arr, 0.0)
        return np.where(np.isfinite(arr), arr, 0.0)

    def div_r(flux_face):
        f = _clean(flux_face[g:-g, g:-g, g:-g])
        A = _align(geometry.area_r, f.shape)
        return (f[1:] * A[1:] - f[:-1] * A[:-1]) / vol

    def div_phi(flux_face):
        f = _clean(flux_face[g:-g, g:-g, g:-g])
        A = _align(geometry.area_phi, f.shape)
        return (f[:, 1:] * A[:, 1:] - f[:, :-1] * A[:, :-1]) / vol

    def div_z(flux_face):
        f = _clean(flux_face[g:-g, g:-g, g:-g])
        A = _align(geometry.area_z, f.shape)
        return (f[:, :, 1:] * A[:, :, 1:] - f[:, :, :-1] * A[:, :, :-1]) / vol

    def update(field_name, flux_key):
        if field_name not in state.fields:
            return
        d = 0.0
        if "r" in fluxes and flux_key in fluxes["r"]:
            d = d + div_r(fluxes["r"][flux_key])
        if "phi" in fluxes and flux_key in fluxes["phi"]:
            d = d + div_phi(fluxes["phi"][flux_key])
        if "z" in fluxes and flux_key in fluxes["z"]:
            d = d + div_z(fluxes["z"][flux_key])
        state.get(field_name)[sl] -= dt * d

    update("rho",      "rho")
    update("rho_vr",   "vr")
    update("rho_vphi", "vphi")
    update("rho_vz",   "vz")
    update("E",        "E")

    # NaN-safe positivity floor on the FULL array (kernels read ghosts)
    if "rho" in state.fields:
        rho = state.get("rho")
        if HAS_CUPY:
            rho[:] = cp.where(cp.isfinite(rho), rho, 1e-6)
            rho[:] = cp.maximum(rho, 1e-6)
        else:
            rho[:] = np.where(np.isfinite(rho), rho, 1e-6)
            rho[:] = np.maximum(rho, 1e-6)
    for name in ("rho_vr", "rho_vphi", "rho_vz", "E"):
        if name in state.fields:
            arr = state.get(name)
            if HAS_CUPY:
                arr[:] = cp.where(cp.isfinite(arr), arr, 0.0)
            else:
                arr[:] = np.where(np.isfinite(arr), arr, 0.0)


if __name__ == "__main__":
    print("Cylindrical residual operator ready.")
    print("Usage: residual_cylindrical(state, geometry, config, fluxes, dt)")
