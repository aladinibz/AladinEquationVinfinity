"""
Colab-ready G2 algebra test
Runtime → Change runtime type → GPU (A100 / T4)
Then run this cell.
"""
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy available — running on GPU")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("CuPy not found — falling back to NumPy")

# ---------------------------------------------------------------------------
# Kernel (paste or import)
# ---------------------------------------------------------------------------
if HAS_CUPY:
    from g2_geometry_algebra_kernel import g2_geometry_algebra_kernel
else:
    g2_geometry_algebra_kernel = None

def run_kernel(rho, vr, vphi, vz, Br, Bphi, Bz,
               lambda_g2=1.0, projection_strength=0.5, rho_min=1e-6):
    if not HAS_CUPY:
        raise RuntimeError("GPU kernel requires CuPy")
    N = rho.size
    S_vr   = cp.zeros_like(rho)
    S_vphi = cp.zeros_like(rho)
    S_vz   = cp.zeros_like(rho)
    S_E    = cp.zeros_like(rho)
    block = 128
    grid  = (N + block - 1) // block
    g2_geometry_algebra_kernel(
        (grid,), (block,),
        (rho, vr, vphi, vz, Br, Bphi, Bz,
         S_vr, S_vphi, S_vz, S_E,
         np.int32(N), np.float32(lambda_g2),
         np.float32(projection_strength), np.float32(rho_min))
    )
    return S_vr, S_vphi, S_vz, S_E

# ---------------------------------------------------------------------------
# Algebra probe (Fano φ, projectors, Newton residual)
# ---------------------------------------------------------------------------
def phi_fano(vx, vy, vz, bx, by, bz, chi):
    return (vx*vy*vz + vx*bx*by + vx*bz*chi
          + vy*bx*bz + vy*by*chi + vz*bx*chi + vz*by*bz)

N = 200_000
rng = np.random.default_rng(42)
rho  = cp.ones(N, dtype=cp.float32) if HAS_CUPY else np.ones(N, dtype=np.float32)
vr   = cp.asarray(rng.standard_normal(N).astype(np.float32))
vphi = cp.asarray(rng.standard_normal(N).astype(np.float32))
vz   = cp.asarray(rng.standard_normal(N).astype(np.float32))
Br   = cp.asarray(rng.standard_normal(N).astype(np.float32))
Bphi = cp.asarray(rng.standard_normal(N).astype(np.float32))
Bz   = cp.asarray(rng.standard_normal(N).astype(np.float32))

# Before
B2  = Br*Br + Bphi*Bphi + Bz*Bz + 1e-12
chi = (Br*vr + Bphi*vphi + Bz*vz) / (rho * B2)
phi0 = phi_fano(vr, vphi, vz, Br, Bphi, Bz, chi)
mean_phi0 = float(cp.mean(cp.abs(phi0)) if HAS_CUPY else np.mean(np.abs(phi0)))

# After kernel
if HAS_CUPY:
    S_vr, S_vphi, S_vz, S_E = run_kernel(rho, vr, vphi, vz, Br, Bphi, Bz,
                                         lambda_g2=1.0, projection_strength=0.8)
    # Recompute phi on the modified fields is not direct; measure source magnitude
    # and energy consistency instead
    energy_err = cp.abs(S_E + (vr*S_vr + vphi*S_vphi + vz*S_vz))
    max_energy_err = float(cp.max(energy_err))
    mean_source = float(cp.mean(cp.abs(S_vr) + cp.abs(S_vphi) + cp.abs(S_vz)))
    print(f"Mean |φ| before          : {mean_phi0:.3e}")
    print(f"Mean |source| after      : {mean_source:.3e}")
    print(f"Max energy inconsistency : {max_energy_err:.3e}")
    print("GPU algebra probe complete.")
else:
    print("No GPU — install CuPy or switch to GPU runtime.")
    print(f"Mean |φ| before (CPU) : {mean_phi0:.3e}")
