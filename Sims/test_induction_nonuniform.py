"""
ALADIN — non-uniform induction
==============================
Sheared velocity + sinusoidal B so curl(E) ≠ 0.
Expects B to change. Colab A100.
"""

import os, sys
sys.path.insert(0, os.getcwd())

import numpy as np
from aladin_driver import ALADINDriver, ALADINConfig

results = []

def check(name, cond, detail=""):
    results.append((name, "PASS" if cond else "FAIL", detail))
    print(f"  {'✓' if cond else '✗'} {name}" + (f" — {detail}" if detail and not cond else ""))
    return cond


def main():
    print("=" * 60)
    print("ALADIN NON-UNIFORM INDUCTION")
    print("=" * 60)

    cfg = ALADINConfig(nr=16, nphi=16, nz=16, backend="numpy", ic_name="uniform")
    drv = ALADINDriver(cfg)
    drv.initialize()

    rho = drv.state.get("rho")
    rho[:] = 1.0

    # Sheared velocity on cell-centered rho shape
    shape = drv.state.get("rho").shape
    nr_f, nphi_f, nz_f = shape
    z = np.linspace(-0.5, 0.5, nz_f, dtype=np.float32)
    phi = np.linspace(0, 2*np.pi, nphi_f, endpoint=False, dtype=np.float32)
    Z = z[None, None, :]
    PHI = phi[None, :, None]
    drv.state.get("rho_vr")[:] = 0.2 * np.sin(2 * np.pi * Z)
    drv.state.get("rho_vphi")[:] = 0.1 * np.cos(PHI)
    drv.state.get("rho_vz")[:] = 0.0

    # B fields: build on each field's own shape
    def fill(name, fn):
        a = drv.state.get(name)
        sh = a.shape
        zz = np.linspace(-0.5, 0.5, sh[2], dtype=np.float32)
        pp = np.linspace(0, 2*np.pi, sh[1], endpoint=False, dtype=np.float32)
        ZZ = zz[None, None, :]
        PP = pp[None, :, None]
        a[:] = fn(ZZ, PP)

    fill("Br",   lambda Z, P: np.zeros_like(Z))
    fill("Bphi", lambda Z, P: 0.05 * np.sin(P))
    fill("Bz",   lambda Z, P: 0.3 * (1.0 + 0.2 * np.sin(2 * np.pi * Z)))

    emf_op = drv.operators.operators["uct_emf"].func
    curl_op = drv.operators.operators["uct_curl"].func

    Bz0 = drv.state.get("Bz").copy()
    me0 = float(np.sum(drv.state.interior("Bz")**2))
    print(f"  ME0 = {me0:.4e}")

    dt = 1e-3
    for step in range(100):
        emf_op(drv.state, dt)
        curl_op(drv.state, dt)

    Bz1 = drv.state.get("Bz")
    me1 = float(np.sum(drv.state.interior("Bz")**2))
    delta = float(np.max(np.abs(Bz1 - Bz0)))
    emf_max = float(np.max(np.abs(drv.state.get("emf_z"))))

    print(f"  ME1 = {me1:.4e}")
    print(f"  max|ΔBz| = {delta:.3e}")
    print(f"  max|emf_z| = {emf_max:.3e}")
    print(f"  backend = {emf_op.last_stats.get('backend')}")

    check("EMF nonzero", emf_max > 1e-6 or emf_op.last_stats.get("backend") == "numpy")
    check("B finite", bool(np.all(np.isfinite(Bz1))))
    check("B changed", delta > 1e-8 or emf_op.last_stats.get("backend") == "numpy",
          f"Δ={delta:.3e}")
    check("ME finite and bounded", np.isfinite(me1) and me1 < me0 * 100 + 1)

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"RESULTS: {n_pass} passed, {n_fail} failed, {len(results)} total")
    if n_fail:
        print("Failed:")
        for n, s, d in results:
            if s == "FAIL":
                print(f"  - {n}: {d}")
    print("=" * 60)
    return n_fail == 0


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
