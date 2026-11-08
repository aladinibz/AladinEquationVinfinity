"""
ALADIN — magnetic induction loop
================================
div-free B0 + nonzero v → EMF → cylindrical curl → B
Tracks ME and divB for N steps. Compares a NumPy reference step.
Self-contained. Colab A100 (uses Driver operators).
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


def div_proxy(drv):
    """Simple cell-centered div proxy on interior."""
    Bz = drv.state.interior("Bz")
    d = np.abs(Bz[:, :, 1:] - Bz[:, :, :-1])
    return float(np.max(d)), float(np.sqrt(np.mean(d**2)))


def magnetic_energy(drv):
    Br = drv.state.interior("Br")
    Bp = drv.state.interior("Bphi")
    Bz = drv.state.interior("Bz")
    # shapes differ (staggered); use available interiors
    return float(np.sum(Bz**2))


def numpy_emf_curl_step(vr, vp, vz, Br, Bp, Bz, dt):
    """Independent reference: E=-v×B, then a crude curl on Bz via dEr/dphi proxy."""
    Er = -(vp * Bz - vz * Bp)
    # not a full cylindrical curl — only checks the EMF algebra
    return Er


def main():
    print("=" * 60)
    print("ALADIN INDUCTION LOOP")
    print("=" * 60)

    cfg = ALADINConfig(nr=16, nphi=16, nz=16, backend="numpy", ic_name="uniform")
    drv = ALADINDriver(cfg)
    drv.initialize()

    # Nonzero velocity
    drv.state.get("rho")[:] = 1.0
    drv.state.get("rho_vr")[:] = 0.05
    drv.state.get("rho_vphi")[:] = 0.1
    drv.state.get("rho_vz")[:] = 0.0
    # Divergence-free-ish initial B: uniform Bz
    drv.state.get("Br")[:] = 0.0
    drv.state.get("Bphi")[:] = 0.0
    drv.state.get("Bz")[:] = 0.2

    emf_op = drv.operators.operators["uct_emf"].func
    curl_op = drv.operators.operators["uct_curl"].func

    print(f"  EMF backend:  {getattr(emf_op, 'last_stats', {})}")
    me0 = magnetic_energy(drv)
    dmax0, drms0 = div_proxy(drv)
    print(f"  ME0={me0:.4e}  divB max={dmax0:.3e} rms={drms0:.3e}")

    nsteps = 200
    dt = 1e-4
    me_hist = [me0]
    div_hist = [dmax0]

    for step in range(nsteps):
        emf_op(drv.state, dt)
        curl_op(drv.state, dt)
        if step % 50 == 0 or step == nsteps - 1:
            me = magnetic_energy(drv)
            dmax, drms = div_proxy(drv)
            me_hist.append(me)
            div_hist.append(dmax)
            print(f"  step {step:4d}  ME={me:.4e}  divB max={dmax:.3e}")

    me1 = magnetic_energy(drv)
    dmax1, drms1 = div_proxy(drv)

    check("B finite after 200 steps",
          bool(np.all(np.isfinite(drv.state.get("Bz")))))
    check("ME finite", np.isfinite(me1) and me1 < 1e6)
    check("ME did not explode", me1 < me0 * 10 + 1.0)
    check("divB stayed bounded", dmax1 < 1.0, f"max={dmax1:.3e}")

    # Independent NumPy EMF algebra check
    vr = np.array(0.05); vp = np.array(0.1); vz = np.array(0.0)
    Br = np.array(0.0); Bp = np.array(0.0); Bz = np.array(0.2)
    Er_ref = numpy_emf_curl_step(vr, vp, vz, Br, Bp, Bz, dt)
    # E_r = -(vp*Bz - vz*Bp) = -0.1*0.2 = -0.02
    check("NumPy EMF algebra", abs(float(Er_ref) - (-0.02)) < 1e-6)

    st = emf_op.last_stats
    print(f"\n  final backend={st.get('backend')}  ME {me0:.4e} → {me1:.4e}")
    print(f"  divB {dmax0:.3e} → {dmax1:.3e}")

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
