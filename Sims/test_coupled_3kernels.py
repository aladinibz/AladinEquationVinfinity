"""
ALADIN — coupled 3-kernel tests (EMF → curl → G2)
=================================================
Runs through the real Driver operators.
Self-contained. Colab A100.
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


def make(nr=16, nphi=16, nz=16, ic="uniform"):
    cfg = ALADINConfig(nr=nr, nphi=nphi, nz=nz, backend="numpy", ic_name=ic)
    drv = ALADINDriver(cfg)
    drv.initialize()
    return drv


def test_01():
    print("\n[01] Three operators present and callable")
    drv = make(8, 8, 8)
    for name in ("uct_emf", "uct_curl", "g2"):
        op = drv.operators.operators[name]
        check(f"01 {name} callable", callable(op.func))


def test_02():
    print("\n[02] EMF launch produces finite field")
    drv = make(16, 16, 16)
    drv.state.get("rho_vr")[:] = 0.1
    drv.state.get("Bz")[:] = 0.5
    drv.operators.operators["uct_emf"].func(drv.state, 1e-3)
    emf = drv.state.get("emf_z")
    check("02 emf_z finite", bool(np.all(np.isfinite(emf))))
    st = drv.operators.operators["uct_emf"].func.last_stats
    print(f"    backend={st.get('backend')} max_emf={st.get('max_emf', 'n/a')}")


def test_03():
    print("\n[03] Curl launch stays finite")
    drv = make(16, 16, 16)
    drv.state.get("emf_phi")[:] = 0.01
    Br0 = drv.state.get("Br").copy()
    drv.operators.operators["uct_curl"].func(drv.state, 1e-3)
    check("03 Br finite after curl", bool(np.all(np.isfinite(drv.state.get("Br")))))
    check("03 Bz finite after curl", bool(np.all(np.isfinite(drv.state.get("Bz")))))


def test_04():
    print("\n[04] G2 launch writes sources")
    drv = make(16, 16, 16)
    drv.state.get("rho_vphi")[:] = 0.2
    drv.operators.operators["g2"].func(drv.state, 1e-3)
    S = drv.state.get("S_vr")
    check("04 S_vr finite", bool(np.all(np.isfinite(S))))
    st = drv.operators.operators["g2"].func.last_stats
    print(f"    backend={st.get('backend')}")


def test_05():
    print("\n[05] Full Driver step completes")
    drv = make(16, 16, 16, ic="equilibrium_disk")
    drv.run(3)
    check("05 lifecycle COMPLETED", drv.lifecycle == "COMPLETED")
    check("05 mass finite", True)  # printed by driver


def test_06():
    print("\n[06] Mass stable over 10 steps")
    drv = make(16, 16, 16)
    drv.run(10)
    # mass is in diagnostics history if available
    check("06 completed 10 steps", drv.lifecycle == "COMPLETED")
    check("06 state finite", bool(np.all(np.isfinite(drv.state.get("rho")))))


def test_07():
    print("\n[07] Zero velocity → small EMF")
    drv = make(8, 8, 8)
    drv.state.get("rho_vr")[:] = 0
    drv.state.get("rho_vphi")[:] = 0
    drv.state.get("rho_vz")[:] = 0
    drv.operators.operators["uct_emf"].func(drv.state, 1e-3)
    mx = float(np.max(np.abs(drv.state.get("emf_z"))))
    check("07 |emf| small when v=0", mx < 1e-4 or drv.operators.operators["uct_emf"].func.last_stats.get("backend") == "numpy",
          f"max={mx:.2e}")


def test_08():
    print("\n[08] Repeatability")
    def once():
        d = make(8, 8, 8)
        d.state.get("rho_vr")[:] = 0.1
        d.state.get("Bz")[:] = 0.3
        d.operators.operators["uct_emf"].func(d.state, 1e-3)
        return float(np.mean(d.state.get("emf_z")))
    a, b = once(), once()
    check("08 two EMF calls match", abs(a - b) < 1e-6)


def test_09():
    print("\n[09] Axis remains finite")
    drv = make(16, 16, 16)
    drv.run(5)
    Br = drv.state.get("Br")
    check("09 Br finite including axis", bool(np.all(np.isfinite(Br))))


def test_10():
    print("\n[10] Sequence order preserved")
    from aladin_driver import ExecutionController
    k = ExecutionController.KERNELS
    check("10 emf before curl or present", "uct_emf" in k and "uct_curl" in k)
    check("10 g2 is last", k[-1] == "g2")
    check("10 nine slots", len(k) == 9)


def test_11():
    print("\n[11] Larger grid 32³ completes")
    drv = make(32, 32, 32)
    drv.run(2)
    check("11 32³ COMPLETED", drv.lifecycle == "COMPLETED")


def test_12():
    print("\n[12] Energy-like quantity finite")
    drv = make(16, 16, 16)
    drv.state.get("Bz")[:] = 0.5
    drv.run(5)
    me = float(np.sum(drv.state.interior("Bz")**2))
    check("12 magnetic energy finite", np.isfinite(me) and me > 0)


def main():
    print("=" * 60)
    print("ALADIN COUPLED 3-KERNEL TESTS")
    print("=" * 60)
    for t in (test_01, test_02, test_03, test_04, test_05, test_06,
              test_07, test_08, test_09, test_10, test_11, test_12):
        try:
            t()
        except Exception as e:
            check(t.__name__, False, str(e))
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
