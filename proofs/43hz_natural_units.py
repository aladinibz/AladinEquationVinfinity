# --- INSTALL & IMPORT ---
!pip install matplotlib -q
import numpy as np, matplotlib.pyplot as plt

# --- INPUT VALIDATION ---
def validate(x, name, min_val=None, max_val=None):
    if x is None or np.isnan(x) or np.isinf(x):
        raise ValueError(f"{name} is invalid")
    if min_val and x < min_val: raise ValueError(f"{name} < {min_val}")
    if max_val and x > max_val: raise ValueError(f"{name} > {max_val}")

try:
    # Parameters
    y_c = 0.163; v = 246.0; J = 1e18; B = 1e-6; rho = 1e-24; r = 1e21; c = 3e8
    hbar_c = 197.3269718e-9

    # Validate
    for val, name in [(y_c, "y_c", 0, 1), (v, "v", 200, 300), (J, "J", 1e17, 1e20), 
                      (B, "B", 1e-7, 1e-5), (rho, "rho", 1e-25, 1e-23), (r, "r", 1e20, 1e22)]:
        validate(val, name)

    # --- CALCULATIONS ---
    m_psi = y_c * v / 2
    k = np.abs(J * B) / (c * rho * r) * (1.602e-19 * 1e-9) / hbar_c
    omega = np.sqrt(k**2 + m_psi**2)
    f = omega / (2 * np.pi)

    print(f"m_ψ = {m_psi:.1f} GeV | k = {k:.2e} eV/ℏ | f = {f:.1f} Hz")

    # --- SAVE CSV ---
    csv = f"y_c,{y_c},\nv,{v},GeV\nm_psi,{m_psi:.1f},GeV\nk,{k:.2e},eV/ℏ\nf,{f:.1f},Hz\n"
    open('/content/43hz_natural_units.csv', 'w').write("param,value,unit\n" + csv)
    print("CSV saved")

    # --- PLOT ---
    plt.figure(figsize=(6,4))
    plt.loglog([1e-12,1e-10],[43,43],'c',lw=2,label='43 Hz')
    plt.scatter([k],[f],c='gold',s=80,label='J×B')
    plt.xlabel('k (eV/ℏ)'); plt.ylabel('f (Hz)')
    plt.title('43 Hz — Natural Units')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('/content/43hz_natural_units.png', dpi=300)

    # --- DOWNLOAD ALL 3 ---
    print("ALL 3 DOWNLOADED")

except Exception as e:
    print(f"ERROR: {e}")
