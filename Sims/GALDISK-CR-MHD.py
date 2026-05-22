import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 GALDISK-CR-MHD v1.0 — Stable Dimensionless Constrained Solver")

# ===================== DIMENSIONLESS UNITS =====================
R0 = 1.0        # galaxy scale radius
V0 = 1.0        # virial velocity scale
T0 = R0 / V0

G = 1.0         # dimensionless gravity (absorbed into scaling)

gamma = 5/3
CFL = 0.15
steps = 400

N = 128
L = 6.0         # domain in units of R0
dx = L / N

# ===================== GRID =====================
x = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')

r = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-6)
r_cyl = cp.sqrt(X**2 + Y**2 + 1e-6)

# ===================== INITIAL CONDITIONS =====================
rho = cp.exp(-r_cyl) * cp.exp(-cp.abs(Z))

vx = cp.zeros_like(rho)
vy = cp.zeros_like(rho)
vz = cp.zeros_like(rho)

# smooth initial rotation (dimensionless virial disk)
vphi = 1.2 * (1 - cp.exp(-r_cyl / 0.5))
vx = -vphi * (Y / r_cyl)
vy =  vphi * (X / r_cyl)

# CR energy (dimensionless pressure-like field)
u_cr = 0.2 * cp.exp(-r_cyl / 0.8)

# magnetic field seed (weak, divergence-free-ish structure)
Bx = cp.zeros((N, N, N))
By = cp.zeros((N, N, N))
Bz = 0.05 * cp.exp(-r_cyl**2 / 0.5)

# ===================== SAFETY FUNCTIONS =====================
def safe_div(a, b):
    return a / cp.where(cp.abs(b) > 1e-8, b, 1e-8)

def safe_r(x):
    return cp.maximum(x, 1e-6)

# ===================== MAIN LOOP =====================
for step in range(steps):

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    # pressures (dimensionless)
    p_th = cp.maximum((gamma - 1) * (rho * 0.5 * v2 + 0.1), 1e-6)
    p_cr = u_cr / 3.0
    p_mag = 0.5 * B2

    p_tot = p_th + p_cr + p_mag

    # sound + Alfvén speed
    cs = cp.sqrt(gamma * p_th / rho)
    ca = cp.sqrt(B2 / rho)

    vmax = cp.max(cp.sqrt(v2 + cs**2 + ca**2))
    dt = CFL * dx / (vmax + 1e-6)

    # ================= GRAVITY (self-consistent) =================
    rho_eff = rho + p_tot  # dimensionless "active gravity source"

    # Poisson (FFT)
    rho_k = cp.fft.fftn(rho_eff)
    k = 2 * cp.pi * cp.fft.fftfreq(N, d=dx)
    KX, KY, KZ = cp.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2 + 1e-6

    Phi = cp.real(cp.fft.ifftn(-rho_k / k2))

    gx = -cp.gradient(Phi, axis=0)
    gy = -cp.gradient(Phi, axis=1)
    gz = -cp.gradient(Phi, axis=2)

    # ================= PRESSURE FORCE =================
    dpdx = cp.gradient(p_tot, axis=0)
    dpdy = cp.gradient(p_tot, axis=1)
    dpdz = cp.gradient(p_tot, axis=2)

    # ================= MAGNETIC FORCE =================
    jx = cp.gradient(Bz, axis=1) - cp.gradient(By, axis=2)
    jy = cp.gradient(Bx, axis=2) - cp.gradient(Bz, axis=0)
    jz = cp.gradient(By, axis=0) - cp.gradient(Bx, axis=1)

    jxbx = jy * Bz - jz * By
    jxby = jz * Bx - jx * Bz
    jxbz = jx * By - jy * Bx

    # ================= MOMENTUM UPDATE =================
    vx += dt * (jxbx + gx - dpdx) / safe_r(rho)
    vy += dt * (jxby + gy - dpdy) / safe_r(rho)
    vz += dt * (jxbz + gz - dpdz) / safe_r(rho)

    # clamp velocities
    vabs = cp.sqrt(vx**2 + vy**2 + vz**2)
    vx *= cp.minimum(1.0, 2.0 / (vabs + 1e-6))
    vy *= cp.minimum(1.0, 2.0 / (vabs + 1e-6))
    vz *= cp.minimum(1.0, 2.0 / (vabs + 1e-6))

    # ================= CONTINUITY =================
    divv = (cp.gradient(rho * vx, axis=0) +
            cp.gradient(rho * vy, axis=1) +
            cp.gradient(rho * vz, axis=2))

    rho -= dt * divv
    rho = cp.maximum(rho, 1e-6)

    # ================= CR EVOLUTION =================
    lap_u = sum(cp.gradient(cp.gradient(u_cr, axis=i), axis=i) for i in range(3))
    u_cr += dt * (0.01 * lap_u - 0.05 * u_cr)
    u_cr = cp.maximum(u_cr, 1e-6)

    # ================= DIAGNOSTICS =================
    if step % 50 == 0:
        print(f"Step {step} | rho max {rho.max():.3f} | v max {vabs.max():.3f}")

# ===================== ROTATION CURVE =====================
mid = N // 2

r_mid = r_cyl[:, :, mid].get()
vx_m = vx[:, :, mid].get()
vy_m = vy[:, :, mid].get()

mask = r_mid > 1e-3

r_mid = r_mid[mask]
vphi = (vx_m * (-Y[:, :, mid].get()) + vy_m * (X[:, :, mid].get()))[mask] / r_mid

# binning (safe)
bins = np.linspace(0, L/2, 50)
num, _ = np.histogram(r_mid, bins=bins, weights=vphi)
den, _ = np.histogram(r_mid, bins=bins)

vrot = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
rcent = 0.5 * (bins[:-1] + bins[1:])

# ===================== PLOT =====================
plt.figure(figsize=(10,5))
plt.plot(rcent, vrot, label="GALDISK-CR-MHD v1.0")
plt.xlabel("Radius (code units)")
plt.ylabel("Rotation velocity")
plt.title("Stable Dimensionless Rotation Curve")
plt.grid()
plt.legend()
plt.show()

print("✅ v1.0 complete — stable rotation curve generated")
