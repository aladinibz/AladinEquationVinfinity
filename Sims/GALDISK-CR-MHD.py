import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 GALDISK-CR-MHD v1.0 — Dimensionless Constrained Solver")

# ===================== DIMENSIONLESS UNITS =====================
N = 128
L = 1.0                 # Rd = 1 code unit
dx = L / N

x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

r = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
r_cyl = cp.sqrt(X**2 + Y**2 + 1e-12)

# ===================== PHYSICAL PARAMETERS (ONLY REAL ONES) =====================
gamma = 5/3
G = 1.0                # dimensionless (set by scaling)
beta0 = 10.0          # initial plasma beta (ONLY magnetization control)
kappa_par = 0.01      # CR diffusion (physical transport parameter)

steps = 600
CFL = 0.15

# ===================== INITIAL CONDITIONS =====================
rho = cp.exp(-r_cyl * 5) * cp.exp(-cp.abs(Z)*10)

vx = cp.zeros_like(rho)
vy = cp.sqrt(1/r_cyl) * (1 - cp.exp(-r_cyl*8))   # Kepler-like seed
vz = cp.zeros_like(rho)

p_th = 0.01 * rho

u_cr = 0.3 * rho

# Magnetic seed from beta constraint
B_mag = cp.sqrt(2 * p_th / beta0)
Bx = cp.zeros_like(rho)
By = B_mag * (-Y / r_cyl)
Bz = cp.zeros_like(rho)

# ===================== MAIN LOOP =====================
for step in range(steps):

    v2 = vx**2 + vy**2 + vz**2

    # ----------------- MAG FIELD -----------------
    Ex = -(vy * Bz - vz * By)
    Ey = -(vz * Bx - vx * Bz)
    Ez = -(vx * By - vy * Bx)

    Bx += cp.gradient(Ez, dx, axis=1) - cp.gradient(Ey, dx, axis=2)
    By += cp.gradient(Ex, dx, axis=2) - cp.gradient(Ez, dx, axis=0)
    Bz += cp.gradient(Ey, dx, axis=0) - cp.gradient(Ex, dx, axis=1)

    B2 = Bx**2 + By**2 + Bz**2

    # ----------------- PRESSURES -----------------
    p_cr = (gamma - 1) * u_cr
    P_tot = p_th + p_cr

    # ----------------- FORCES (ONLY PHYSICAL) -----------------
    Jx = cp.gradient(Bz, dx, axis=1) - cp.gradient(By, dx, axis=2)
    Jy = cp.gradient(Bx, dx, axis=2) - cp.gradient(Bz, dx, axis=0)
    Jz = cp.gradient(By, dx, axis=0) - cp.gradient(Bx, dx, axis=1)

    JxB_x = Jy * Bz - Jz * By
    JxB_y = Jz * Bx - Jx * Bz
    JxB_z = Jx * By - Jy * Bx

    # Pressure gradients
    Fx = (JxB_x - cp.gradient(P_tot, dx, axis=0)) / rho
    Fy = (JxB_y - cp.gradient(P_tot, dx, axis=1)) / rho
    Fz = (JxB_z - cp.gradient(P_tot, dx, axis=2)) / rho

    # Gravity (Poisson)
    Phi_k = cp.fft.fftn(rho)
    kx = 2*np.pi*cp.fft.fftfreq(N, d=dx)
    KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2 + 1e-8

    Phi = cp.real(cp.fft.ifftn(-4*np.pi*G*Phi_k / k2))

    Fx += -cp.gradient(Phi, dx, axis=0)
    Fy += -cp.gradient(Phi, dx, axis=1)
    Fz += -cp.gradient(Phi, dx, axis=2)

    # ----------------- UPDATE VELOCITY -----------------
    vx += Fx * CFL
    vy += Fy * CFL
    vz += Fz * CFL

    # ----------------- CONTINUITY -----------------
    div_v = cp.gradient(vx, dx, axis=0) + cp.gradient(vy, dx, axis=1) + cp.gradient(vz, dx, axis=2)
    rho -= rho * div_v * CFL

    rho = cp.clip(rho, 1e-6, 10)

    # ----------------- CR TRANSPORT -----------------
    lap_cr = sum(cp.gradient(cp.gradient(u_cr, dx, axis=i), dx, axis=i) for i in range(3))
    u_cr += kappa_par * lap_cr - u_cr * div_v * CFL

# ===================== OUTPUT =====================
r_mid = r_cyl[:, :, N//2].get()
v_phi = ((X*vy - Y*vx)/r_cyl)[:, :, N//2].get()

plt.plot(r_mid.flatten(), v_phi.flatten(), '.')
plt.xlabel("Radius")
plt.ylabel("v_phi")
plt.title("GALDISK-CR-MHD v1.0 Rotation Curve")
plt.show()
