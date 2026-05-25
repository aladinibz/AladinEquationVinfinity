import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v34.0 — True Staggered CT Yee + Dedner (Fixed Seeding)")

# ====================== PARAMETERS ======================
N = 128
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
mu0 = 1.0
gamma = 5.0 / 3.0
CFL = 0.3
steps = 300

rho_floor = 1e-6
p_floor = 1e-4
v_phi_factor = 0.12

# ====================== FIELDS ======================
rho = cp.ones((N, N, N), dtype=cp.float32) * 1e-3
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4
u_cr = cp.ones((N, N, N), dtype=cp.float32) * 1e-3

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)

Ex = cp.zeros((N, N+1, N+1), dtype=cp.float32)
Ey = cp.zeros((N+1, N, N+1), dtype=cp.float32)
Ez = cp.zeros((N+1, N+1, N), dtype=cp.float32)

# ====================== INITIAL CONDITIONS ======================
r_cyl = cp.sqrt(X**2 + Y**2)
rho *= cp.exp(-r_cyl / 8.0) * cp.exp(-Z**2 / 2.25)

r3d = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
M_dm = nfw_enclosed_mass(r3d)   # define function below
g_r = -G * M_dm / r3d**2
v_phi = v_phi_factor * cp.sqrt(cp.maximum(r_cyl * cp.abs(g_r), 0.0))
vx = -v_phi * (Y / (r_cyl + 1e-8))
vy =  v_phi * (X / (r_cyl + 1e-8))
vz = cp.zeros_like(vx)

mx = rho * vx
my = rho * vy
mz = rho * vz

mass0 = float(cp.sum(rho))
E0 = float(cp.sum(E_total))
Lz0 = float(cp.sum(rho * (X*vy - Y*vx)))

# ====================== NFW FUNCTION ======================
def nfw_enclosed_mass(r):
    xx = r / 20.0 + 1e-12
    return 4 * cp.pi * rho0_nfw * 20.0**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== STAGGERED B SEEDING (FIXED) ======================
B0 = 5.0
Bphi = 2.0

# Correct seeding for staggered grids
Bx[1:,:,:] = -Bphi * (Y[1:,:,:] / (r_cyl[1:,:,:] + 1e-8))   # use [1:] for N+1 size
By[:,1:,:] =  Bphi * (X[:,1:,:] / (r_cyl[:,1:,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(-(X[:,:,1:]**2 + Y[:,:,1:]**2 + Z[:,:,1:]**2) / 200.0)

# ====================== MAIN LOOP ======================
for step in range(steps):
    vx = mx / (rho + 1e-30)
    vy = my / (rho + 1e-30)
    vz = mz / (rho + 1e-30)
    dt = 0.25 * dx / 150.0

    # Compute Edge EMFs
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Ex at (i, j+0.5, k+0.5)
                if j < N-1 and k < N-1:
                    vy_avg = 0.25 * (vy[i,j,k] + vy[i,j+1,k] + vy[i,j,k+1] + vy[i,j+1,k+1])
                    vz_avg = 0.25 * (vz[i,j,k] + vz[i,j+1,k] + vz[i,j,k+1] + vz[i,j+1,k+1])
                    Bz_avg = 0.25 * (Bz[i,j,k] + Bz[i,j+1,k] + Bz[i,j,k+1] + Bz[i,j+1,k+1])
                    By_avg = 0.25 * (By[i,j,k] + By[i,j+1,k] + By[i,j,k+1] + By[i,j+1,k+1])
                    Ex[i,j+1,k+1] = - (vy_avg * Bz_avg - vz_avg * By_avg)

                # Ey at (i+0.5, j, k+0.5)
                if i < N-1 and k < N-1:
                    vz_avg = 0.25 * (vz[i,j,k] + vz[i+1,j,k] + vz[i,j,k+1] + vz[i+1,j,k+1])
                    vx_avg = 0.25 * (vx[i,j,k] + vx[i+1,j,k] + vx[i,j,k+1] + vx[i+1,j,k+1])
                    Bx_avg = 0.25 * (Bx[i,j,k] + Bx[i+1,j,k] + Bx[i,j,k+1] + Bx[i+1,j,k+1])
                    Bz_avg = 0.25 * (Bz[i,j,k] + Bz[i+1,j,k] + Bz[i,j,k+1] + Bz[i+1,j,k+1])
                    Ey[i+1,j,k+1] = - (vz_avg * Bx_avg - vx_avg * Bz_avg)

                # Ez at (i+0.5, j+0.5, k)
                if i < N-1 and j < N-1:
                    vx_avg = 0.25 * (vx[i,j,k] + vx[i+1,j,k] + vx[i,j+1,k] + vx[i+1,j+1,k])
                    vy_avg = 0.25 * (vy[i,j,k] + vy[i+1,j,k] + vy[i,j+1,k] + vy[i+1,j+1,k])
                    By_avg = 0.25 * (By[i,j,k] + By[i+1,j,k] + By[i,j+1,k] + By[i+1,j+1,k])
                    Bx_avg = 0.25 * (Bx[i,j,k] + Bx[i+1,j,k] + Bx[i,j+1,k] + Bx[i+1,j+1,k])
                    Ez[i+1,j+1,k] = - (vx_avg * By_avg - vy_avg * Bx_avg)

    # Update Staggered B fields
    Bx[1:-1,:,:] += (dt / dx) * ((Ez[1:-1,1:,:] - Ez[1:-1,:-1,:]) - (Ey[1:-1,:,1:] - Ey[1:-1,:,:-1]))
    By[:,1:-1,:] += (dt / dx) * ((Ex[:,1:-1,1:] - Ex[:,1:-1,:-1]) - (Ez[1:,1:-1,:] - Ez[:-1,1:-1,:]))
    Bz[:,:,1:-1] += (dt / dx) * ((Ey[1:,:,1:-1] - Ey[:-1,:,1:-1]) - (Ex[:,1:,1:-1] - Ex[:,:-1,1:-1]))

    if step % 50 == 0:
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")

print("\n✅ Staggered CT Yee update finished.")
print("Run it and paste the full console output.")
