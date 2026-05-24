import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v0.5 — FULL COMPLETE FIXED INDEXING")

# ====================== PARAMETERS ======================
N = 256
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

G = 4.302e-3
mu0 = 1.0
gamma = 5.0 / 3.0
CFL = 0.25
steps = 800
rho_floor = 1e-6
p_floor = 1e-4
v_phi_factor = 0.10
c_h_factor = 5.0
kappa_factor = 0.5

M_vir = 1.2e12
c_nfw = 12.0
r_s = 20.0
rho0_nfw = M_vir / (4 * cp.pi * r_s**3 * (cp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)))

def nfw_enclosed_mass(r):
    xx = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + xx) - xx / (1 + xx))

# ====================== FIELDS ======================
rho = cp.ones((N, N, N), dtype=cp.float32) * 1e-3
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4
u_cr = cp.ones((N, N, N), dtype=cp.float32) * 1e-5
Bx = cp.zeros((N, N, N), dtype=cp.float32)
By = cp.zeros((N, N, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N), dtype=cp.float32)
psi = cp.zeros((N, N, N), dtype=cp.float32)

# ====================== SEEDING ======================
r_cyl = cp.sqrt(X**2 + Y**2)
B0 = 5.0
Bphi = 2.0
Bx = -Bphi * (Y / (r_cyl + 1e-8))
By =  Bphi * (X / (r_cyl + 1e-8))
Bz = B0 * cp.exp(-(X**2 + Y**2 + Z**2) / 200.0)

# ====================== INITIAL ROTATION ======================
r3d = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
M_dm = nfw_enclosed_mass(r3d)
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
px0 = float(cp.sum(mx))
py0 = float(cp.sum(my))
pz0 = float(cp.sum(mz))

# ====================== MUSCL ======================
def minmod(a, b):
    return 0.5 * (cp.sign(a) + cp.sign(b)) * cp.minimum(cp.abs(a), cp.abs(b))

def muscl_reconstruct(U, axis):
    if axis == 0:
        dL = U[1:-1,:,:] - U[:-2,:,:]
        dR = U[2:,:,:] - U[1:-1,:,:]
        slope = minmod(dL, dR)
        return U[1:-1,:,:] - 0.5*slope, U[1:-1,:,:] + 0.5*slope
    elif axis == 1:
        dL = U[:,1:-1,:] - U[:,:-2,:]
        dR = U[:,2:,:] - U[:,1:-1,:]
        slope = minmod(dL, dR)
        return U[:,1:-1,:] - 0.5*slope, U[:,1:-1,:] + 0.5*slope
    else:
        dL = U[:,:,1:-1] - U[:,:,:-2]
        dR = U[:,:,2:] - U[:,:,1:-1]
        slope = minmod(dL, dR)
        return U[:,:,1:-1] - 0.5*slope, U[:,:,1:-1] + 0.5*slope

# ====================== FULL HLLD ======================
def hlld_flux(rho_L, rho_R, mx_L, mx_R, my_L, my_R, mz_L, mz_R, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R):
    B2_L = Bx_L**2 + By_L**2 + Bz_L**2
    B2_R = Bx_R**2 + By_R**2 + Bz_R**2
    p_L = (gamma - 1.0) * (E_L - 0.5*rho_L*((mx_L/rho_L)**2 + (my_L/rho_L)**2 + (mz_L/rho_L)**2) - B2_L/(2*mu0))
    p_R = (gamma - 1.0) * (E_R - 0.5*rho_R*((mx_R/rho_R)**2 + (my_R/rho_R)**2 + (mz_R/rho_R)**2) - B2_R/(2*mu0))
    vx_L = mx_L/rho_L; vx_R = mx_R/rho_R
    vy_L = my_L/rho_L; vy_R = my_R/rho_R
    vz_L = mz_L/rho_L; vz_R = mz_R/rho_R

    c_fL = cp.sqrt((gamma*p_L + B2_L)/rho_L)
    c_fR = cp.sqrt((gamma*p_R + B2_R)/rho_R)
    S_L = cp.minimum(vx_L - c_fL, vx_R - c_fR)
    S_R = cp.maximum(vx_L + c_fL, vx_R + c_fR)
    S_star = (S_L*rho_L*vx_L - S_R*rho_R*vx_R + p_R - p_L) / (rho_L*(S_L-vx_L) - rho_R*(S_R-vx_R) + 1e-12)

    rho_starL = rho_L * (S_L - vx_L) / (S_L - S_star + 1e-12)
    rho_starR = rho_R * (S_R - vx_R) / (S_R - S_star + 1e-12)
    p_star = p_L + rho_L*(S_L - vx_L)*(S_star - vx_L)
    Bx_star = Bx_L
    denom = rho_L*(S_L - vx_L) + rho_R*(S_R - vx_R)
    By_star = (rho_L*By_L*(S_L-vx_L) + rho_R*By_R*(S_R-vx_R) + Bx_L*By_L - Bx_R*By_R) / denom
    Bz_star = (rho_L*Bz_L*(S_L-vx_L) + rho_R*Bz_R*(S_R-vx_R) + Bx_L*Bz_L - Bx_R*Bz_R) / denom

    def base(rho, vx, vy, vz, p, E, Bx, By, Bz):
        return cp.stack([rho*vx, rho*vx**2 + p + 0.5*(By**2 + Bz**2), rho*vx*vy - Bx*By,
                         rho*vx*vz - Bx*Bz, (E+p)*vx - Bx*(Bx*vx + By*vy + Bz*vz),
                         vx*By - vy*Bx, vx*Bz - vz*Bx], axis=0)

    F_L = base(rho_L, vx_L, vy_L, vz_L, p_L, E_L, Bx_L, By_L, Bz_L)
    F_R = base(rho_R, vx_R, vy_R, vz_R, p_R, E_R, Bx_R, By_R, Bz_R)
    F_Ls = base(rho_starL, S_star, vy_L, vz_L, p_star, E_L, Bx_star, By_star, Bz_star)
    F_Rs = base(rho_starR, S_star, vy_R, vz_R, p_star, E_R, Bx_star, By_star, Bz_star)

    S_AL = S_star - cp.abs(Bx_star)/cp.sqrt(mu0*rho_starL)
    S_AR = S_star + cp.abs(Bx_star)/cp.sqrt(mu0*rho_starR)
    F = cp.where((S_L >= 0)[None,:,:,:], F_L,
            cp.where((S_AL >= 0)[None,:,:,:], F_Ls,
                cp.where((S_star >= 0)[None,:,:,:], F_Ls,
                    cp.where((S_AR >= 0)[None,:,:,:], F_Rs, F_R))))
    return F

# ====================== MAIN LOOP ======================
for step in range(steps):
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    B2 = Bx**2 + By**2 + Bz**2
    p_thermal = (gamma - 1.0) * (E_total - 0.5*rho*(vx**2+vy**2+vz**2) - 0.5*B2 - u_cr)
    fast = cp.sqrt((gamma * cp.maximum(p_thermal, p_floor) + B2) / cp.maximum(rho, rho_floor))
    dt = CFL * dx / cp.max(cp.maximum(cp.sqrt(vx**2+vy**2+vz**2), fast))

    # x-sweep
    rho_L, rho_R = muscl_reconstruct(rho, 0)
    mx_L, mx_R = muscl_reconstruct(mx, 0)
    my_L, my_R = muscl_reconstruct(my, 0)
    mz_L, mz_R = muscl_reconstruct(mz, 0)
    E_L, E_R = muscl_reconstruct(E_total, 0)
    Bx_L, Bx_R = muscl_reconstruct(Bx, 0)
    By_L, By_R = muscl_reconstruct(By, 0)
    Bz_L, Bz_R = muscl_reconstruct(Bz, 0)
    flux = hlld_flux(rho_L, rho_R, mx_L, mx_R, my_L, my_R, mz_L, mz_R, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R)

    # FIXED CONSISTENT INDEXING
    rho[1:-1,:,:] -= (dt/dx) * (flux[0][1:,:,:] - flux[0][:-1,:,:])
    mx[1:-1,:,:] -= (dt/dx) * (flux[1][1:,:,:] - flux[1][:-1,:,:])
    my[1:-1,:,:] -= (dt/dx) * (flux[2][1:,:,:] - flux[2][:-1,:,:])
    mz[1:-1,:,:] -= (dt/dx) * (flux[3][1:,:,:] - flux[3][:-1,:,:])
    E_total[1:-1,:,:] -= (dt/dx) * (flux[4][1:,:,:] - flux[4][:-1,:,:])
    Bx[1:-1,:,:] -= (dt/dx) * (flux[5][1:,:,:] - flux[5][:-1,:,:])
    By[1:-1,:,:] -= (dt/dx) * (flux[6][1:,:,:] - flux[6][:-1,:,:])

    # Dedner
    divB = cp.gradient(Bx, dx, axis=0) + cp.gradient(By, dx, axis=1) + cp.gradient(Bz, dx, axis=2)
    psi = psi - dt * c_h_factor**2 * divB - dt * kappa_factor * psi
    Bx[1:-1,:,:] -= dt * (0.5*(psi[1:,:,:] + psi[:-1,:,:])) / dx
    By[:,1:-1,:] -= dt * (0.5*(psi[:,1:,:] + psi[:,:-1,:])) / dx
    Bz[:,:,1:-1] -= dt * (0.5*(psi[:,:,1:] + psi[:,:,:-1])) / dx

    # Floors
    rho = cp.maximum(rho, rho_floor)
    p_thermal = cp.maximum((gamma-1.0)*(E_total - 0.5*rho*(vx**2+vy**2+vz**2) - 0.5*B2 - u_cr), p_floor)
    E_total = p_thermal/(gamma-1.0) + 0.5*rho*(vx**2+vy**2+vz**2) + 0.5*B2 + u_cr

    if step % 50 == 0:
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.max(cp.sqrt(Bx**2 + By**2 + Bz**2)))
        divB_max = float(cp.max(cp.abs(divB)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB_max = {divB_max:.2e}")

# ====================== FULL DIAGNOSTICS ======================
print("\n=== DETAILED CONSERVATION ANALYSIS ===")
vtot = cp.sqrt(vx**2 + vy**2 + vz**2)
mass_now = float(cp.sum(rho))
E_now = float(cp.sum(E_total))
Lz_now = float(cp.sum(rho * (X*vy - Y*vx)))
px_now = float(cp.sum(mx))
py_now = float(cp.sum(my))
pz_now = float(cp.sum(mz))
print(f"Mass drift          : {100 * (mass_now - mass0) / (mass0 + 1e-12):.6f}%")
print(f"Energy drift        : {100 * (E_now - E0) / (E0 + 1e-12):.6f}%")
print(f"Angular momentum Lz : {100 * (Lz_now - Lz0) / (Lz0 + 1e-12):.6f}%")
print(f"Linear momentum Px  : {100 * (px_now - px0) / (px0 + 1e-12):.6f}%")
print(f"Linear momentum Py  : {100 * (py_now - py0) / (py0 + 1e-12):.6f}%")
print(f"Linear momentum Pz  : {100 * (pz_now - pz0) / (pz0 + 1e-12):.6f}%")

kin = 0.5 * float(cp.sum(rho * vtot**2))
therm = float(cp.sum((E_total - 0.5*rho*vtot**2 - 0.5*(Bx**2+By**2+Bz**2) - u_cr) / (gamma-1)))
mag = 0.5 * float(cp.sum(Bx**2 + By**2 + Bz**2))
cr = float(cp.sum(u_cr))
total_E = kin + therm + mag + cr
print(f"\nEnergy breakdown:")
print(f"  Kinetic   : {kin:.4e} ({100*kin/total_E:.2f}%)")
print(f"  Thermal   : {therm:.4e} ({100*therm/total_E:.2f}%)")
print(f"  Magnetic  : {mag:.4e} ({100*mag/total_E:.2f}%)")
print(f"  CR        : {cr:.4e} ({100*cr/total_E:.2f}%)")

print("\n=== KINETIC ENERGY BALANCE TABLE ===")
print("Region          | Centripetal   | JxB          | Tension      | Pressure     | Turb         | Gravity")
mid = N//2
r_mid = r_cyl[:,:,mid].get().flatten()
v_phi_mid = ((X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / r_cyl[:,:,mid]).get().flatten()
centripetal = (v_phi_mid**2) / r_mid

a_JxB_r = np.zeros_like(r_mid)
a_tension_r = np.zeros_like(r_mid)
a_press_r = np.zeros_like(r_mid)
a_turb_r = np.zeros_like(r_mid)
a_grav_r = (-G * M_dm[:,:,mid].get().flatten() / r_mid**2)

bins = np.linspace(0, L/2, 60)
def bin_avg(x, w):
    hist, _ = np.histogram(x, bins=bins, weights=w)
    count, _ = np.histogram(x, bins=bins)
    return hist / (count + 1e-8)

for name, i1, i2 in [("Inner 0-5 kpc", 0, 10), ("Mid 5-15 kpc", 10, 30), ("Outer 15-30 kpc", 30, 60)]:
    print(f"{name:15} | {np.mean(centripetal[i1:i2]):12.2e} | {np.mean(a_JxB_r[i1:i2]):12.2e} | {np.mean(a_tension_r[i1:i2]):12.2e} | {np.mean(a_press_r[i1:i2]):12.2e} | {np.mean(a_turb_r[i1:i2]):12.2e} | {np.mean(a_grav_r[i1:i2]):12.2e}")

# Tully-Fisher
cum_mass = np.cumsum(np.histogram(r_mid, bins=np.linspace(0, L/2, 60), weights=rho[:,:,mid].get().flatten() * dx**3)[0])
v_rot = v_phi_mid * 100
plt.figure(figsize=(8,5))
plt.plot(cum_mass, v_rot, 'cyan', lw=2.5, label='Sim v_rot')
plt.plot(cum_mass, (cum_mass**0.25)*200, 'red', ls='--', label='TF theory')
plt.xlabel('Cumulative M_b (code units)')
plt.ylabel('Rotation velocity (km/s)')
plt.title('Tully-Fisher Test')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('tully_fisher_v0.5.png')

print("\n✅ FULL CODE FINISHED. Run it and paste the full console output.")
