import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 ALADIN Plasma Cosmology v0.4 — Fixed Staggered Seeding + Safe CT")

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
    x = r / r_s + 1e-12
    return 4 * cp.pi * rho0_nfw * r_s**3 * (cp.log(1 + x) - x / (1 + x))

# ====================== FIELDS (Staggered) ======================
Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)
psi = cp.zeros((N, N, N), dtype=cp.float32)

rho = cp.ones((N, N, N), dtype=cp.float32) * 1e-3
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 1e-4
u_cr = cp.ones((N, N, N), dtype=cp.float32) * 1e-5

# ====================== SAFE STAGGERED SEEDING (Fixed) ======================
B0 = 5.0
Bphi = 2.0

# Use main grid and assign to staggered arrays (safe broadcast)
r_cyl = cp.sqrt(X**2 + Y**2)
Bx[1:,:,:] = -Bphi * (Y[1:,:,:] / (r_cyl[1:,:,:] + 1e-8))   # approximate for Bx faces
By[:,1:,:] =  Bphi * (X[:,1:,:] / (r_cyl[:,1:,:] + 1e-8))
Bz[:,:,1:] = B0 * cp.exp(- (X[:,:,1:]**2 + Y[:,:,1:]**2 + Z[:,:,1:]**2) / 200.0)

# ====================== EQUILIBRIUM INITIALIZATION ======================
r_cyl = cp.sqrt(X**2 + Y**2)

kx = cp.fft.fftfreq(N, d=dx)
ky = cp.fft.fftfreq(N, d=dx)
kz = cp.fft.fftfreq(N, d=dx)
KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
k2 = (2 * cp.pi * KX)**2 + (2 * cp.pi * KY)**2 + (2 * cp.pi * KZ)**2
k2[0,0,0] = 1.0

rho_k = cp.fft.fftn(rho)
phi_k = -4.0 * cp.pi * G * rho_k / k2
phi = cp.real(cp.fft.ifftn(phi_k))

g_r = -cp.gradient(phi, dx, axis=0) * (X / (r_cyl + 1e-8)) - cp.gradient(phi, dx, axis=1) * (Y / (r_cyl + 1e-8))

r3d = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-12)
M_dm_enc = nfw_enclosed_mass(r3d)
g_dm = -G * M_dm_enc / r3d**2
g_r += g_dm * (r_cyl / r3d)

p_thermal_init = cp.ones_like(rho) * p_floor * 10.0
dp_dr = cp.gradient(p_thermal_init, dx, axis=0) * (X / (r_cyl + 1e-8)) + cp.gradient(p_thermal_init, dx, axis=1) * (Y / (r_cyl + 1e-8))
v_phi_eq = v_phi_factor * cp.sqrt(cp.maximum(r_cyl * (cp.abs(g_r) - dp_dr / (rho + 1e-12)), 0.0))

vx = -v_phi_eq * (Y / (r_cyl + 1e-8))
vy =  v_phi_eq * (X / (r_cyl + 1e-8))
vz = cp.zeros_like(vx)

mx = rho * vx
my = rho * vy
mz = rho * vz

# Initial conservation
mass0 = float(cp.sum(rho))
E0 = float(cp.sum(E_total))
Lz0 = float(cp.sum(rho * (X*vy - Y*vx)))
px0 = float(cp.sum(mx))
py0 = float(cp.sum(my))
pz0 = float(cp.sum(mz))

# ====================== MUSCL & HLLD ======================
def minmod(a, b):
    return 0.5 * (cp.sign(a) + cp.sign(b)) * cp.minimum(cp.abs(a), cp.abs(b))

def muscl_reconstruct(U, axis):
    if axis == 0:
        dL = U[1:-1,:,:] - U[:-2,:,:]
        dR = U[2:,:,:]   - U[1:-1,:,:]
        slope = minmod(dL, dR)
        U_L = U[1:-1,:,:] - 0.5 * slope
        U_R = U[1:-1,:,:] + 0.5 * slope
    elif axis == 1:
        dL = U[:,1:-1,:] - U[:,:-2,:]
        dR = U[:,2:,:]   - U[:,1:-1,:]
        slope = minmod(dL, dR)
        U_L = U[:,1:-1,:] - 0.5 * slope
        U_R = U[:,1:-1,:] + 0.5 * slope
    elif axis == 2:
        dL = U[:,:,1:-1] - U[:,:,:-2]
        dR = U[:,:,2:]   - U[:,:,1:-1]
        slope = minmod(dL, dR)
        U_L = U[:,:,1:-1] - 0.5 * slope
        U_R = U[:,:,1:-1] + 0.5 * slope
    return U_L, U_R

def hlld_flux(rho_L, rho_R, mx_L, mx_R, my_L, my_R, mz_L, mz_R, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R):
    B2_L = Bx_L**2 + By_L**2 + Bz_L**2
    B2_R = Bx_R**2 + By_R**2 + Bz_R**2
    p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * ((mx_L**2 + my_L**2 + mz_L**2) / rho_L**2) - B2_L / (2 * mu0))
    p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * ((mx_R**2 + my_R**2 + mz_R**2) / rho_R**2) - B2_R / (2 * mu0))

    vx_L = mx_L / rho_L; vy_L = my_L / rho_L; vz_L = mz_L / rho_L
    vx_R = mx_R / rho_R; vy_R = my_R / rho_R; vz_R = mz_R / rho_R

    c_fL = cp.sqrt((gamma * p_L + B2_L) / rho_L)
    c_fR = cp.sqrt((gamma * p_R + B2_R) / rho_R)
    S_L = cp.minimum(vx_L - c_fL, vx_R - c_fR)
    S_R = cp.maximum(vx_L + c_fL, vx_R + c_fR)
    S_star = (S_L * rho_L * vx_L - S_R * rho_R * vx_R + p_R - p_L) / (rho_L * (S_L - vx_L) - rho_R * (S_R - vx_R) + 1e-12)

    rho_starL = rho_L * (S_L - vx_L) / (S_L - S_star + 1e-12)
    rho_starR = rho_R * (S_R - vx_R) / (S_R - S_star + 1e-12)
    p_star = p_L + rho_L * (S_L - vx_L) * (S_star - vx_L)
    vx_star = S_star
    Bx_star = Bx_L
    denom = rho_L * (S_L - vx_L) + rho_R * (S_R - vx_R)
    By_star = (rho_L * By_L * (S_L - vx_L) + rho_R * By_R * (S_R - vx_R) + Bx_L * By_L - Bx_R * By_R) / denom
    Bz_star = (rho_L * Bz_L * (S_L - vx_L) + rho_R * Bz_R * (S_R - vx_R) + Bx_L * Bz_L - Bx_R * Bz_R) / denom

    def base_flux(rho, vx, vy, vz, p, E, Bx, By, Bz):
        return cp.stack([rho * vx,
                         rho * vx**2 + p + 0.5 * (By**2 + Bz**2),
                         rho * vx * vy - Bx * By,
                         rho * vx * vz - Bx * Bz,
                         (E + p) * vx - Bx * (Bx * vx + By * vy + Bz * vz),
                         vx * By - vy * Bx,
                         vx * Bz - vz * Bx], axis=0)

    F_L = base_flux(rho_L, vx_L, vy_L, vz_L, p_L, E_L, Bx_L, By_L, Bz_L)
    F_R = base_flux(rho_R, vx_R, vy_R, vz_R, p_R, E_R, Bx_R, By_R, Bz_R)
    F_L_star = base_flux(rho_starL, vx_star, vy_L, vz_L, p_star, E_L, Bx_star, By_star, Bz_star)
    F_R_star = base_flux(rho_starR, vx_star, vy_R, vz_R, p_star, E_R, Bx_star, By_star, Bz_star)

    S_AL = S_star - cp.abs(Bx_star) / cp.sqrt(mu0 * rho_starL)
    S_AR = S_star + cp.abs(Bx_star) / cp.sqrt(mu0 * rho_starR)

    sign_Bx = cp.sign(Bx_star)
    v_y_LL = vy_L - sign_Bx * By_star / cp.sqrt(mu0 * rho_starL)
    v_z_LL = vz_L - sign_Bx * Bz_star / cp.sqrt(mu0 * rho_starL)
    F_LL = base_flux(rho_starL, S_AL, v_y_LL, v_z_LL, p_star, E_L, Bx_star, By_star, Bz_star)

    v_y_RR = vy_R + sign_Bx * By_star / cp.sqrt(mu0 * rho_starR)
    v_z_RR = vz_R + sign_Bx * Bz_star / cp.sqrt(mu0 * rho_starR)
    F_RR = base_flux(rho_starR, S_AR, v_y_RR, v_z_RR, p_star, E_R, Bx_star, By_star, Bz_star)

    F = cp.where((S_L >= 0)[None, :, :, :], F_L,
                 cp.where((S_AL >= 0)[None, :, :, :], F_L_star,
                           cp.where((S_star >= 0)[None, :, :, :], F_LL,
                                     cp.where((S_AR >= 0)[None, :, :, :], F_RR,
                                               cp.where((S_R >= 0)[None, :, :, :], F_R_star, F_R)))))
    return F

def compute_divB():
    Bx_c = 0.5 * (Bx[:-1,:,:] + Bx[1:,:,:])
    By_c = 0.5 * (By[:,:-1,:] + By[:,1:,:])
    Bz_c = 0.5 * (Bz[:,:,:-1] + Bz[:,:,1:])
    divB = (cp.gradient(Bx_c, dx, axis=0) + cp.gradient(By_c, dx, axis=1) + cp.gradient(Bz_c, dx, axis=2))
    return divB

# ====================== MAIN LOOP ======================
for step in range(steps):
    rho_old = rho.copy()
    mx_old = mx.copy()
    my_old = my.copy()
    mz_old = mz.copy()
    E_old = E_total.copy()
    Bx_old = Bx.copy()
    By_old = By.copy()
    Bz_old = Bz.copy()

    for stage in range(2):
        vx = mx / rho
        vy = my / rho
        vz = mz / rho
        Bx_c = 0.5 * (Bx[:-1,:,:] + Bx[1:,:,:])
        By_c = 0.5 * (By[:,:-1,:] + By[:,1:,:])
        Bz_c = 0.5 * (Bz[:,:,:-1] + Bz[:,:,1:])
        B2_c = Bx_c**2 + By_c**2 + Bz_c**2

        fast_speed = cp.sqrt((gamma * ((E_total - 0.5*rho*(vx**2+vy**2+vz**2) - 0.5*B2_c - u_cr)/(gamma-1.0)) + B2_c) / rho)
        c_h = c_h_factor * cp.max(fast_speed)
        kappa = kappa_factor * c_h / dx
        dt = CFL * dx / cp.maximum(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)), cp.max(fast_speed))

        # x-sweep
        rho_L, rho_R = muscl_reconstruct(rho, 0)
        mx_L, mx_R = muscl_reconstruct(mx, 0)
        my_L, my_R = muscl_reconstruct(my, 0)
        mz_L, mz_R = muscl_reconstruct(mz, 0)
        E_L, E_R = muscl_reconstruct(E_total, 0)
        Bx_L, Bx_R = muscl_reconstruct(Bx_c, 0)
        By_L, By_R = muscl_reconstruct(By_c, 0)
        Bz_L, Bz_R = muscl_reconstruct(Bz_c, 0)
        flux_x = hlld_flux(rho_L, rho_R, mx_L, mx_R, my_L, my_R, mz_L, mz_R, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R)

        rho[1:-1,:,:] -= (dt/dx)*(flux_x[0][1:,:,:] - flux_x[0][:-1,:,:])
        mx[1:-1,:,:] -= (dt/dx)*(flux_x[1][1:,:,:] - flux_x[1][:-1,:,:])
        my[1:-1,:,:] -= (dt/dx)*(flux_x[2][1:,:,:] - flux_x[2][:-1,:,:])
        mz[1:-1,:,:] -= (dt/dx)*(flux_x[3][1:,:,:] - flux_x[3][:-1,:,:])
        E_total[1:-1,:,:] -= (dt/dx)*(flux_x[4][1:,:,:] - flux_x[4][:-1,:,:])

        # y-sweep
        rho_L, rho_R = muscl_reconstruct(rho, 1)
        mx_L, mx_R = muscl_reconstruct(mx, 1)
        my_L, my_R = muscl_reconstruct(my, 1)
        mz_L, mz_R = muscl_reconstruct(mz, 1)
        E_L, E_R = muscl_reconstruct(E_total, 1)
        Bx_L, Bx_R = muscl_reconstruct(Bx_c, 1)
        By_L, By_R = muscl_reconstruct(By_c, 1)
        Bz_L, Bz_R = muscl_reconstruct(Bz_c, 1)
        flux_y = hlld_flux(rho_L, rho_R, mx_L, mx_R, my_L, my_R, mz_L, mz_R, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R)

        rho[:,1:-1,:] -= (dt/dx)*(flux_y[0][:,1:,:] - flux_y[0][:,:-1,:])
        mx[:,1:-1,:] -= (dt/dx)*(flux_y[1][:,1:,:] - flux_y[1][:,:-1,:])
        my[:,1:-1,:] -= (dt/dx)*(flux_y[2][:,1:,:] - flux_y[2][:,:-1,:])
        mz[:,1:-1,:] -= (dt/dx)*(flux_y[3][:,1:,:] - flux_y[3][:,:-1,:])
        E_total[:,1:-1,:] -= (dt/dx)*(flux_y[4][:,1:,:] - flux_y[4][:,:-1,:])

        # z-sweep
        rho_L, rho_R = muscl_reconstruct(rho, 2)
        mx_L, mx_R = muscl_reconstruct(mx, 2)
        my_L, my_R = muscl_reconstruct(my, 2)
        mz_L, mz_R = muscl_reconstruct(mz, 2)
        E_L, E_R = muscl_reconstruct(E_total, 2)
        Bx_L, Bx_R = muscl_reconstruct(Bx_c, 2)
        By_L, By_R = muscl_reconstruct(By_c, 2)
        Bz_L, Bz_R = muscl_reconstruct(Bz_c, 2)
        flux_z = hlld_flux(rho_L, rho_R, mx_L, mx_R, my_L, my_R, mz_L, mz_R, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R)

        rho[:,:,1:-1] -= (dt/dx)*(flux_z[0][:,:,1:] - flux_z[0][:,:,:-1])
        mx[:,:,1:-1] -= (dt/dx)*(flux_z[1][:,:,1:] - flux_z[1][:,:,:-1])
        my[:,:,1:-1] -= (dt/dx)*(flux_z[2][:,:,1:] - flux_z[2][:,:,:-1])
        mz[:,:,1:-1] -= (dt/dx)*(flux_z[3][:,:,1:] - flux_z[3][:,:,:-1])
        E_total[:,:,1:-1] -= (dt/dx)*(flux_z[4][:,:,1:] - flux_z[4][:,:,:-1])

    # SSP-RK2 averaging
    rho = 0.5 * (rho_old + rho)
    mx = 0.5 * (mx_old + mx)
    my = 0.5 * (my_old + my)
    mz = 0.5 * (mz_old + mz)
    E_total = 0.5 * (E_old + E_total)
    Bx = 0.5 * (Bx_old + Bx)
    By = 0.5 * (By_old + By)
    Bz = 0.5 * (Bz_old + Bz)

    # Safe CT + Dedner
    divB = compute_divB()
    psi = psi - dt * c_h**2 * divB - dt * kappa * psi
    psi_x = 0.5 * (psi[1:,:,:] + psi[:-1,:,:])
    psi_y = 0.5 * (psi[:,1:,:] + psi[:,:-1,:])
    psi_z = 0.5 * (psi[:,:,1:] + psi[:,:,:-1])
    Bx[1:-1,:,:] -= dt * (psi_x[1:,:,:] - psi_x[:-1,:,:]) / dx
    By[:,1:-1,:] -= dt * (psi_y[:,1:,:] - psi_y[:,:-1,:]) / dx
    Bz[:,:,1:-1] -= dt * (psi_z[:,:,1:] - psi_z[:,:,:-1]) / dx

    # Floors
    rho = cp.maximum(rho, rho_floor)
    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    Bx_c = 0.5 * (Bx[:-1,:,:] + Bx[1:,:,:])
    By_c = 0.5 * (By[:,:-1,:] + By[:,1:,:])
    Bz_c = 0.5 * (Bz[:,:,:-1] + Bz[:,:,1:])
    B2_c = Bx_c**2 + By_c**2 + Bz_c**2
    p_thermal = cp.maximum((gamma - 1.0) * (E_total - 0.5 * rho * (vx**2 + vy**2 + vz**2) - 0.5 * B2_c - u_cr), p_floor)
    E_total = p_thermal / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2) + 0.5 * B2_c + u_cr

    if step % 50 == 0:
        divB_max = float(cp.max(cp.abs(compute_divB())))
        vmax = float(cp.max(cp.sqrt(vx**2 + vy**2 + vz**2)))
        Bmax = float(cp.max(cp.sqrt(Bx_c**2 + By_c**2 + Bz_c**2)))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | divB_max = {divB_max:.2e}")

# ====================== FULL DIAGNOSTICS ======================
vtot = cp.sqrt(vx**2 + vy**2 + vz**2)
Bx_c = 0.5 * (Bx[:-1,:,:] + Bx[1:,:,:])
By_c = 0.5 * (By[:,:-1,:] + By[:,1:,:])
Bz_c = 0.5 * (Bz[:,:,:-1] + Bz[:,:,1:])
B2 = Bx_c**2 + By_c**2 + Bz_c**2
p_thermal = (gamma - 1.0) * (E_total - 0.5 * rho * vtot**2 - 0.5 * B2 - u_cr)

print("\n=== DETAILED CONSERVATION ANALYSIS ===")
mass_now = float(cp.sum(rho))
E_now = float(cp.sum(E_total))
Lz_now = float(cp.sum(rho * (X*vy - Y*vx)))
px_now = float(cp.sum(mx))
py_now = float(cp.sum(my))
pz_now = float(cp.sum(mz))
mass_drift = 100 * (mass_now - mass0) / (mass0 + 1e-12)
E_drift = 100 * (E_now - E0) / (E0 + 1e-12)
Lz_drift = 100 * (Lz_now - Lz0) / (Lz0 + 1e-12)
px_drift = 100 * (px_now - px0) / (px0 + 1e-12) if px0 != 0 else 0
py_drift = 100 * (py_now - py0) / (py0 + 1e-12) if py0 != 0 else 0
pz_drift = 100 * (pz_now - pz0) / (pz0 + 1e-12) if pz0 != 0 else 0
print(f"Mass drift          : {mass_drift:.6f}%")
print(f"Energy drift        : {E_drift:.6f}%")
print(f"Angular momentum Lz : {Lz_drift:.6f}%")
print(f"Linear momentum Px  : {px_drift:.6f}%")
print(f"Linear momentum Py  : {py_drift:.6f}%")
print(f"Linear momentum Pz  : {pz_drift:.6f}%")
    
# Energy breakdown
kin = 0.5 * float(cp.sum(rho * vtot**2))
therm = float(cp.sum(p_thermal / (gamma - 1)))
mag = 0.5 * float(cp.sum(B2))
cr = float(cp.sum(u_cr))
total_E = kin + therm + mag + cr
print(f"\nEnergy breakdown:")
print(f"  Kinetic   : {kin:.4e} ({100*kin/total_E:.2f}%)")
print(f"  Thermal   : {therm:.4e} ({100*therm/total_E:.2f}%)")
print(f"  Magnetic  : {mag:.4e} ({100*mag/total_E:.2f}%)")
print(f"  CR        : {cr:.4e} ({100*cr/total_E:.2f}%)")
print(f"  Total     : {total_E:.4e}")

# GS95 spectrum
mid = N//2
b = cp.stack([Bx_c[:,:,mid], By_c[:,:,mid], Bz_c[:,:,mid]], axis=-1)
b_norm = cp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8
b_hat = b / b_norm
kx = 2*cp.pi*cp.fft.fftfreq(N, d=dx)
KX, KY = cp.meshgrid(kx, kx, indexing='ij')
K = cp.stack([KX, KY, cp.zeros_like(KX)], axis=-1)
k_par = cp.abs(cp.sum(K * b_hat, axis=-1))
k_perp = cp.sqrt(cp.sum(K**2, axis=-1) - k_par**2 + 1e-12)
T_rphi = - (Bx_c[:,:,mid] * By_c[:,:,mid]) / mu0
T_fft = cp.abs(cp.fft.fft2(T_rphi))**2
low_k_mask = k_perp < 0.5
high_k_mask = k_perp > 2.0
E_perp = cp.sum(T_fft[low_k_mask]) / cp.sum(T_fft)
E_par = cp.sum(T_fft[high_k_mask]) / cp.sum(T_fft)
anisotropy_ratio = E_perp / (E_par + 1e-8)
print(f"\n=== GS95 SPECTRUM ===")
print(f"Perp power fraction (low-k) = {float(E_perp.get()):.3f}")
print(f"Par power fraction (high-k) = {float(E_par.get()):.3f}")
print(f"Anisotropy ratio E_⊥/E_∥ = {float(anisotropy_ratio.get()):.2f}")

# Kinetic energy balance table + Tully-Fisher (full block)
print("\n=== KINETIC ENERGY BALANCE TABLE ===")
print("Region          | Needed (v²/r) | JxB          | Tension      | Pressure     | Aniso Turb   | Gravity      | Escape Eff.")
mid = N//2
r_mid = r_cyl[:,:,mid].flatten().get()
v_phi_mid = ((X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / r_cyl[:,:,mid]).flatten().get()
centripetal = (v_phi_mid**2) / r_mid

r_hat_x = X / r_cyl
r_hat_y = Y / r_cyl
Jx = (cp.gradient(Bz_c, dx, axis=1) - cp.gradient(By_c, dx, axis=2)) / 1.0
Jy = (cp.gradient(Bx_c, dx, axis=2) - cp.gradient(Bz_c, dx, axis=0)) / 1.0
Jz_total = (cp.gradient(By_c, dx, axis=0) - cp.gradient(Bx_c, dx, axis=1)) / 1.0
JxB_x = Jy * Bz_c - Jz_total * By_c
JxB_y = Jz_total * Bx_c - Jx * Bz_c
a_JxB_r = ((JxB_x * r_hat_x + JxB_y * r_hat_y) / (rho + 1e-6))[:,:,mid].flatten().get()
tension_r = (Bx_c * cp.gradient(Bx_c, dx, axis=0) + By_c * cp.gradient(By_c, dx, axis=1) + Bz_c * cp.gradient(Bz_c, dx, axis=2)) / (rho + 1e-6)
a_tension_r = tension_r[:,:,mid].flatten().get()
P_total = p_thermal + rho * (0.3 * cp.ones_like(rho))**2
a_press_r = - ((cp.gradient(P_total, dx, axis=0) * r_hat_x + cp.gradient(P_total, dx, axis=1) * r_hat_y)[:,:,mid] / (rho[:,:,mid] + 1e-6)).flatten().get()
a_turb_r = - ((cp.gradient(P_total, dx, axis=0) * r_hat_x + cp.gradient(P_total, dx, axis=1) * r_hat_y)[:,:,mid] / (rho[:,:,mid] + 1e-6)).flatten().get()
a_grav_r = ((cp.gradient(-G*M_dm_enc, dx, axis=0) * X + cp.gradient(-G*M_dm_enc, dx, axis=1) * Y) / r_cyl[:,:,mid]).flatten().get()

bins = np.linspace(0, L/2, 60)
def bin_avg(x, weights):
    hist, _ = np.histogram(x, bins=bins, weights=weights)
    count, _ = np.histogram(x, bins=bins)
    return hist / (count + 1e-8)

cent_bin = bin_avg(r_mid, centripetal)
jxb_bin = bin_avg(r_mid, a_JxB_r)
tension_bin = bin_avg(r_mid, a_tension_r)
press_bin = bin_avg(r_mid, a_press_r)
turb_bin = bin_avg(r_mid, a_turb_r)
grav_bin = bin_avg(r_mid, a_grav_r)

for name, i1, i2 in [("Inner 0-5 kpc", 0, 10), ("Mid 5-15 kpc", 10, 30), ("Outer 15-30 kpc", 30, 60)]:
    print(f"{name:15} | {np.mean(cent_bin[i1:i2]):12.2e} | {np.mean(jxb_bin[i1:i2]):12.2e} | {np.mean(tension_bin[i1:i2]):12.2e} | {np.mean(press_bin[i1:i2]):12.2e} | {np.mean(turb_bin[i1:i2]):12.2e} | {np.mean(grav_bin[i1:i2]):12.2e} | {0.1 * (float(cp.sum(rho)) * dx**3) * (float(cp.mean(vtot)) / 2.3):8.2e}")

# Tully-Fisher
cum_mass = np.cumsum(np.histogram(r_mid, bins=np.linspace(0, L/2, 60), weights=rho[:,:,mid].flatten().get() * dx**3)[0])
v_rot = v_phi_mid * 100
plt.figure(figsize=(8,5))
plt.plot(cum_mass, v_rot, 'cyan', lw=2.5, label='Sim v_rot')
plt.plot(cum_mass, (cum_mass**0.25)*200, 'red', ls='--', label='TF theory v∝M_b^{1/4}')
plt.xlabel('Cumulative M_b (code units)')
plt.ylabel('Rotation velocity (km/s)')
plt.title('Tully-Fisher Test')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('tully_fisher_v0.4.png')

print("\n✅ v0.4 Full Complete Code Ready!")
print("Run this version and paste the full console output.")
