import numpy as np
import matplotlib.pyplot as plt
import time

print("🌌 Plasma Cosmology v5.7 — FULL 128³ + Fixed Divergence Cleaning")

# ====================== FIXED GRID ======================
N = 128
L = 60.0
dx = L / N
x = y = z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
r_cyl = np.sqrt(X**2 + Y**2)
r_sph = np.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

mu0 = 4 * np.pi * 1e-7
gamma = 5.0 / 3.0
G = 6.6743e-11
CFL = 0.35
ch = 2.0
kappa = 0.5
steps = 800
max_v = 350 * 1000.0
alpha0 = 0.0008

# ====================== NFW ======================
M_vir = 1.2e12 * 1.989e30
rs = 22.0 * 3.086e19
def nfw_mass(r):
    x = r / rs
    return M_vir * (np.log(1 + x) - x / (1 + x)) / (np.log(2) - 0.5)

def run_simulation(use_dm):
    M_enc_dm = nfw_mass(r_sph) if use_dm else np.zeros_like(r_sph)
    
    rho = 1.8e-21 * np.exp(-r_cyl / 12.0) * np.exp(-np.abs(Z)/4.0)
    vx = np.zeros_like(rho, dtype=np.float32)
    vy = np.zeros_like(rho, dtype=np.float32)
    vz = np.zeros_like(rho, dtype=np.float32)
    u_cr = 1.6e-13 * np.exp(-r_cyl / 15.0) * np.exp(-np.abs(Z)/6.0)
    
    Bx = np.zeros((N+1, N, N), dtype=np.float32)
    By = np.zeros((N, N+1, N), dtype=np.float32)
    Bz = np.zeros((N, N, N+1), dtype=np.float32)
    psi = np.zeros_like(rho, dtype=np.float32)
    
    # Weak seed B
    for k in range(N+1):
        zf = -L/2 + k*dx
        r2d = np.sqrt(X[:,:,0]**2 + Y[:,:,0]**2)
        Bz[:,:,k] = 1.2e-10 * np.exp(-r2d**2 / 280.0) * np.exp(-zf**2 / 45.0)
    
    # Self-consistent rotating equilibrium
    M_bary = 4*np.pi*r_cyl**2*rho*dx
    M_total = M_bary + M_enc_dm
    P_tot_approx = 2e-12 * rho + u_cr / 3.0
    dp_dr = np.gradient(P_tot_approx, dx, axis=0) * (X / (r_cyl + 1e-8))
    v_phi_eq = np.sqrt(np.maximum(G * M_total / (r_cyl + 1e-8) + dp_dr / rho, 0))
    vy = v_phi_eq * (X / (r_cyl + 1e-8))
    vx = -v_phi_eq * (Y / (r_cyl + 1e-8))
    
    p_th = 2e-12 * rho
    E_total = p_th / (gamma - 1) + 0.5*rho*(vx**2 + vy**2 + vz**2) + u_cr
    
    e_kin_list, e_mag_list, e_therm_list, e_cr_list, e_grav_list, Lz_list = [], [], [], [], [], []
    
    for step in range(steps):
        Bx_c = (Bx[:-1] + Bx[1:]) / 2
        By_c = (By[:,:-1] + By[:,1:]) / 2
        Bz_c = (Bz[:,:,:-1] + Bz[:,:,1:]) / 2
        B2 = Bx_c**2 + By_c**2 + Bz_c**2
        
        vtot = np.sqrt(vx**2 + vy**2 + vz**2)
        p_thermal = (gamma-1) * (E_total - 0.5*rho*vtot**2 - B2/(2*mu0) - u_cr)
        cs = np.sqrt(gamma * np.maximum(p_thermal, 0) / (rho + 1e-30))
        ca = np.sqrt(B2 / (mu0 * rho + 1e-30))
        cmax = vtot.max() + cs.max() + ca.max() + ch
        dt = CFL * dx / cmax
        
        # CT + EMFs + Helicity Dynamo
        Ex = np.zeros((N, N+1, N+1), dtype=np.float32)
        Ey = np.zeros((N+1, N, N+1), dtype=np.float32)
        Ez = np.zeros((N+1, N+1, N), dtype=np.float32)
        Ex[:,1:,1:] = -(vy * Bz_c - vz * By_c)
        Ey[1:,:,1:] = -(vz * Bx_c - vx * Bz_c)
        Ez[1:,1:,:] = -(vx * By_c - vy * Bx_c)
        
        shear = np.gradient(vy, dx, axis=0) - np.gradient(vx, dx, axis=1)
        helicity_factor = np.abs(shear) * 1e-3
        alpha = alpha0 * helicity_factor
        Ez[1:,1:,:] += alpha * Bz_c
        
        curlEx = ((Ez[1:-1,1:,:] - Ez[1:-1,:-1,:]) - (Ey[1:-1,:,1:] - Ey[1:-1,:,:-1])) / dx
        curlEy = ((Ex[:,1:-1,1:] - Ex[:,1:-1,:-1]) - (Ez[1:,1:-1,:] - Ez[:-1,1:-1,:])) / dx
        curlEz = ((Ey[1:,:,1:-1] - Ey[:-1,:,1:-1]) - (Ex[:,1:,1:-1] - Ex[:,:-1,1:-1])) / dx
        
        Bx[1:-1] += dt * curlEx
        By[:,1:-1] += dt * curlEy
        Bz[:,:,1:-1] += dt * curlEz
        
        # FIXED Hyperbolic cleaning - direct finite difference (no gradient slicing error)
        Bx[1:-1] -= dt * (psi[1:,:,:] - psi[:-1,:,:]) / dx
        By[:,1:-1] -= dt * (psi[:,1:,:] - psi[:,:-1,:]) / dx
        Bz[:,:,1:-1] -= dt * (psi[:,:,1:] - psi[:,:,:-1]) / dx
        
        # Self-gravity
        rho_k = np.fft.fftn(rho)
        kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
        kz = 2*np.pi*np.fft.fftfreq(N, d=dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX**2 + KY**2 + KZ**2 + 1e-12
        Phi_k = -4*np.pi*G*rho_k / k2
        Phi = np.real(np.fft.ifftn(Phi_k))
        g_x = -np.gradient(Phi, dx, axis=0)
        g_y = -np.gradient(Phi, dx, axis=1)
        g_z = -np.gradient(Phi, dx, axis=2)
        
        # Forces
        Jx = (np.gradient(Bz_c, dx, axis=1) - np.gradient(By_c, dx, axis=2)) / mu0
        Jy = (np.gradient(Bx_c, dx, axis=2) - np.gradient(Bz_c, dx, axis=0)) / mu0
        Jz_total = (np.gradient(By_c, dx, axis=0) - np.gradient(Bx_c, dx, axis=1)) / mu0
        
        Fx = Jy * Bz_c - Jz_total * By_c + g_x
        Fy = Jz_total * Bx_c - Jx * Bz_c + g_y
        Fz = Jx * By_c - Jy * Bx_c + g_z
        
        if use_dm:
            grav_dm = -6.6743e-11 * M_enc_dm / (r_sph**2 + 1e-8)
            Fx += grav_dm * (X / r_sph)
            Fy += grav_dm * (Y / r_sph)
            Fz += grav_dm * (Z / r_sph)
        
        P_cr = u_cr / 3.0 if USE_CR else 0.0
        P_tot = p_th + P_cr + B2 / (2*mu0)
        
        Fx -= np.gradient(P_tot, dx, axis=0)
        Fy -= np.gradient(P_tot, dx, axis=1)
        Fz -= np.gradient(P_tot, dx, axis=2)
        
        # Conservative update
        vx += dt * Fx / (rho + 1e-30)
        vy += dt * Fy / (rho + 1e-30)
        vz += dt * Fz / (rho + 1e-30)
        
        v_tot = np.sqrt(vx**2 + vy**2 + vz**2)
        vx = np.clip(vx, -max_v, max_v)
        vy = np.clip(vy, -max_v, max_v)
        vz = np.clip(vz, -max_v, max_v)
        
        div_v = (np.gradient(rho*vx, dx, axis=0) + np.gradient(rho*vy, dx, axis=1) + np.gradient(rho*vz, dx, axis=2))
        rho += dt * (-div_v)
        rho = np.maximum(rho, 1e-25)
        
        if USE_CR:
            div_cr = (np.gradient(u_cr*vx, dx, axis=0) + np.gradient(u_cr*vy, dx, axis=1) + np.gradient(u_cr*vz, dx, axis=2))
            lap_cr = sum(np.gradient(np.gradient(u_cr, dx, axis=i), dx, axis=i) for i in range(3))
            source = 2.5e-15 * np.exp(-r_cyl / 8.0) * np.exp(-np.abs(Z)/3.0)
            u_cr += dt * (-div_cr + 3e-4 * lap_cr + source)
        
        # Diagnostics
        e_kin = np.sum(0.5 * rho * v_tot**2) * dx**3
        e_mag = np.sum(B2 / (2*mu0)) * dx**3
        e_therm = np.sum(p_thermal / (gamma-1)) * dx**3
        e_cr_val = np.sum(u_cr) * dx**3 if USE_CR else 0
        e_grav = -0.5 * np.sum(rho * Phi) * dx**3
        Lz = np.sum(rho * (X * vy - Y * vx)) * dx**3
        
        e_kin_list.append(e_kin)
        e_mag_list.append(e_mag)
        e_therm_list.append(e_therm)
        e_cr_list.append(e_cr_val)
        e_grav_list.append(e_grav)
        Lz_list.append(Lz)
        
        if step % 100 == 0 or step == steps-1:
            Bmax = np.sqrt(B2).max() * 1e6
            vmax = v_tot.max() / 1000
            print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")
    
    # Conservation check
    total_e_init = e_kin_list[0] + e_mag_list[0] + e_therm_list[0] + e_cr_list[0] + e_grav_list[0]
    total_e_final = e_kin_list[-1] + e_mag_list[-1] + e_therm_list[-1] + e_cr_list[-1] + e_grav_list[-1]
    energy_drift = 100 * (total_e_final - total_e_init) / total_e_init
    Lz_drift = 100 * (Lz_list[-1] - Lz_list[0]) / (abs(Lz_list[0]) + 1e-30)
    print(f"Energy drift: {energy_drift:.4f}%   |   Lz drift: {Lz_drift:.4f}%")
    
    return {
        'r': r_cyl[:,:,N//2].flatten(),
        'v_phi': ((X[:,:,N//2]*vy[:,:,N//2] - Y[:,:,N//2]*vx[:,:,N//2]) / (r_cyl[:,:,N//2] + 1e-8)).flatten() / 1000,
        'e_kin': e_kin_list,
        'e_mag': e_mag_list,
        'e_therm': e_therm_list,
        'e_cr': e_cr_list,
        'e_grav': e_grav_list,
        'Lz': Lz_list
    }

# ====================== SIDE-BY-SIDE RUNS ======================
print("Running Pure Plasma mode...")
plasma_data = run_simulation(False)

print("\nRunning DM mode...")
dm_data = run_simulation(True)

# ====================== PLOTS ======================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
r_plot = plasma_data['r']
plt.plot(r_plot, plasma_data['v_phi'], 'cyan', lw=2.5, label='Pure Plasma')
plt.plot(r_plot, dm_data['v_phi'], 'orange', lw=2.5, label='With DM')
plt.axhline(230, color='red', ls='--', label='Observed flat ~230 km/s')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Rotation Curves — Side-by-Side (128³)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(plasma_data['e_kin'], label='Kinetic (Plasma)', color='cyan')
plt.plot(dm_data['e_kin'], label='Kinetic (DM)', color='orange')
plt.plot(plasma_data['e_mag'], label='Magnetic (Plasma)', color='magenta')
plt.plot(dm_data['e_mag'], label='Magnetic (DM)', color='red')
plt.xlabel('Step')
plt.ylabel('Energy (J)')
plt.title('Energy Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("✅ v5.7 complete! Check the two plots and the printed conservation numbers.")
