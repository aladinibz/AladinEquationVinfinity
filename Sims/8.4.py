import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🌌 Plasma Cosmology v8.4 — 128³ A100 GPU + Plasma Stress-Energy Gravity")

N = 128
L = 60.0
dx = L / N
x = y = z = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
r_cyl = cp.sqrt(X**2 + Y**2)
r_sph = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

mu0 = 4 * cp.pi * 1e-7
gamma = 5.0 / 3.0
G = cp.float64(6.6743e-11)
c = cp.float64(3e8)          # speed of light for T00/c²
kB = 1.380649e-23
mH = 1.67e-27
CFL = 0.12
steps = 800
max_v = 300000.0
alpha0 = 0.0010
nu_visc = 0.08
kappa_cr = cp.float32(1.0e-3)
B_eq = cp.float32(5e-6)

M_vir = cp.float64(1.2e12 * 1.989e30)
rs = cp.float64(22.0 * 3.086e19)

def nfw_mass(r):
    x = r / rs
    return M_vir * (cp.log(1 + x) - x / (1 + x)) / (cp.log(2) - 0.5)

def run_simulation(use_dm, use_cr=True):
    M_enc_dm = nfw_mass(r_sph) if use_dm else cp.zeros_like(r_sph, dtype=cp.float64)
    
    rho = 1.8e-21 * cp.exp(-r_cyl / 12.0) * cp.exp(-cp.abs(Z)/4.0)
    vx = cp.zeros_like(rho, dtype=cp.float32)
    vy = cp.zeros_like(rho, dtype=cp.float32)
    vz = cp.zeros_like(rho, dtype=cp.float32)
    u_cr = 1.6e-13 * cp.exp(-r_cyl / 15.0) * cp.exp(-cp.abs(Z)/6.0) if use_cr else cp.zeros_like(rho)
    
    Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
    By = cp.zeros((N, N+1, N), dtype=cp.float32)
    Bz = cp.zeros((N, N, N+1), dtype=cp.float32)
    psi = cp.zeros_like(rho, dtype=cp.float32)
    
    # Vertical seed
    for k in range(N+1):
        zf = -L/2 + k*dx
        r2d = cp.sqrt(X[:,:,0]**2 + Y[:,:,0]**2)
        Bz[:,:,k] = 5e-8 * cp.exp(-r2d**2 / 250.0) * cp.exp(-zf**2 / 40.0)
    
    # Toroidal seed (safe indexing)
    r2d = cp.sqrt(X[:,:,0]**2 + Y[:,:,0]**2)
    Bphi = 2e-7 * cp.exp(-r2d / 15.0)
    for k in range(N):
        r_k = r_cyl[:,:,k] + 1e-8
        slice_y = Y[:,:,k] / r_k
        slice_x = X[:,:,k] / r_k
        Bx[:-1,:,k] -= Bphi * slice_y
        By[:,:-1,k] += Bphi * slice_x
    
    # Small perturbation
    pert = 0.02 * cp.random.normal(0, 1, rho.shape, dtype=cp.float32)
    rho *= (1 + pert)
    vy *= (1 + 0.015 * pert)
    
    # Equilibrium rotation
    M_bary = 4*cp.pi*r_cyl**2*rho*dx
    M_total = M_bary.astype(cp.float64) + M_enc_dm
    P_tot_approx = 2e-12 * rho + (u_cr / 3.0 if use_cr else 0.0)
    dp_dr = cp.gradient(P_tot_approx, dx, axis=0) * (X / (r_cyl + 1e-8))
    v_phi_eq = cp.sqrt(cp.maximum(G * M_total / (r_cyl + 1e-8) + dp_dr / rho, 0))
    v_phi_eq = cp.clip(v_phi_eq, 0, 280000)
    vy = v_phi_eq * (X / (r_cyl + 1e-8))
    vx = -v_phi_eq * (Y / (r_cyl + 1e-8))
    
    p_th = 2e-12 * rho
    E_total = p_th / (gamma - 1) + 0.5*rho*(vx**2 + vy**2 + vz**2) + u_cr
    
    for step in range(steps):
        Bx_c = (Bx[:-1] + Bx[1:]) / 2
        By_c = (By[:,:-1] + By[:,1:]) / 2
        Bz_c = (Bz[:,:,:-1] + Bz[:,:,1:]) / 2
        B2 = Bx_c**2 + By_c**2 + Bz_c**2
        
        vtot = cp.sqrt(vx**2 + vy**2 + vz**2)
        p_thermal = (gamma-1) * (E_total - 0.5*rho*vtot**2 - B2/(2*mu0) - u_cr)
        p_thermal = cp.maximum(p_thermal, 1e-13)
        
        # Plasma stress-energy contribution
        T00_kin = 0.5 * rho * vtot**2
        T00_mag = B2 / (2*mu0)
        T00_therm = p_thermal / (gamma - 1)
        T00_cr = u_cr
        T00_total = T00_kin + T00_mag + T00_therm + T00_cr
        rho_eff = rho + T00_total / (c**2)   # <--- your gravity_plasma.py idea
        
        cs = cp.sqrt(gamma * p_thermal / (rho + 1e-30))
        ca = cp.sqrt(B2 / (mu0 * rho + 1e-30))
        cmax = vtot.max() + cs.max() + ca.max() + 2.0
        dt = CFL * dx / cmax
        
        # CT + quenched dynamo
        Ex = cp.zeros((N, N+1, N+1), dtype=cp.float32)
        Ey = cp.zeros((N+1, N, N+1), dtype=cp.float32)
        Ez = cp.zeros((N+1, N+1, N), dtype=cp.float32)
        Ex[:,1:,1:] = -(vy * Bz_c - vz * By_c)
        Ey[1:,:,1:] = -(vz * Bx_c - vx * Bz_c)
        Ez[1:,1:,:] = -(vx * By_c - vy * Bx_c)
        
        shear = cp.gradient(vy, dx, axis=0) - cp.gradient(vx, dx, axis=1)
        alpha = alpha0 * cp.abs(shear) * 1e-3
        alpha = alpha / (1 + B2 / B_eq**2)
        Ez[1:,1:,:] += alpha * Bz_c
        
        curlEx = ((Ez[1:-1,1:,:] - Ez[1:-1,:-1,:]) - (Ey[1:-1,:,1:] - Ey[1:-1,:,:-1])) / dx
        curlEy = ((Ex[:,1:-1,1:] - Ex[:,1:-1,:-1]) - (Ez[1:,1:-1,:] - Ez[:-1,1:-1,:])) / dx
        curlEz = ((Ey[1:,:,1:-1] - Ey[:-1,:,1:-1]) - (Ex[:,1:,1:-1] - Ex[:,:-1,1:-1])) / dx
        
        Bx[1:-1] += dt * curlEx
        By[:,1:-1] += dt * curlEy
        Bz[:,:,1:-1] += dt * curlEz
        
        # Hyperbolic cleaning
        divB = (cp.gradient(Bx_c, dx, axis=0) + cp.gradient(By_c, dx, axis=1) + cp.gradient(Bz_c, dx, axis=2))
        psi -= dt * (2.0**2 * divB + (2.0 / (0.8 * dx)) * psi)
        Bx[1:-1] -= dt * (psi[1:,:,:] - psi[:-1,:,:]) / dx
        By[:,1:-1] -= dt * (psi[:,1:,:] - psi[:,:-1,:]) / dx
        Bz[:,:,1:-1] -= dt * (psi[:,:,1:] - psi[:,:,:-1]) / dx
        
        # Gravity with plasma stress-energy
        if step % 50 == 0:
            rho_k = cp.fft.fftn(rho_eff)          # <--- using ρ_eff
            kx = 2*cp.pi*cp.fft.fftfreq(N, d=dx)
            KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
            k2 = KX**2 + KY**2 + KZ**2 + 1e-12
            Phi_k = -4*cp.pi*G*rho_k / k2
            Phi = cp.real(cp.fft.ifftn(Phi_k))
        
        g_x = -cp.gradient(Phi, dx, axis=0)
        g_y = -cp.gradient(Phi, dx, axis=1)
        g_z = -cp.gradient(Phi, dx, axis=2)
        
        # JxB Lorentz force
        Jx = (cp.gradient(Bz_c, dx, axis=1) - cp.gradient(By_c, dx, axis=2)) / mu0
        Jy = (cp.gradient(Bx_c, dx, axis=2) - cp.gradient(Bz_c, dx, axis=0)) / mu0
        Jz_total = (cp.gradient(By_c, dx, axis=0) - cp.gradient(Bx_c, dx, axis=1)) / mu0
        JxB_x = Jy * Bz_c - Jz_total * By_c
        JxB_y = Jz_total * Bx_c - Jx * Bz_c
        
        P_cr = u_cr / 3.0 if use_cr else 0.0
        P_tot = p_thermal + P_cr + B2 / (2*mu0)
        P_tot = cp.maximum(P_tot, 1e-13)
        
        Fx = JxB_x + g_x - cp.gradient(P_tot, dx, axis=0)
        Fy = JxB_y + g_y - cp.gradient(P_tot, dx, axis=1)
        Fz = JxB_z + g_z - cp.gradient(P_tot, dx, axis=2)
        
        if use_dm:
            grav_dm = -6.6743e-11 * M_enc_dm / (r_sph**2 + 1e-8)
            Fx += grav_dm * (X / r_sph)
            Fy += grav_dm * (Y / r_sph)
            Fz += grav_dm * (Z / r_sph)
        
        vx += dt * Fx / (rho + 1e-30)
        vy += dt * Fy / (rho + 1e-30)
        vz += dt * Fz / (rho + 1e-30)
        
        vx = cp.clip(vx, -max_v, max_v)
        vy = cp.clip(vy, -max_v, max_v)
        vz = cp.clip(vz, -max_v, max_v)
        
        div_v = (cp.gradient(rho*vx, dx, axis=0) + cp.gradient(rho*vy, dx, axis=1) + cp.gradient(rho*vz, dx, axis=2))
        rho += dt * (-div_v)
        rho = cp.maximum(rho, 1e-25)
        
        if use_cr:
            div_cr = (cp.gradient(u_cr*vx, dx, axis=0) + cp.gradient(u_cr*vy, dx, axis=1) + cp.gradient(u_cr*vz, dx, axis=2))
            lap_cr = sum(cp.gradient(cp.gradient(u_cr, dx, axis=i), dx, axis=i) for i in range(3))
            source = 2.5e-15 * cp.exp(-r_cyl / 8.0) * cp.exp(-cp.abs(Z)/3.0)
            u_cr += dt * (-div_cr + kappa_cr * lap_cr + source)
            u_cr = cp.maximum(u_cr, 1e-15)
        
        T = p_thermal * (1.4 * mH) / (rho * kB)
        T = cp.clip(T, 100.0, 1e6)
        Lambda = 1e-22 * cp.sqrt(T / 1e4)
        nH2 = (rho / mH)**2
        cooling = nH2 * Lambda * 1e-7
        E_total -= dt * cooling
        E_total -= dt * p_thermal * div_v / rho
        
        rho = cp.nan_to_num(rho, nan=1e-25, posinf=1e-20, neginf=1e-25)
        vx = cp.nan_to_num(vx, nan=0.0, posinf=max_v, neginf=-max_v)
        vy = cp.nan_to_num(vy, nan=0.0, posinf=max_v, neginf=-max_v)
        vz = cp.nan_to_num(vz, nan=0.0, posinf=max_v, neginf=-max_v)
        E_total = cp.nan_to_num(E_total, nan=1e-10, posinf=1e-10, neginf=1e-10)
        
        if step % 50 == 0 or step == steps-1:
            Bmax = cp.sqrt(B2).max() * 1e6
            vmax = cp.sqrt(vx**2 + vy**2 + vz**2).max() / 1000
            print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")
    
    # Rotation curve
    mid = N//2
    r_mid = r_cyl[:,:,mid].flatten().get()
    v_phi_mid = ((X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / (r_cyl[:,:,mid] + 1e-8)).flatten().get() / 1000
    bins = np.linspace(0, L/2, 60)
    v_rot, _ = np.histogram(r_mid, bins=bins, weights=v_phi_mid)
    counts, _ = np.histogram(r_mid, bins=bins)
    v_rot = v_rot / (counts + 1e-8)
    r_centers = (bins[:-1] + bins[1:]) / 2
    
    print("\nRotation curve summary (km/s):")
    print("Inner (0-5 kpc) avg =", np.mean(v_rot[:10]))
    print("Mid (5-15 kpc) avg =", np.mean(v_rot[10:30]))
    print("Outer (15-30 kpc) avg =", np.mean(v_rot[30:]))
    
    return {'r': r_centers, 'v_phi': v_rot}

print("Running Pure Plasma mode...")
plasma_data = run_simulation(False, True)

print("\nRunning DM mode...")
dm_data = run_simulation(True, True)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(plasma_data['r'], plasma_data['v_phi'], 'cyan', lw=2.5, label='Pure Plasma')
plt.plot(dm_data['r'], dm_data['v_phi'], 'orange', lw=2.5, label='With DM')
plt.axhline(230, color='red', ls='--', label='Observed flat ~230 km/s')
plt.xlabel('Radius (kpc)')
plt.ylabel('Rotation velocity (km/s)')
plt.title('Rotation Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('rotation_curves_v8.4.png')

print("✅ v8.4 complete with plasma stress-energy gravity! Plot should appear now.")
