# =============================================================================
# ALADIN v1.6 – 120k steps – Bessel + Kink Seed – Sausage + Kink Detection
# =============================================================================
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq
import scipy.signal as signal
from scipy.special import j0  # Bessel J0 radial profile

# ---------------------- SIMULATION PARAMETERS ----------------------
GRID_SIZE = 128
MAX_STEPS = 120_000
dt_max = 1.2e-4
CFL = 0.75

c = 3.0e8
J0 = 1.0e18
J_pl = 43.0**3
alpha = c * (J0/J_pl)**(1/3)
f_res = c*(J0**(1/3))/alpha

print("🚀 ALADIN v1.6 – 120k steps – Bessel + Kink seed")
print(f"dt_max = {dt_max:.1e}, CFL = {CFL}")
print(f"Derived resonance frequency: {f_res:.10f} Hz")

# ---------------------- PHYSICAL CONSTANTS ----------------------
mu0 = 4*np.pi*1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
gamma = 5/3
R_gas = 1.0

# ---------------------- GRID SETUP ----------------------
N_GRID = GRID_SIZE
r = cp.linspace(0, 2*a, N_GRID)
phi = cp.linspace(0, 2*cp.pi, N_GRID)
z = cp.linspace(0, 10*a, N_GRID)
dr = r[1]-r[0]; dphi = phi[1]-phi[0]; dz = z[1]-z[0]
R, Phi, Z = cp.meshgrid(r, phi, z, indexing='ij')
R_safe = cp.maximum(R, 1e-3)
invR = 1.0 / R_safe
invR2 = 1.0 / (R_safe**2)

# ---------------------- INITIAL FIELDS ----------------------
rho = cp.full((N_GRID,N_GRID,N_GRID), rho0)
v_r = cp.zeros_like(rho)
v_phi = cp.zeros_like(rho)
v_z = cp.zeros_like(rho)

# Z-pinch B-field
J_z0 = J0 * cp.exp(-R**2 / a**2)
B_phi = mu0*J0*a*(1 - cp.exp(-R**2 / a**2))
B_r = cp.zeros_like(rho)
B_z = cp.zeros_like(rho)
p = cp.full_like(rho, rho0*vA**2)
T = p / (rho*R_gas + 1e-10)

# ---------------------- SAUSAGE + KINK SEED ----------------------
k_sausage = 0.7
r_1d = r.get()
j0_kr = j0(k_sausage*r_1d)
j0_kr = cp.array(j0_kr)
j0_kr = cp.maximum(j0_kr,0.0)
j0_kr /= cp.max(cp.abs(j0_kr))
j0_3d = j0_kr[:,cp.newaxis,cp.newaxis]
rho += rho0 * j0_3d * cp.cos(k_sausage * Z)

v_phi += 0.4*vA*cp.sin(Phi)*cp.cos(k_sausage*Z)
v_r   += 0.2*vA*cp.cos(Phi)*cp.sin(k_sausage*Z)

# ---------------------- HELPER FUNCTIONS ----------------------
def safe(arr,pos=1e6,neg=-1e6):
    return cp.nan_to_num(cp.asanyarray(arr),nan=0.0,posinf=pos,neginf=neg)

def cyl_gradient(field):
    grad_r = cp.gradient(field, dr, axis=0)
    grad_phi = cp.gradient(field, dphi, axis=1)/R_safe
    grad_z = cp.gradient(field, dz, axis=2)
    return safe(grad_r), safe(grad_phi), safe(grad_z)

def cyl_laplacian(field):
    dphi_dr = cp.gradient(field, dr, axis=0)
    term_r = invR * cp.gradient(R_safe*dphi_dr, dr, axis=0)
    term_phi = invR2 * cp.gradient(cp.gradient(field, dphi, axis=1), dphi, axis=1)
    term_z = cp.gradient(cp.gradient(field, dz, axis=2), dz, axis=2)
    return safe(term_r + term_phi + term_z)

def cyl_divergence(vr,vphi,vz):
    term_r = invR*cp.gradient(R_safe*vr, dr, axis=0)
    term_phi = cp.gradient(vphi, dphi, axis=1)
    term_z = cp.gradient(vz, dz, axis=2)
    return safe(term_r + term_phi + term_z)

# ---------------------- DIAGNOSTIC STORAGE ----------------------
time_history = []
genie_center_history = []
C_mean_history = []

history = {"E_total":[], "genie_amp":[],"C_mean":[],"max_v_over_c":[]}

t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)
step = 0

progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.6 – 120k steps – Sausage+Kink")

# ---------------------- MAIN LOOP ----------------------
while step < MAX_STEPS:
    step += 1

    vmax = cp.max(cp.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-12))
    vA_local = cp.max(cp.sqrt((B_r**2 + B_phi**2 + B_z**2)/(mu0*rho + 1e-12)))
    min_length = cp.min(cp.array([dr, dz, cp.mean(R_safe)*dphi]))
    dt_new = CFL * min_length / (vmax+vA_local + 1e-6)
    dt = 0.7*dt_prev + 0.3*dt_new
    dt = cp.minimum(dt, dt_max)
    dt = cp.maximum(dt, 5e-6)
    dt_prev = dt
    t += dt
    time_history.append(float(t))

    # Compute currents
    J_r   = (cp.gradient(B_z, dphi, axis=1)/R_safe - cp.gradient(B_phi, dz, axis=2))/mu0
    J_phi = (cp.gradient(B_r, dz, axis=2) - cp.gradient(B_z, dr, axis=0))/mu0
    J_z   = (cp.gradient(R_safe*B_phi, dr, axis=0)/R_safe - cp.gradient(B_r, dphi, axis=1))/mu0

    # Lorentz force
    v2 = v_r**2 + v_phi**2 + v_z**2
    gamma_rel = 1.0 / cp.sqrt(1.0 - v2/c**2 + 1e-18)
    gamma_rel = cp.clip(gamma_rel, 1.0, 15.0)
    JxB_r   = gamma_rel * (J_phi*B_z - J_z*B_phi)
    JxB_phi = gamma_rel * (J_z*B_r - J_r*B_z)
    JxB_z   = gamma_rel * (J_r*B_phi - J_phi*B_r)

    # Pressure gradients
    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)

    # Update velocities
    v_r   += dt*(JxB_r - grad_p_r)/(rho*gamma_rel + 1e-7)
    v_phi += dt*(JxB_phi - grad_p_phi)/(rho*gamma_rel + 1e-7)
    v_z   += dt*(JxB_z - grad_p_z)/(rho*gamma_rel + 1e-7)
    v_r   = cp.clip(v_r,-0.1*c,0.1*c)
    v_phi = cp.clip(v_phi,-0.1*c,0.1*c)
    v_z   = cp.clip(v_z,-0.1*c,0.1*c)

    # Update density and pressure
    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt*drho_dt
    rho = cp.clip(rho,1e-8,10*rho0)
    div_v = cyl_divergence(v_r,v_phi,v_z)
    p += dt*((gamma-1)*(-p*div_v))
    p = cp.clip(p,1e-6,100*cp.mean(p))
    T = p/(rho*R_gas + 1e-10)

    # Diagnostics
    E_total = float(cp.mean(0.5*rho*(v_r**2+v_phi**2+v_z**2)) +
                    cp.mean(0.5*(B_r**2 + B_phi**2 + B_z**2)/mu0))
    history["E_total"].append(E_total)
    history["max_v_over_c"].append(float(cp.max(cp.sqrt(v2))/c))
    genie_center_history.append(float(cp.mean(v_phi)))
    C_mean_history.append(float(cp.mean(rho)))

    if step % 1000 == 0:
        progress_bar.set_postfix({"E_total":E_total})
        progress_bar.update(1000)
    else:
        progress_bar.update(1)

progress_bar.close()

# ---------------------- POST-PROCESSING ----------------------
print("\n=== POST-PROCESSING: 43Hz, Sausage & Kink ===")
t_np = np.array(time_history)
sig_genie = np.array(genie_center_history)
sig_C = np.array(C_mean_history)

# 43Hz detection
t_uniform = np.linspace(t_np.min(), t_np.max(), len(t_np))
sig_genie_u = np.interp(t_uniform, t_np, sig_genie)
sig_genie_detrend = signal.detrend(sig_genie_u)
sos = signal.butter(5,[40,46],btype='band',fs=1/(t_uniform[1]-t_uniform[0]),output='sos')
sig_genie_filtered = signal.sosfiltfilt(sos, sig_genie_detrend)
freq = rfftfreq(len(t_uniform), d=t_uniform[1]-t_uniform[0])
power_genie = np.abs(rfft(sig_genie_filtered))**2
idx_43 = np.argmin(np.abs(freq-43.0))
peak_power_genie = power_genie[idx_43] if idx_43<len(power_genie) else 0.0
verdict = "❌ no clear 43 Hz"; verdict="✅ STRONG 43 Hz LOCK (filtered)" if peak_power_genie>1e-4 else verdict
print(f"Peak 43Hz power: {peak_power_genie:.2e} | Verdict: {verdict}")

# Sausage z-mode detection
rho_zmean = cp.mean(rho, axis=(0,1)).get()
k = np.fft.rfftfreq(len(rho_zmean), d=float(dz))
power_mass = np.abs(np.fft.rfft(rho_zmean))**2 / len(rho_zmean)
dom_k_mass = k[np.argmax(power_mass[1:])+1] if len(power_mass)>1 else 0
sausage_power = np.mean(power_mass[np.abs(k- k_sausage)<0.15])
print(f"Dominant z-mode k = {dom_k_mass:.3f} (sausage k = {k_sausage:.3f}) | Power in sausage band = {sausage_power:.2e}")

# Kink azimuthal detection
phi_mean = cp.mean(rho, axis=(0,2)).get()
k_phi = np.fft.rfftfreq(N_GRID, d=float(dphi))
power_phi = np.abs(np.fft.rfft(phi_mean))**2 / N_GRID
print("Azimuthal mode power (kink m=1..5):")
for m_val in range(1,6):
    if m_val < len(power_phi):
        print(f"m={m_val} power = {power_phi[m_val]:.2e}")

# Plots
plt.figure(figsize=(14,6))
plt.plot(t_uniform,sig_genie_filtered,'cyan',lw=2,label='Genie v_phi 40–46Hz')
plt.xlabel('Time'); plt.ylabel('Filtered amplitude'); plt.grid(alpha=0.3); plt.legend()
plt.title('Filtered 43Hz Signal')
plt.savefig('plot_43hz_filtered.png'); plt.close()

plt.figure(figsize=(12,6))
plt.loglog(k[1:], power_mass[1:], 'orange', lw=2.5, label='Rho z-mean')
plt.axvline(k_sausage,color='red',ls='--',lw=2,label='Sausage k')
plt.xlabel('Wavenumber k'); plt.ylabel('Power'); plt.title('Sausage z-Spatial Spectrum'); plt.grid(alpha=0.3,which='both'); plt.legend()
plt.savefig('plot_z_sausage.png'); plt.close()

plt.figure(figsize=(12,6))
m = np.arange(len(power_phi))
plt.semilogy(m,power_phi,'lime',lw=2,label='Azimuthal power')
plt.axvline(1,color='red',ls='--',label='m=1 kink')
plt.xlabel('Azimuthal mode m'); plt.ylabel('Power'); plt.title('Azimuthal Kink Spectrum'); plt.grid(alpha=0.3); plt.legend()
plt.savefig('plot_azimuthal_kink.png'); plt.close()

print("\n✅ ALADIN v1.6 – 120k steps finished with full diagnostics")
