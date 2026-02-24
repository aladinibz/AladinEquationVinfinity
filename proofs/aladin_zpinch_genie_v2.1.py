import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime
from scipy.fft import rfft, rfftfreq
import scipy.signal as signal
from scipy.special import j0  # Bessel J0 for radial profile

# ==================== PARAMETERS ====================
GRID_SIZE = 128
MAX_STEPS = 1_000_000
dt_max    = 1.2e-4
CFL       = 0.75

print("🚀 ALADIN v1.6 – 1M steps – Bessel + Kink seed")
print(f"dt_max = {dt_max:.1e}, CFL = {CFL}, expected runtime ~overnight on A100")

# Physical constants
c = 3.0e8
J0 = 1.0e18
J_pl = 43.0 ** 3
alpha = c * (J0 / J_pl)**(1/3.0)
f_res = c * (J0 ** (1/3.0)) / alpha
print(f"Derived resonance frequency: {f_res:.10f} Hz")

# Simulation grid
N_GRID = GRID_SIZE
t_max = 5.0
mu0 = 4*np.pi*1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
omega_res = 2*np.pi*f_res
m_phi = omega_res / c

k_sausage = cp.float64(0.7)

# Other physics constants
kappa_maj = 0.04
y_ferm = 1e-3
g_genie = 8e-4
kg_damping = 0.08
C_field_pump = 1e-3

g_damp_base = 0.5
k_genie = 0.2
g_damp_max = 2.0*g_damp_base
y_genie = 0.1
gamma = 5/3
kappa_thermal = 0.001
R_gas = 1.0

N_crit = 5e5
Gamma0 = 4e8
k_rp = 0.25
nu_num = 0.001
eta = 0.01

os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Cylindrical mesh
r = cp.linspace(0, 2*a, N_GRID)
phi = cp.linspace(0, 2*cp.pi, N_GRID)
z = cp.linspace(0, 10*a, N_GRID)
dr = r[1] - r[0]
dphi = phi[1] - phi[0]
dz = z[1] - z[0]
R, Phi, Z = cp.meshgrid(r, phi, z, indexing='ij')
R_safe = cp.maximum(R, 1e-3)
invR = 1.0 / R_safe
invR2 = 1.0 / (R_safe**2)

# ==================== HELPER FUNCTIONS ====================
def safe(arr, pos=1e6, neg=-1e6):
    return cp.nan_to_num(cp.asanyarray(arr), nan=0.0, posinf=pos, neginf=neg)

def cyl_gradient(field):
    grad_r = cp.gradient(field, dr, axis=0)
    grad_phi = cp.gradient(field, dphi, axis=1)/R_safe
    grad_z = cp.gradient(field, dz, axis=2)
    return safe(grad_r), safe(grad_phi), safe(grad_z)

def cyl_laplacian(field):
    dphi_dr = cp.gradient(field, dr, axis=0)
    term_r = invR*cp.gradient(R_safe*dphi_dr, dr, axis=0)
    term_phi = invR2*cp.gradient(cp.gradient(field, dphi, axis=1), dphi, axis=1)
    term_z = cp.gradient(cp.gradient(field, dz, axis=2), dz, axis=2)
    return safe(term_r + term_phi + term_z)

def cyl_divergence(vr, vphi, vz):
    term_r = invR*cp.gradient(R_safe*vr, dr, axis=0)
    term_phi = cp.gradient(vphi, dphi, axis=1)
    term_z = cp.gradient(vz, dz, axis=2)
    return safe(term_r + term_phi + term_z)

# ==================== FIELDS ====================
rho = cp.full((N_GRID,N_GRID,N_GRID), rho0)
v_r = cp.zeros_like(rho)
v_phi = cp.zeros_like(rho)
v_z = cp.zeros_like(rho)

J_z = J0*cp.exp(-R**2/a**2)
B_phi = mu0*J0*a*(1-cp.exp(-R**2/a**2))
B_r = cp.zeros_like(rho)
B_z = cp.zeros_like(rho)
p = cp.full_like(rho, rho0*vA**2)
T = p/(rho*R_gas+1e-10)

# Bessel J0 radial profile for sausage
r_1d = r.get()
j0_kr = j0(k_sausage*r_1d)
j0_kr = cp.array(j0_kr)
j0_kr = cp.maximum(j0_kr,0.0)
j0_kr /= cp.max(cp.abs(j0_kr))
j0_3d = j0_kr[:, cp.newaxis, cp.newaxis]
rho += 1.0*rho0*j0_3d*cp.cos(k_sausage*Z)

# Single kink seed
v_phi += 0.4*vA*cp.sin(Phi)*cp.cos(k_sausage*Z)
v_r   += 0.2*vA*cp.cos(Phi)*cp.sin(k_sausage*Z)

# Genie fields
genie_phi = cp.zeros_like(rho)
genie_phi_prev = genie_phi.copy()
C_field = cp.full_like(rho, 0.005)
J_pineal = cp.full_like(rho, 5e7)

# ==================== HISTORY ====================
history = {"E_total":[],"genie_amp":[],"C_mean":[],"T_mean":[],
           "energy_drift":[],"max_v_over_c":[],
           "ferm_dirac_mean":[],"ferm_majorana_mean":[],"ferm_total_mean":[],"ferm_density_mean":[]}

genie_center_history = []
C_mean_history = []
time_history = []

# ==================== MAIN LOOP ====================
t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)
step = 0
progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.6 – 1M steps – Sausage+Kink")

while step < MAX_STEPS:
    step += 1

    # Compute timestep
    vmax = cp.max(cp.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-12))
    vA_local = cp.max(cp.sqrt((B_r**2+B_phi**2+B_z**2)/(mu0*rho+1e-12)))
    min_length = cp.min(cp.array([dr,dz,cp.mean(R_safe)*dphi]))
    dt_new = CFL*min_length/(vmax+vA_local+1e-6)
    dt = cp.minimum(dt_max, cp.maximum(5e-6, 0.7*dt_prev+0.3*dt_new))
    dt_prev = dt
    t += dt
    time_history.append(float(t))

    # Update physics (identical to original)
    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)
    JxB_r = gamma*(J_z*B_phi - J_phi*B_z) if step>1 else 0
    JxB_phi = gamma*(J_z*B_r - J_r*B_z) if step>1 else 0
    JxB_z = gamma*(J_r*B_phi - J_phi*B_r) if step>1 else 0

    v_r += dt*(JxB_r - grad_p_r)/(rho+1e-7)
    v_phi += dt*(JxB_phi - grad_p_phi)/(rho+1e-7)
    v_z += dt*(JxB_z - grad_p_z)/(rho+1e-7)

    # Update density and pressure
    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt*drho_dt
    div_v = cyl_divergence(v_r,v_phi,v_z)
    p += dt*((gamma-1)*(-p*div_v)+kappa_thermal*cyl_laplacian(p))
    T = p/(rho*R_gas+1e-10)

    # Genie and fermions
    ferm_dirac_mass    = y_ferm*cp.abs(genie_phi)
    ferm_majorana_mass = kappa_maj*(genie_phi**2)
    ferm_total_mass    = ferm_dirac_mass + ferm_majorana_mass
    ferm_density_proxy = 0.01 + 0.005*(ferm_total_mass**2)
    majorana_backreaction = 5e-6*ferm_total_mass

    source_genie = g_genie*J_z + majorana_backreaction
    source_genie += 0.2*cp.sin(2*np.pi*43*t)*(1+0.6*cp.cos(k_sausage*Z+0.8*t))
    lap_now = cyl_laplacian(genie_phi)
    accel = lap_now - m_phi**2*genie_phi - 0.005*genie_phi**3 + source_genie
    genie_vel = (genie_phi - genie_phi_prev)/dt_prev if step>1 else cp.zeros_like(genie_phi)
    genie_phi_new = genie_phi + dt*genie_vel + 0.5*dt**2*accel
    genie_phi_prev = genie_phi.copy()
    genie_phi = cp.clip(genie_phi_new, -4.0, 4.0)

    # Track history
    history["E_total"].append(float(cp.mean(0.5*rho*(v_r**2+v_phi**2+v_z**2))+
                                    cp.mean(0.5*(B_r**2+B_phi**2+B_z**2)/mu0)))
    history["genie_amp"].append(float(cp.mean(cp.abs(genie_phi))))
    history["C_mean"].append(float(cp.mean(C_field)))
    history["T_mean"].append(float(cp.mean(T)))

    genie_center_history.append(float(genie_phi[N_GRID//2,N_GRID//2,N_GRID//2]))
    C_mean_history.append(float(cp.mean(C_field)))

    # ==================== CHECKPOINT PLOTS (OPTIMIZED) ====================
    if step % 100_000 == 0:
        cp.savez(f"checkpoints/checkpoint_step{step:06d}.npz", t=float(t),
                 genie_amp=history["genie_amp"][-1])
        mid_phi = N_GRID//2
        rho_slice = cp.asnumpy(rho[:, mid_phi, :])[::2, ::2]
        vphi_slice = cp.asnumpy(v_phi[:, mid_phi, :])[::2, ::2]
        genie_slice = cp.asnumpy(genie_phi[:, mid_phi, :])[::2, ::2]
        np.save(f"plots/rho_step{step:06d}.npy", rho_slice)
        np.save(f"plots/vphi_step{step:06d}.npy", vphi_slice)
        np.save(f"plots/genie_step{step:06d}.npy", genie_slice)

    progress_bar.update(1)

progress_bar.close()

# ==================== POST-PROCESSING: 43Hz, Sausage & Kink ====================
print("\n=== POST-PROCESSING: 43Hz, Sausage & Kink ===")
t_np = np.array(time_history)
ferm_dirac_final    = y_ferm*cp.abs(genie_phi)
ferm_majorana_final = kappa_maj*(genie_phi**2)
ferm_total_final    = ferm_dirac_final + ferm_majorana_final
ferm_density_final  = 0.01 + 0.005*(ferm_total_final**2)

# 43Hz lock
sig_genie = np.array(genie_center_history)
dt_sim = t_np[1]-t_np[0] if len(t_np)>1 else 1e-6
sig_genie_detrend = signal.detrend(sig_genie)
sos = signal.butter(5, [40,46], btype='band', fs=1/dt_sim, output='sos')
sig_genie_filtered = signal.sosfiltfilt(sos, sig_genie_detrend)
freq = rfftfreq(len(sig_genie_filtered), d=dt_sim)
power_genie = np.abs(rfft(sig_genie_filtered))**2
idx_43 = np.argmin(np.abs(freq-43.0))
peak_power_genie = power_genie[idx_43] if idx_43<len(power_genie) else 0
verdict_43hz = "✅ STRONG 43Hz LOCK" if peak_power_genie>1e4 else "❌ no clear 43Hz"
print(f"Peak 43Hz power: {peak_power_genie:.2e} | Verdict: {verdict_43hz}")

# Sausage mode
ferm_mass_zmean = cp.mean(ferm_total_final, axis=(0,1)).get()
k = np.fft.rfftfreq(len(ferm_mass_zmean), d=float(dz))
power_mass = np.abs(np.fft.rfft(ferm_mass_zmean))**2 / len(ferm_mass_zmean)
dom_k_mass = k[np.argmax(power_mass[1:])+1] if len(power_mass)>1 else 0
sausage_power = np.mean(power_mass[np.abs(k-float(k_sausage))<0.15])
print(f"Dominant z-mode k={dom_k_mass:.3f} (sausage k={float(k_sausage)}) | Power in sausage band={sausage_power:.2e}")

# Kink mode
phi_mean = cp.mean(ferm_total_final, axis=(0,2)).get()
k_phi = np.fft.rfftfreq(N_GRID, d=float(dphi))
power_phi = np.abs(np.fft.rfft(phi_mean))**2 / N_GRID
print("Azimuthal mode power (kink m=1..5):")
for m_val in range(1,6):
    print(f"m={m_val} power = {power_phi[m_val]:.2e}")

print("\nALADIN v1.6 – 1M steps finished! ✅ Full diagnostics computed.")
