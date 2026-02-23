import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq
import scipy.signal as signal
from scipy.special import j0  # Bessel J0 for radial profile

# ----------------- SIMULATION PARAMETERS -----------------
GRID_SIZE = 128
MAX_STEPS = 120000  # <--- 120k steps
dt_max    = 1.2e-4
CFL       = 0.75
print("🚀 ALADIN v1.6 – 120k steps – Bessel + Kink seed")
print(f"dt_max = {dt_max:.1e}, CFL = {CFL}, expected runtime ~1–2 hours on A100")

c = 3.0e8
J0 = 1.0e18
J_pl = 43.0 ** 3
alpha = c * (J0 / J_pl)**(1/3.0)
f_res = c * (J0 ** (1/3.0)) / alpha
print(f"Derived resonance frequency: {f_res:.10f} Hz")

N_GRID = GRID_SIZE
t_max = 5.0

# ----------------- PHYSICS CONSTANTS -----------------
mu0 = 4 * np.pi * 1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
omega_res = 2 * np.pi * f_res
m_phi = omega_res / c

kappa_maj = 0.04
y_ferm = 1e-3
g_genie = 8e-4
kg_damping = 0.08
C_field_pump = 1e-3

g_damp_base = 0.5
k_genie = 0.2
g_damp_max = 2.0 * g_damp_base
y_genie = 0.1
gamma = 5/3
kappa_thermal = 0.001
R_gas = 1.0

N_crit = 5e5
Gamma0 = 4e8
k_rp = 0.25
g_C = 2e-3
nu_num = 0.001
eta = 0.01

os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ----------------- GRID SETUP -----------------
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

k_sausage = cp.float64(0.7)

# ----------------- UTILITY FUNCTIONS -----------------
def safe(arr, pos=1e6, neg=-1e6):
    return cp.nan_to_num(cp.asanyarray(arr), nan=0.0, posinf=pos, neginf=neg)

def cyl_gradient(field):
    grad_r = cp.gradient(field, dr, axis=0)
    grad_phi = cp.gradient(field, dphi, axis=1) / R_safe
    grad_z = cp.gradient(field, dz, axis=2)
    return safe(grad_r), safe(grad_phi), safe(grad_z)

def cyl_laplacian(field):
    dphi_dr = cp.gradient(field, dr, axis=0)
    term_r = invR * cp.gradient(R_safe * dphi_dr, dr, axis=0)
    term_phi = invR2 * cp.gradient(cp.gradient(field, dphi, axis=1), dphi, axis=1)
    term_z = cp.gradient(cp.gradient(field, dz, axis=2), dz, axis=2)
    return safe(term_r + term_phi + term_z)

def cyl_divergence(vr, vphi, vz):
    term_r = invR * cp.gradient(R_safe * vr, dr, axis=0)
    term_phi = cp.gradient(vphi, dphi, axis=1)
    term_z = cp.gradient(vz, dz, axis=2)
    return safe(term_r + term_phi + term_z)

# ----------------- INITIAL FIELDS -----------------
rho = cp.full((N_GRID, N_GRID, N_GRID), rho0)
v_r = cp.zeros_like(rho)
v_phi = cp.zeros_like(rho)
v_z = cp.zeros_like(rho)

J_z = J0 * cp.exp(-R**2 / a**2)
B_phi = mu0 * J0 * a * (1 - cp.exp(-R**2 / a**2))
B_r = cp.zeros_like(rho)
B_z = cp.zeros_like(rho)
p = cp.full_like(rho, rho0 * vA**2)
T = p / (rho * R_gas + 1e-10)

# ----------------- Bessel J0 radial profile (sausage) -----------------
r_1d = r.get()
j0_kr = j0(k_sausage * r_1d)
j0_kr = cp.array(j0_kr)
j0_kr = cp.maximum(j0_kr, 0.0)
j0_kr /= cp.max(cp.abs(j0_kr))
j0_3d = j0_kr[:, cp.newaxis, cp.newaxis]
rho += 1.0 * rho0 * j0_3d * cp.cos(k_sausage * Z)

# ----------------- KINK SEED -----------------
v_phi += 0.1 * vA * cp.sin(Phi)  # small m=1 kink
v_r += 0.05 * vA * cp.cos(Phi)

# ----------------- GENIE + C FIELD -----------------
genie_phi = cp.zeros_like(rho)
genie_phi_prev = genie_phi.copy()
C_field = cp.full_like(rho, 0.005)
J_pineal = cp.full_like(rho, 5e7)

# ----------------- HISTORY -----------------
genie_center_history = []
C_mean_history = []
time_history = []
history = {
    "E_total": [], "genie_amp": [], "C_mean": [], "T_mean": [],
    "energy_drift": [], "max_v_over_c": [],
    "ferm_dirac_mean": [], "ferm_majorana_mean": [],
    "ferm_total_mean": [], "ferm_density_mean": []
}

t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)
step = 0
progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.6 – 120k steps")

# ----------------- MAIN LOOP -----------------
while t < t_max and step < MAX_STEPS:
    step += 1
    vmax = cp.max(cp.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-12))
    vA_local = cp.max(cp.sqrt((B_r**2 + B_phi**2 + B_z**2) / (mu0 * rho + 1e-12)))
    min_length = cp.min(cp.array([dr, dz, cp.mean(R_safe)*dphi]))
    dt_new = CFL * min_length / (vmax + vA_local + 1e-6)
    dt = 0.7*dt_prev + 0.3*dt_new
    dt = cp.clip(dt, 5e-6, dt_max)
    dt_prev = dt
    t += dt
    time_history.append(float(t))

    # ----------------- SIMPLE MHD UPDATE -----------------
    J_r = (cp.gradient(B_z, dphi, axis=1)/R_safe - cp.gradient(B_phi, dz, axis=2))/mu0
    J_phi = (cp.gradient(B_r, dz, axis=2) - cp.gradient(B_z, dr, axis=0))/mu0
    J_z = (cp.gradient(R_safe*B_phi, dr, axis=0)/R_safe - cp.gradient(B_r, dphi, axis=1))/mu0
    v2 = v_r**2 + v_phi**2 + v_z**2
    gamma_rel = 1.0 / cp.sqrt(1.0 - cp.clip(v2,0,0.99*c**2)/c**2 + 1e-18)
    JxB_r = gamma_rel * (J_phi*B_z - J_z*B_phi)
    JxB_phi = gamma_rel * (J_z*B_r - J_r*B_z)
    JxB_z = gamma_rel * (J_r*B_phi - J_phi*B_r)
    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)
    v_r += dt*(JxB_r - grad_p_r)/(rho*gamma_rel+1e-7)
    v_phi += dt*(JxB_phi - grad_p_phi)/(rho*gamma_rel+1e-7)
    v_z += dt*(JxB_z - grad_p_z)/(rho*gamma_rel+1e-7)
    v_r = cp.clip(v_r, -0.1*c, 0.1*c)
    v_phi = cp.clip(v_phi, -0.1*c, 0.1*c)
    v_z = cp.clip(v_z, -0.1*c, 0.1*c)
    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt*drho_dt
    rho = cp.clip(rho, 1e-8, 10*rho0)
    div_v = cyl_divergence(v_r, v_phi, v_z)
    p += dt*((gamma-1)*(-p*div_v) + kappa_thermal*cyl_laplacian(p))
    p = cp.clip(p, 1e-6, 100*cp.mean(p))
    T = p / (rho*R_gas + 1e-10)

    # ----------------- GENIE UPDATE -----------------
    ferm_dirac_mass = y_ferm*cp.abs(genie_phi)
    ferm_majorana_mass = kappa_maj*(genie_phi**2)
    ferm_total_mass = ferm_dirac_mass + ferm_majorana_mass
    majorana_backreaction = 5e-6 * ferm_total_mass
    source_genie = g_genie*J_z + majorana_backreaction + 0.2*cp.sin(2*cp.pi*43*t)*(1+0.6*cp.cos(k_sausage*Z + 0.8*t))
    lap_now = cyl_laplacian(genie_phi)
    accel = lap_now - m_phi**2*genie_phi - 0.005*genie_phi**3 + source_genie
    genie_vel = (genie_phi - genie_phi_prev)/dt_prev if step>1 else cp.zeros_like(genie_phi)
    genie_phi_new = genie_phi + dt*genie_vel + 0.5*dt**2*accel
    genie_phi_new *= cp.exp(-kg_damping*dt)
    genie_phi_prev = genie_phi.copy()
    genie_phi = genie_phi_new

    Gamma_sr = Gamma0*(N_crit*C_field**2)/(1+N_crit*C_field**2)
    pump_eff = C_field_pump*(1+k_rp*C_field)
    pineal_res = 0.18*cp.sin(2*cp.pi*43*t+0.15)
    C_source = g_C*genie_phi*(1 + pineal_res*J_pineal/J0)
    C_field += dt*(C_source + pump_eff*(1-C_field**2) - 0.02*C_field**3 - Gamma_sr*C_field)
    C_field = cp.clip(C_field, 0.0, 25.0)
    rho += 3e-7*C_field**2*dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    # ----------------- HISTORY -----------------
    center = N_GRID//2
    genie_center_history.append(float(genie_phi[center, center, center]))
    C_mean_history.append(float(cp.mean(C_field)))
    E_total = float(cp.mean(0.5*rho*v2) + cp.mean(0.5*(B_r**2+B_phi**2+B_z**2)/mu0))
    drift = abs((E_total - (history["E_total"][-1] if history["E_total"] else E_total)) / (history["E_total"][-1]+1e-10)) if history["E_total"] else 0.0
    history["E_total"].append(E_total)
    history["genie_amp"].append(float(cp.mean(cp.abs(genie_phi))))
    history["C_mean"].append(float(cp.mean(C_field)))
    history["T_mean"].append(float(cp.mean(T)))
    history["energy_drift"].append(drift)
    history["max_v_over_c"].append(float(cp.max(cp.sqrt(v2))/c))
    history["ferm_dirac_mean"].append(float(cp.mean(ferm_dirac_mass)))
    history["ferm_majorana_mean"].append(float(cp.mean(ferm_majorana_mass)))
    history["ferm_total_mean"].append(float(cp.mean(ferm_total_mass)))
    history["ferm_density_mean"].append(float(cp.mean(0.01+0.005*(ferm_total_mass**2))))

    if step % 5000 == 0:
        maj_frac = history["ferm_majorana_mean"][-1] / (history["ferm_total_mean"][-1]+1e-20)
        print(f"Step {step:6d} | t = {float(t):.6f} | dt = {float(dt):.2e} | genie = {history['genie_amp'][-1]:.2e} | Majorana_frac = {maj_frac*100:.1f}%")

    progress_bar.update(1)

progress_bar.close()

# ----------------- POST-PROCESSING -----------------
print("\n=== POST-PROCESSING: 43Hz, Sausage & Kink ===")
t_np = np.array(time_history)
ferm_dirac_final = y_ferm*cp.abs(genie_phi)
ferm_majorana_final = kappa_maj*(genie_phi**2)
ferm_total_final = ferm_dirac_final + ferm_majorana_final
ferm_density_final = 0.01 + 0.005*(ferm_total_final**2)

# --- 43Hz bandpass ---
if len(genie_center_history) > 200:
    sig_genie = np.array(genie_center_history)
    sig_C = np.array(C_mean_history)
    t_uniform = np.linspace(t_np.min(), t_np.max(), len(t_np))
    dt_sim = t_uniform[1]-t_uniform[0]
    sig_genie_u = np.interp(t_uniform, t_np, sig_genie)
    sig_C_u = np.interp(t_uniform, t_np, sig_C)
    sig_genie_detrend = signal.detrend(sig_genie_u)
    sig_C_detrend = signal.detrend(sig_C_u)
    sos = signal.butter(5, [40,46], btype='band', fs=1/dt_sim, output='sos')
    sig_genie_filtered = signal.sosfiltfilt(sos, sig_genie_detrend)
    sig_C_filtered = signal.sosfiltfilt(sos, sig_C_detrend)
    freq = rfftfreq(len(t_uniform), d=dt_sim)
    power_genie = np.abs(rfft(sig_genie_filtered))**2
    idx_43 = np.argmin(np.abs(freq-43.0))
    peak_power_genie = power_genie[idx_43]
    verdict = "❌ no clear 43 Hz"
    if peak_power_genie>1e4:
        verdict="✅ STRONG 43 Hz LOCK (filtered)"
    print(f"Simulated time span: {t_np[-1]:.6f}")
    print(f"Peak 43 Hz power: {peak_power_genie:.2e} | Verdict: {verdict}")

# --- Sausage m=0 ---
ferm_mass_zmean = cp.mean(ferm_total_final, axis=(0,1)).get()
k = np.fft.rfftfreq(len(ferm_mass_zmean), d=float(dz))
power_mass = np.abs(np.fft.rfft(ferm_mass_zmean))**2 / len(ferm_mass_zmean)
dom_k_mass = k[np.argmax(power_mass[1:])+1] if len(power_mass)>1 else 0
sausage_power = np.mean(power_mass[np.abs(k-float(k_sausage))<0.15])
print(f"Dominant z-mode k = {dom_k_mass:.3f} (sausage k={float(k_sausage):.3f}) | Power in sausage band = {sausage_power:.2e}")

# --- Kink m=1 ---
phi_mean = cp.mean(ferm_total_final, axis=(0,2)).get()
k_phi = np.fft.rfftfreq(N_GRID, d=float(dphi))
power_phi = np.abs(np.fft.rfft(phi_mean))**2 / N_GRID
print("Azimuthal mode power (kink m=1..5):")
for m_val in range(1,6):
    if m_val<len(power_phi):
        print(f"m={m_val} power = {power_phi[m_val]:.2e}")

print("\nALADIN v1.6 – 120k steps finished! ✅ Full diagnostics computed.")
