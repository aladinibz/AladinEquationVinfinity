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
MAX_STEPS = 120000
dt_max = 5e-5       # safer for strong seeds
CFL = 0.5           # reduce CFL to stabilize
t_max = 5.0

# Physical constants
c = 3.0e8
J0 = 1.0e18
J_pl = 43.0 ** 3
alpha = c * (J0 / J_pl)**(1/3.0)
f_res = c * (J0 ** (1/3.0)) / alpha

mu0 = 4 * np.pi * 1e-7
rho0 = 1.0
a = 1.0
vA = 1.0
omega_res = 2 * np.pi * f_res
m_phi = omega_res / c

k_sausage = 0.5  # slightly lower for stability
kg_damping = 0.08
gamma = 5/3
R_gas = 1.0
kappa_thermal = 0.001

# Directories
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ==================== GRID ====================
r = cp.linspace(0, 2*a, GRID_SIZE)
phi = cp.linspace(0, 2*cp.pi, GRID_SIZE)
z = cp.linspace(0, 10*a, GRID_SIZE)
dr, dphi, dz = r[1]-r[0], phi[1]-phi[0], z[1]-z[0]

R, Phi, Z = cp.meshgrid(r, phi, z, indexing='ij')
R_safe = cp.maximum(R, 1e-3)
invR = 1.0 / R_safe
invR2 = 1.0 / (R_safe**2)

# ==================== FIELDS ====================
rho = cp.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), rho0)
v_r = cp.zeros_like(rho)
v_phi = cp.zeros_like(rho)
v_z = cp.zeros_like(rho)

J_z = J0 * cp.exp(-R**2 / a**2)
B_phi = mu0 * J0 * a * (1 - cp.exp(-R**2 / a**2))
B_r = cp.zeros_like(rho)
B_z = cp.zeros_like(rho)
p = cp.full_like(rho, rho0*vA**2)
T = p / (rho*R_gas + 1e-10)

# ==================== Bessel J0 Sausage Seed ====================
r_1d = r.get()
j0_kr = j0(k_sausage * r_1d)
j0_kr = cp.array(j0_kr)
j0_kr = cp.maximum(j0_kr, 0.0)
j0_kr /= cp.max(cp.abs(j0_kr))
j0_3d = j0_kr[:, cp.newaxis, cp.newaxis]
rho += 0.1 * rho0 * j0_3d * cp.cos(k_sausage*Z)  # ramped amplitude

# ==================== Kink seed ====================
v_phi += 0.05*vA*cp.sin(Phi)*cp.cos(k_sausage*Z)  # ramped initial kink
v_r   += 0.03*vA*cp.cos(Phi)*cp.sin(k_sausage*Z)

# ==================== AUX FIELDS ====================
genie_phi = cp.zeros_like(rho)
genie_phi_prev = genie_phi.copy()
C_field = cp.full_like(rho, 0.005)
J_pineal = cp.full_like(rho, 5e7)

history = {"E_total": [], "genie_amp": [], "C_mean": [],
           "ferm_dirac_mean": [], "ferm_majorana_mean": [], "ferm_total_mean": []}

time_history, genie_center_history, C_mean_history = [], [], []

# ==================== DERIVED FUNCTIONS ====================
def safe(arr, pos=1e6, neg=-1e6):
    return cp.nan_to_num(cp.asanyarray(arr), nan=0.0, posinf=pos, neginf=neg)

def cyl_gradient(field):
    grad_r = cp.gradient(field, dr, axis=0)
    grad_phi = cp.gradient(field, dphi, axis=1)/R_safe
    grad_z = cp.gradient(field, dz, axis=2)
    return safe(grad_r), safe(grad_phi), safe(grad_z)

def cyl_laplacian(field):
    term_r = invR*cp.gradient(R_safe*cp.gradient(field, dr, axis=0), dr, axis=0)
    term_phi = invR2*cp.gradient(cp.gradient(field, dphi, axis=1), dphi, axis=1)
    term_z = cp.gradient(cp.gradient(field, dz, axis=2), dz, axis=2)
    return safe(term_r + term_phi + term_z)

def cyl_divergence(vr, vphi, vz):
    term_r = invR*cp.gradient(R_safe*vr, dr, axis=0)
    term_phi = cp.gradient(vphi, dphi, axis=1)
    term_z = cp.gradient(vz, dz, axis=2)
    return safe(term_r + term_phi + term_z)

# ==================== MAIN LOOP ====================
t = cp.float64(0.0)
dt_prev = cp.float64(dt_max)
step = 0

progress_bar = tqdm(total=MAX_STEPS, desc="ALADIN v1.6 – 120k steps – Sausage+Kink")

while step < MAX_STEPS and t < t_max:
    step += 1

    vmax = cp.max(cp.sqrt(v_r**2 + v_phi**2 + v_z**2 + 1e-12))
    vA_local = cp.max(cp.sqrt((B_r**2 + B_phi**2 + B_z**2)/(mu0*rho + 1e-12)))
    min_length = cp.min(cp.array([dr, dz, cp.mean(R_safe)*dphi]))
    dt_new = CFL * min_length / (vmax + vA_local + 1e-12)
    dt = 0.7*dt_prev + 0.3*dt_new
    dt = cp.minimum(dt, dt_max)
    dt_prev = dt
    t += dt
    time_history.append(float(t))

    # ==================== gamma / clip ====================
    v2 = v_r**2 + v_phi**2 + v_z**2
    gamma_rel = 1.0 / cp.sqrt(1.0 - v2/c**2 + 1e-12)
    gamma_rel = cp.clip(gamma_rel, 1.0, 10.0)
    v_r = cp.clip(v_r, -0.05*c, 0.05*c)
    v_phi = cp.clip(v_phi, -0.05*c, 0.05*c)
    v_z = cp.clip(v_z, -0.05*c, 0.05*c)

    # ==================== Forces ====================
    JxB_r = gamma_rel*(J_phi*B_z - J_z*B_phi) if step>1 else 0
    JxB_phi = gamma_rel*(J_z*B_r - J_r*B_z) if step>1 else 0
    JxB_z = gamma_rel*(J_r*B_phi - J_phi*B_r) if step>1 else 0

    grad_p_r, grad_p_phi, grad_p_z = cyl_gradient(p)
    v_r += dt*(JxB_r - grad_p_r)/(rho*gamma_rel + 1e-7)
    v_phi += dt*(JxB_phi - grad_p_phi)/(rho*gamma_rel + 1e-7)
    v_z += dt*(JxB_z - grad_p_z)/(rho*gamma_rel + 1e-7)

    drho_dt = -cyl_divergence(rho*v_r, rho*v_phi, rho*v_z)
    rho += dt*drho_dt
    rho = cp.clip(rho, 1e-8, 10*rho0)

    div_v = cyl_divergence(v_r, v_phi, v_z)
    p += dt*((gamma-1)*(-p*div_v) + kappa_thermal*cyl_laplacian(p))
    p = cp.clip(p, 1e-6, 100*cp.mean(p))
    T = p / (rho*R_gas + 1e-10)

    # ==================== Genie Field ====================
    lap_now = cyl_laplacian(genie_phi)
    accel = lap_now - m_phi**2*genie_phi - 0.005*genie_phi**3
    genie_vel = (genie_phi - genie_phi_prev)/dt_prev if step>1 else cp.zeros_like(genie_phi)
    genie_phi_new = genie_phi + dt*genie_vel + 0.5*dt**2*accel
    genie_phi_new *= cp.exp(-kg_damping*dt)
    genie_phi_prev = genie_phi.copy()
    genie_phi = genie_phi_new

    # ==================== History ====================
    history["E_total"].append(float(cp.mean(0.5*rho*(v_r**2+v_phi**2+v_z**2)) +
                                    cp.mean(0.5*(B_r**2+B_phi**2+B_z**2)/mu0)))
    history["genie_amp"].append(float(cp.mean(cp.abs(genie_phi))))
    history["C_mean"].append(float(cp.mean(C_field)))

    if step % 5000 == 0:
        print(f"Step {step} | t={float(t):.5f} | Genie amp={history['genie_amp'][-1]:.2e}")

    progress_bar.update(1)

progress_bar.close()

# ==================== POST-PROCESSING ====================
print("\n=== POST-PROCESSING: 43Hz, Sausage & Kink ===")

# Z-spectrum (sausage)
ferm_mass_zmean = cp.mean(rho, axis=(0,1)).get()
k = np.fft.rfftfreq(len(ferm_mass_zmean), d=float(dz))
power_mass = np.abs(np.fft.rfft(ferm_mass_zmean))**2 / len(ferm_mass_zmean)
dom_k_mass = k[np.argmax(power_mass[1:])+1] if len(power_mass)>1 else 0
sausage_power = np.mean(power_mass[np.abs(k - k_sausage)<0.15])
print(f"Dominant z-mode k = {dom_k_mass:.3f} (sausage k={k_sausage:.3f}) | Power in sausage band = {sausage_power:.2e}")

# Phi-spectrum (kink)
phi_mean = cp.mean(rho, axis=(0,2)).get()
k_phi = np.fft.rfftfreq(GRID_SIZE, d=float(dphi))
power_phi = np.abs(np.fft.rfft(phi_mean))**2 / GRID_SIZE
print("Azimuthal mode power (kink m=1..5):")
for m_val in range(1,6):
    if m_val < len(power_phi):
        print(f"m={m_val} power = {power_phi[m_val]:.2e}")

print("\nALADIN v1.6 – 120k steps finished! ✅ Full diagnostics computed.")
