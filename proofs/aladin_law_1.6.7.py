"""
ALADIN v1.6.7 — Refined QED Heuristic (Phenomenological)
Following ChatGPT critic suggestions: more honesty, better scaling, safeguards
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.makedirs('aladin_plots', exist_ok=True)

print("🚀 ALADIN v1.6.7 — Refined QED Cascade Heuristic\n")

# ========================= GRID =========================
n = 128
r_max = 2.0
z_max = 8.0

r = np.linspace(r_max/(2*n), r_max, n, dtype=np.float32)
theta = np.linspace(0, 2*np.pi, n, endpoint=False, dtype=np.float32)
z = np.linspace(-z_max/2, z_max/2, n, dtype=np.float32)
R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')

dr = r[1] - r[0]
dtheta = theta[1] - theta[0]
dz = z[1] - z[0]
R_safe = np.maximum(R, dr)

mu0 = 4 * np.pi * 1e-7
c = 3.0e8
gamma_ad = 5.0/3
EPS = 1e-12
CFL = 0.06

E_c = 1.3e18                      # Schwinger critical field

# ================== TUNABLE QED PARAMETERS ==================
schwinger_prefactor = 1.5e-3      # Control base Schwinger strength
bw_strength = 9.0                 # Breit-Wheeler multiplication strength
pair_scaling = 3e24               # Density injection scaling (heuristic!)
energy_fraction = 0.32            # Fraction of pair energy added to E_tot

# ====================== FIELDS ======================
Br  = np.zeros((n+1, n, n), dtype=np.float32)
Bth = np.zeros((n, n+1, n), dtype=np.float32)
Bz  = np.zeros((n, n, n+1), dtype=np.float32)

rho   = np.full((n, n, n), 1.75e12, dtype=np.float32)
mr    = np.zeros((n, n, n), dtype=np.float32)
mt    = np.zeros((n, n, n), dtype=np.float32)
mz    = rho * (0.55 * c * np.exp(-(R / 0.42)**2))
E_tot = np.full_like(rho, 2.5e13, dtype=np.float32)

Bth_cc = (mu0 * 1e18 * R / 2.0) * np.exp(-(R / 0.6)**2)
Bth[:, :-1, :] = Bth_cc
Bz[:, :, :-1] = 0.015 * mu0 * 1e18 * r_max

# ====================== FUNCTIONS ======================
def primitive_recovery(rho, mr, mt, mz, E_tot):
    vr = mr / (rho + EPS)
    vth = mt / (rho + EPS)
    vz = mz / (rho + EPS)
    kinetic = 0.5 * rho * (vr**2 + vth**2 + vz**2)
    Br_avg = 0.5*(Br[:-1] + Br[1:])
    Bt_avg = 0.5*(Bth[:,:-1,:] + Bth[:,1:,:])
    Bz_avg = 0.5*(Bz[:,:,:-1] + Bz[:,:,1:])
    B2 = Br_avg**2 + Bt_avg**2 + Bz_avg**2
    p = (gamma_ad - 1) * (E_tot - kinetic - B2/(2*mu0))
    p = np.maximum(p, 1e9)
    return vr, vth, vz, p

def get_growth(mode_hist, t_hist):
    if len(mode_hist) < 25: return 0.0
    log_amp = np.log(np.maximum(np.array(mode_hist), 1e-8))
    slope, _ = np.polyfit(t_hist, log_amp, 1)
    return max(slope, 0.0)

# ====================== SCAN ======================
J0_list = np.logspace(5.3, 7.0, 10)
results = []

for J0 in tqdm(J0_list, desc="v1.6.7 QED Test"):
    rho.fill(1.75e12)
    mr.fill(0); mt.fill(0)
    mz[:] = rho * (0.55 * c * np.exp(-(R / 0.42)**2))
    E_tot.fill(2.5e13)
    Bth[:, :-1, :] = (mu0 * J0 * R / 2.0) * np.exp(-(R / 0.6)**2)
    
    mode_m1 = []
    t = 0.0
    dt = 1e-9
    
    for step in range(130):
        vr, vth, vz, p = primitive_recovery(rho, mr, mt, mz, E_tot)
        
        vmax = np.max(np.abs([vr, vth, vz])) + 1e7
        dt = CFL * min(dr, R.min()*dtheta, dz) / vmax
        t += dt

        # Induction (simplified v×B)
        Bth[:, :-1, :] += dt * 0.35 * (vr[:,np.newaxis,:] * (Bz[:,:,1:] - Bz[:,:,:-1])/dz)

        # Lorentz force
        dBth = (Bth[1:,:-1,:] - Bth[:-1,:-1,:]) / dr
        Jz = (1/mu0) * (dBth.mean(axis=1) + Bth.mean(axis=1)/R)
        force_r = Jz * Bth.mean(axis=1)
        mr += dt * force_r * rho

        # === IMPROVED QED CASCADE HEURISTIC ===
        E_mag = np.maximum(np.abs(vr * Bth.mean(axis=1)), 1e15)
        chi = E_mag / E_c
        
        schwinger_rate = schwinger_prefactor * np.exp(-np.pi / np.maximum(chi, 1e-6))
        bw_mult = 1.0 + bw_strength * chi**1.6
        
        pair_rate = schwinger_rate * bw_mult * (1.0 + 0.25 * step)   # temporal ramp
        
        delta_rho = pair_rate * dt * pair_scaling
        rho += delta_rho
        E_tot += delta_rho * c**2 * energy_fraction

        # Modal extraction
        vr_mean = np.mean(vr, axis=2)
        fft_m = np.abs(np.fft.fft(vr_mean, axis=1))
        power_m1 = np.sum(fft_m[:,1]**2 * R[:,0])
        mode_m1.append(power_m1)
    
    gamma = get_growth(mode_m1, np.arange(len(mode_m1))*dt)
    Pi = (mu0 * J0**2 * r_max**2) / (rho.mean() * c**2)
    results.append([Pi, gamma])
    print(f"J0={J0:.2e} | Π={Pi:.3f} | γ={gamma:.4f} | ρ_mean={rho.mean():.2e}")

results = np.array(results)

plt.figure(figsize=(13, 7))
plt.scatter(results[:,0], results[:,1], c='red', s=150, label='m=1 Kink Growth')
plt.axvline(x=8.0, color='black', ls='--', lw=3, label='Proposed Π_crit ≈ 8')
plt.xscale('log')
plt.xlabel(r'$\Pi = \mu_0 J^2 R^2 / (\rho c^2)$')
plt.ylabel(r'Growth rate $\gamma$ (s$^{-1}$)')
plt.title('ALADIN v1.6.7 — Refined QED Cascade Heuristic')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('aladin_plots/v1.6.7_QED_Refined.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ v1.6.7 ready. Run and send me the printed table + plot description.")
