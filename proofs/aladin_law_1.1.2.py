"""
ALADIN v1.1.2 — Fixed Constrained Transport (CT) + 128³
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

os.makedirs('eigenmode_plots', exist_ok=True)

print("🚀 ALADIN v1.1.2 — Fixed Full CT\n")

# ========================= GRID 128 =========================
GRID_SIZE = 128
r_max = 2.0
z_max = 8.0

nr = ntheta = nz = GRID_SIZE
print(f"Grid: {nr}³")

dr = r_max / nr
dtheta = 2*np.pi / ntheta
dz = z_max / nz

r = np.linspace(dr/2, r_max, nr, dtype=np.float32)
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False, dtype=np.float32)
z = np.linspace(-z_max/2, z_max/2, nz, dtype=np.float32)
R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')

# ========================= CONSTANTS =========================
mu0 = 4 * np.pi * 1e-7
c = 3.0e8
m_e = 9.109e-31
gamma = 5.0/3
EPS = 1e-10
CFL = 0.18
eta = 2.0e-4

J0_list = [8.0e5, 1.25e6, 1.9e6]
rho0_list = [1.1e12, 2.4e12]

n_steps = 110
save_every = 8

results = {'Pi': [], 'gamma_m0': [], 'gamma_m1': [], 'J0': [], 'rho0': []}

start = time.time()

for J0 in J0_list:
    for rho0 in rho0_list:
        rho = np.full_like(R, rho0, dtype=np.float32)
        p = np.full_like(R, 1e10, dtype=np.float32)
        e_int = p / (gamma - 1)

        v_r = np.zeros_like(R, dtype=np.float32)
        v_theta = np.zeros_like(R, dtype=np.float32)
        v_z = np.zeros_like(R, dtype=np.float32)

        # Staggered B fields for CT
        B_r   = np.zeros((nr+1, ntheta, nz), dtype=np.float32)
        B_theta = np.zeros((nr, ntheta+1, nz), dtype=np.float32)
        B_z   = np.zeros((nr, ntheta, nz+1), dtype=np.float32)

        # Initial B_theta (proper staggering)
        B_theta_cc = (mu0 * J0 * R / 2).astype(np.float32) * np.exp(-(R/(0.6*r_max))**2)
        B_theta[:, :-1, :] = B_theta_cc[:, :, np.newaxis]

        B_z[:, :, :-1] = 2800.0

        v_r += 0.004 * np.sin(Theta) * np.exp(-((R-0.78)**2)/0.24)

        t_hist = []
        m0_energy = []
        m1_energy = []
        t = 0.0

        for step in tqdm(range(n_steps), leave=False, desc=f"J0={J0:.1e}"):
            rho = np.maximum(rho, 5e8)
            p = np.maximum(p, 5e8)

            # Cell-centered fields
            B_r_cc = (B_r[:-1] + B_r[1:]) / 2
            B_theta_cc = (B_theta[:, :-1] + B_theta[:, 1:]) / 2
            B_z_cc = (B_z[:, :, :-1] + B_z[:, :, 1:]) / 2

            Bmag = np.sqrt(B_r_cc**2 + B_theta_cc**2 + B_z_cc**2 + EPS)

            vA = Bmag / np.sqrt(mu0 * rho + EPS)
            vfluid = np.sqrt(v_r**2 + v_theta**2 + v_z**2 + EPS)
            max_speed = max(float(vA.max()), float(vfluid.max()), 1e4)
            dt = CFL * min(dr, dz) / max_speed
            dt = min(dt, 6e-7)

            # === CONSTRAINED TRANSPORT EMFs ===
            E_r = -(v_theta * B_z_cc - v_z * B_theta_cc)
            E_theta = -(v_z * B_r_cc - v_r * B_z_cc)
            E_z = -(v_r * B_theta_cc - v_theta * B_r_cc)

            # Update staggered B
            B_r[1:-1] += dt * ((E_theta[:,1:,:] - E_theta[:,:-1,:]) / (R[1:-1,:,np.newaxis] * dtheta) - 
                               (E_z[1:,:,:] - E_z[:-1,:,:]) / dz)

            B_theta[:,1:-1] += dt * ((E_z[1:,:,:] - E_z[:-1,:,:]) / dr - 
                                     (E_r[:,:,1:] - E_r[:,:,:-1]) / dz)

            B_z[:,:,1:-1] += dt * ((1/R[:,:,np.newaxis]) * (R[:,:,np.newaxis] * E_theta[:,1:,:] - R[:,:,np.newaxis] * E_theta[:,:-1,:]) / dtheta -
                                   (E_r[1:,:,:] - E_r[:-1,:,:]) / dr)

            # Resistive diffusion
            B_theta_cc += eta * dt * np.gradient(np.gradient(B_theta_cc, axis=0), axis=0)/dr**2

            # Continuity + Pair
            div_v = np.gradient(v_r, axis=0)/dr + v_r/R + np.gradient(v_theta, axis=1)/(R*dtheta) + np.gradient(v_z, axis=2)/dz
            rho += -rho * div_v * dt
            pair_rate = np.clip(5e-5 * (Bmag / 4.4e9)**2, 0, 2e-4)
            rho += 2 * m_e * pair_rate * dt

            # Momentum
            J_z = (1/(mu0*R)) * np.gradient(R*B_theta_cc, axis=0)/dr
            JxB_r = -J_z * B_theta_cc
            dp_dr = np.gradient(p, axis=0)/dr
            adv_r = v_r * np.gradient(v_r, axis=0)/dr + (v_theta/R)*np.gradient(v_r, axis=1)/dtheta - v_theta**2 / R

            rho_eff = rho + p / c**2 + EPS
            v_r -= dt * (dp_dr - JxB_r + rho * adv_r) / rho_eff
            v_theta -= dt * 0.028 * J_z * B_r_cc / rho_eff

            # Energy
            pdv = -p * div_v
            ohmic = eta * J_z**2
            e_int += (pdv + ohmic) * dt
            p = np.maximum((gamma - 1) * e_int, 5e8)

            t += dt

            if step % save_every == 0:
                Pi = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)
                t_hist.append(t)

                vr_fft = np.fft.fft(v_r, axis=1)
                power_m = np.sum(np.abs(vr_fft)**2 * R[:,:,np.newaxis] * dr * dz, axis=(0,2))

                m0 = float(power_m[0])
                m1 = float(power_m[1]) if len(power_m) > 1 else 0.0

                m0_energy.append(m0)
                m1_energy.append(m1)

        # Growth rates
        def get_growth(t_arr, energy):
            if len(energy) < 30:
                return 0.0
            log_e = np.log(np.array(energy) + 1e-12)
            best_gamma = 0.0
            best_r2 = -np.inf
            window_size = 25
            for i in range(len(log_e) - window_size):
                t_win = t_arr[i:i+window_size]
                log_win = log_e[i:i+window_size]
                if np.std(log_win) < 1e-6:
                    continue
                slope, _ = np.polyfit(t_win, log_win, 1)
                y_pred = slope * t_win
                r2 = 1 - np.sum((log_win - y_pred)**2) / np.sum((log_win - np.mean(log_win))**2)
                if r2 > best_r2:
                    best_r2 = r2
                    best_gamma = slope
            return best_gamma

        gamma_m0 = get_growth(np.array(t_hist), m0_energy)
        gamma_m1 = get_growth(np.array(t_hist), m1_energy)

        Pi_final = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)

        results['Pi'].append(Pi_final)
        results['gamma_m0'].append(gamma_m0)
        results['gamma_m1'].append(gamma_m1)
        results['J0'].append(J0)
        results['rho0'].append(rho0)

        print(f"Pi = {Pi_final:.2f} | γ_m0 = {gamma_m0:.2e} | γ_m1 = {gamma_m1:.2e}")

print(f"\nTotal runtime: {time.time()-start:.1f} s")

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(results['Pi'], results['gamma_m1'], s=160, edgecolors='black', label='m=1 Kink', color='red')
plt.scatter(results['Pi'], results['gamma_m0'], s=140, edgecolors='black', label='m=0 Sausage', color='blue', marker='s')
plt.axvline(8.0, color='red', ls='--', lw=3, label='Proposed Π_crit ≈ 8')
plt.xlabel('Π = μ₀ J² R² / (ρ c²)', fontsize=14)
plt.ylabel('Linear Growth Rate γ (s⁻¹)', fontsize=14)
plt.title('ALADIN v1.1.2 — Fixed Full CT + 128³')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
