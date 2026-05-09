"""
ALADIN v1.1.9 — 128³ LOCKED - Bulletproof Power Extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

os.makedirs('jet_plots', exist_ok=True)

print("🚀 ALADIN v1.1.9 — 128³ LOCKED - Power Extraction Fixed\n")

# ========================= GRID 128 LOCKED =========================
GRID_SIZE = 128
r_max = 2.0
z_max = 8.0

nr = ntheta = nz = GRID_SIZE
print(f"Grid: {nr}³ LOCKED")

dr = r_max / nr
dtheta = 2 * np.pi / ntheta
dz = z_max / nz

r = np.linspace(dr/2, r_max, nr, dtype=np.float32)
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False, dtype=np.float32)
z = np.linspace(-z_max/2, z_max/2, nz, dtype=np.float32)
R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')

# ========================= CONSTANTS =========================
mu0 = 4 * np.pi * 1e-7
c = 3.0e8
m_e = 9.109e-31
gamma_ad = 5.0/3

GAMMA_BULK = 5.0

EPS = 1e-10
CFL = 0.18
eta = 2.0e-4
ch_divb = 0.85

J0_list = [8.0e5, 1.25e6]
rho0_list = [1.1e12, 2.4e12]

n_steps = 100
save_every = 8

results = {'Pi': [], 'Pi_rel': [], 'gamma_m0': [], 'gamma_m1': [], 'J0': [], 'rho0': []}

start = time.time()

for J0 in J0_list:
    for rho0 in rho0_list:
        rho = np.full_like(R, rho0, dtype=np.float32)
        p = np.full_like(R, 1e10, dtype=np.float32)
        e_int = p / (gamma_ad - 1)

        v_r = np.zeros_like(R, dtype=np.float32)
        v_theta = np.zeros_like(R, dtype=np.float32)
        v_z = np.zeros_like(R, dtype=np.float32)

        B_r = np.zeros_like(R, dtype=np.float32)
        B_theta = (mu0 * J0 * R / 2).astype(np.float32) * np.exp(-(R/(0.6*r_max))**2)
        B_z = np.full_like(R, 2800.0, dtype=np.float32)

        v_z += 0.8 * c * np.exp(-(R / (0.4 * r_max))**2)
        v_r += 0.004 * np.sin(Theta) * np.exp(-((R-0.78)**2)/0.24)

        t_hist = []
        m0_energy = []
        m1_energy = []
        t = 0.0

        for step in tqdm(range(n_steps), leave=False, desc=f"J0={J0:.1e}"):
            rho = np.maximum(rho, 5e8)
            p = np.maximum(p, 5e8)

            Bmag = np.sqrt(B_r**2 + B_theta**2 + B_z**2 + EPS)

            vA = Bmag / np.sqrt(mu0 * rho + EPS)
            vfluid = np.sqrt(v_r**2 + v_theta**2 + v_z**2 + EPS)

            max_speed = max(float(vA.max()), float(vfluid.max()), 1e4)
            dt = CFL * min(dr, dz) / max_speed
            dt = min(dt, 6e-7)

            # Induction (simplified)
            vxB_r = v_theta * B_z - v_z * B_theta
            vxB_theta = v_z * B_r - v_r * B_z
            vxB_z = v_r * B_theta - v_theta * B_r

            dBr_dt = (1/R)*np.gradient(R*vxB_z, axis=1)/dtheta - np.gradient(vxB_theta, axis=2)/dz
            dBtheta_dt = np.gradient(vxB_r, axis=2)/dz - np.gradient(vxB_z, axis=0)/dr
            dBz_dt = (1/R)*np.gradient(R*vxB_theta, axis=0)/dr - (1/R)*np.gradient(vxB_r, axis=1)/dtheta

            B_r += (dBr_dt - eta * np.gradient(np.gradient(B_r, axis=0), axis=0)/dr**2) * dt
            B_theta += (dBtheta_dt - eta * np.gradient(np.gradient(B_theta, axis=0), axis=0)/dr**2) * dt
            B_z += (dBz_dt - eta * np.gradient(np.gradient(B_z, axis=0), axis=0)/dr**2) * dt

            # Divergence cleaning
            divB = np.gradient(B_r, axis=0)/dr + B_r/R + np.gradient(B_theta, axis=1)/(R*dtheta) + np.gradient(B_z, axis=2)/dz
            B_r -= ch_divb * divB * dr
            B_theta -= ch_divb * divB * R
            B_z -= ch_divb * divB * dz

            # Continuity + Pair
            div_v = np.gradient(v_r, axis=0)/dr + v_r/R + np.gradient(v_theta, axis=1)/(R*dtheta) + np.gradient(v_z, axis=2)/dz
            rho += -rho * div_v * dt
            pair_rate = np.clip(5e-5 * (Bmag / 4.4e9)**2, 0, 2e-4)
            rho += 2 * m_e * pair_rate * dt

            # Momentum
            J_z = (1/(mu0*R)) * np.gradient(R*B_theta, axis=0)/dr
            JxB_r = -J_z * B_theta
            dp_dr = np.gradient(p, axis=0)/dr
            adv_r = v_r * np.gradient(v_r, axis=0)/dr + (v_theta/R)*np.gradient(v_r, axis=1)/dtheta - v_theta**2 / R

            rho_eff = (rho + p / c**2) * (GAMMA_BULK ** 2) + EPS
            v_r -= dt * (dp_dr - JxB_r + rho * adv_r) / rho_eff
            v_theta -= dt * 0.028 * J_z * B_r / rho_eff

            # Energy
            pdv = -p * div_v
            ohmic = eta * J_z**2
            e_int += (pdv + ohmic) * dt
            p = np.maximum((gamma_ad - 1) * e_int, 5e8)

            t += dt

            if step % save_every == 0:
                Pi = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)
                Pi_rel = Pi / (GAMMA_BULK ** 2)
                t_hist.append(t)

                vr_fft = np.fft.fft(v_r, axis=1)
                power_m = np.sum(np.abs(vr_fft)**2 * R[:,:,np.newaxis] * dr * dz, axis=(0,2))

                # BULLETPROOF extraction
                m0 = float(np.real(power_m[0]))
                m1 = float(np.real(power_m[1])) if len(power_m) > 1 else 0.0

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
        Pi_rel = Pi_final / (GAMMA_BULK ** 2)

        results['Pi'].append(Pi_final)
        results['gamma_m0'].append(gamma_m0)
        results['gamma_m1'].append(gamma_m1)
        results['J0'].append(J0)
        results['rho0'].append(rho0)

        print(f"Pi = {Pi_final:.2f} | Pi_rel = {Pi_rel:.2f} | γ_m0 = {gamma_m0:.2e} | γ_m1 = {gamma_m1:.2e}")

print(f"\nTotal runtime: {time.time()-start:.1f} s")

# Final Plot
plt.figure(figsize=(12, 8))
plt.scatter(results['Pi'], results['gamma_m1'], s=160, edgecolors='black', label='m=1 Kink', color='red')
plt.scatter(results['Pi'], results['gamma_m0'], s=140, edgecolors='black', label='m=0 Sausage', color='blue', marker='s')
plt.axvline(8.0, color='red', ls='--', lw=3, label='Π_crit ≈ 8')
plt.xlabel('Π = μ₀ J² R² / (ρ c²)', fontsize=14)
plt.ylabel('Linear Growth Rate γ (s⁻¹)', fontsize=14)
plt.title(f'Relativistic Jet (γ = {GAMMA_BULK}) - 128³')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
