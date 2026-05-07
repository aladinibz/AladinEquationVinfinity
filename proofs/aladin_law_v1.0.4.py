"""
ALADIN v1.0.4 — Better Divergence Control + Proper Mode Extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

%matplotlib inline

print("🚀 ALADIN v1.0.4 — Improved Divergence + Mode Analysis\n")

# ========================= GRID =========================
GRID_SIZE = 96          # Change to 128 if your session survives
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
EPS = 1e-12
CFL = 0.24
eta = 1.6e-4
ch_divb = 1.2          # stronger cleaning speed

J0_list = [8e5, 1.3e6, 2.1e6]
rho0_list = [1.1e12, 2.4e12]

n_steps = 140
save_every = 8

results = {'Pi': [], 'gamma_m1': [], 'J0': [], 'rho0': []}

start = time.time()

for J0 in J0_list:
    for rho0 in rho0_list:
        rho = np.full_like(R, rho0, dtype=np.float32)
        p = np.full_like(R, 1e10, dtype=np.float32)
        e_int = p / (gamma - 1)

        v_r = np.zeros_like(R, dtype=np.float32)
        v_theta = np.zeros_like(R, dtype=np.float32)
        v_z = np.zeros_like(R, dtype=np.float32)

        B_r = np.zeros_like(R, dtype=np.float32)
        B_theta = (mu0 * J0 * R / 2).astype(np.float32) * np.exp(-(R/(0.6*r_max))**2)
        B_z = np.full_like(R, 2800.0, dtype=np.float32)

        v_r += 0.005 * np.sin(Theta) * np.exp(-((R-0.75)**2)/0.22)

        t_hist = []
        mode_m1_hist = []
        t = 0.0

        for step in tqdm(range(n_steps), leave=False, desc=f"J0={J0:.1e}"):
            Bmag = np.sqrt(B_r**2 + B_theta**2 + B_z**2 + EPS)
            vA = Bmag / np.sqrt(mu0 * rho + EPS)
            vfluid = np.sqrt(v_r**2 + v_theta**2 + v_z**2 + EPS)
            max_speed = max(float(vA.max()), float(vfluid.max()), 1e4)
            dt = CFL * min(dr, dz) / max_speed
            dt = min(dt, 8e-7)

            # Induction
            vxB_r = v_theta * B_z - v_z * B_theta
            vxB_theta = v_z * B_r - v_r * B_z
            vxB_z = v_r * B_theta - v_theta * B_r

            dBr_dt = (1/R)*np.gradient(R*vxB_z, axis=1)/dtheta - np.gradient(vxB_theta, axis=2)/dz
            dBtheta_dt = np.gradient(vxB_r, axis=2)/dz - np.gradient(vxB_z, axis=0)/dr
            dBz_dt = (1/R)*np.gradient(R*vxB_theta, axis=0)/dr - (1/R)*np.gradient(vxB_r, axis=1)/dtheta

            B_r += (dBr_dt - eta * np.gradient(np.gradient(B_r, axis=0), axis=0)/dr**2) * dt
            B_theta += (dBtheta_dt - eta * np.gradient(np.gradient(B_theta, axis=0), axis=0)/dr**2) * dt
            B_z += (dBz_dt - eta * np.gradient(np.gradient(B_z, axis=0), axis=0)/dr**2) * dt

            # Strong hyperbolic cleaning
            divB = np.gradient(B_r, axis=0)/dr + B_r/R + np.gradient(B_theta, axis=1)/(R*dtheta) + np.gradient(B_z, axis=2)/dz
            B_r -= ch_divb * divB * dr
            B_theta -= ch_divb * divB * R
            B_z -= ch_divb * divB * dz

            # Continuity + Pair
            div_v = np.gradient(v_r, axis=0)/dr + v_r/R + np.gradient(v_theta, axis=1)/(R*dtheta) + np.gradient(v_z, axis=2)/dz
            rho += -rho * div_v * dt
            pair_rate = 6e-5 * (Bmag / 4.4e9)**2
            rho += 2 * m_e * pair_rate * dt

            # Momentum
            J_z = (1/(mu0*R)) * np.gradient(R*B_theta, axis=0)/dr
            JxB_r = -J_z * B_theta
            dp_dr = np.gradient(p, axis=0)/dr
            adv_r = v_r * np.gradient(v_r, axis=0)/dr + (v_theta/R)*np.gradient(v_r, axis=1)/dtheta - v_theta**2 / R

            rho_eff = rho + p / c**2 + EPS
            v_r -= dt * (dp_dr - JxB_r + rho * adv_r) / rho_eff
            v_theta -= dt * 0.035 * J_z * B_r / rho_eff

            # Energy
            pdv = -p * div_v
            ohmic = eta * J_z**2
            e_int += (pdv + ohmic) * dt
            p = np.maximum((gamma - 1) * e_int, 1e8)

            t += dt

            if step % save_every == 0:
                Pi = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)
                t_hist.append(t)

                # Better mode extraction
                mid = nz//2
                slice_vr = v_r[:,:,mid]
                ft = np.abs(np.fft.fft(np.mean(slice_vr, axis=0)))
                mode_m1 = float(ft[1]) if len(ft) > 1 else 0.0
                mode_m1_hist.append(mode_m1)

        gamma_fit = np.polyfit(np.array(t_hist), np.log(np.array(mode_m1_hist)+1e-10), 1)[0] if len(mode_m1_hist) > 15 else 0.0
        Pi_final = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)

        results['Pi'].append(Pi_final)
        results['gamma'].append(gamma_fit)
        results['J0'].append(J0)
        results['rho0'].append(rho0)

        print(f"Pi = {Pi_final:.2f} | γ_m1 = {gamma_fit:.2e}")

print(f"\nTotal runtime: {time.time()-start:.1f} s")

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(results['Pi'], results['gamma'], s=160, edgecolors='black', c=results['Pi'], cmap='RdYlGn')
plt.axvline(8.0, color='red', ls='--', lw=3, label='Proposed Π_crit ≈ 8')
plt.xlabel('Π = μ₀ J² R² / (ρ c²)', fontsize=14)
plt.ylabel('m=1 Kink Growth Rate γ (s⁻¹)', fontsize=14)
plt.title('ALADIN v1.0.4 — 96³ Grid (Higher Resolution Test)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
