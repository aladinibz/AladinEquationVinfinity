"""
ALADIN v0.9.3 — Fixed for Colab
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

%matplotlib inline

print("🚀 ALADIN v0.9.3 - Fixed Version Starting...\n")

# ========================= SETUP =========================
mu0 = 4 * np.pi * 1e-7
c = 3.0e8
m_e = 9.109e-31
E_c = 1.3e18
gamma = 5.0/3
EPS = 1e-12
CFL = 0.35

# Grid (small for speed)
nr = ntheta = nz = 48
r_max = 2.0
z_max = 6.0

dr = r_max / nr
dtheta = 2*np.pi / ntheta
dz = z_max / nz

r = np.linspace(dr/2, r_max, nr)
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
z = np.linspace(-z_max/2, z_max/2, nz)
R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')

# Scan
J0_list = [6e5, 9e5, 1.3e6, 2e6]
rho0_list = [8e11, 1.5e12, 3e12]

n_steps = 200
save_every = 10

results = {'Pi': [], 'gamma': [], 'stable': [], 'J0': [], 'rho0': []}

start = time.time()

for J0 in J0_list:
    for rho0 in rho0_list:
        rho = np.full_like(R, rho0, dtype=np.float32)
        p = np.full_like(R, 1e10, dtype=np.float32)
        v_r = np.zeros_like(R, dtype=np.float32)
        B_theta = (mu0 * J0 * R / 2) * np.exp(-(R/(0.6*r_max))**2)
        B_z = np.full_like(R, 3000.0, dtype=np.float32)
        n_pair = np.zeros_like(R, dtype=np.float32)

        v_r += 0.006 * np.sin(Theta) * np.exp(-((R-0.7)**2)/0.2)

        t_hist = []
        mode_hist = []
        t = 0.0

        for step in tqdm(range(n_steps), leave=False, desc=f"J0={J0:.1e}"):
            Bmag = np.sqrt(B_theta**2 + B_z**2 + EPS)
            vA = Bmag / np.sqrt(mu0 * rho + EPS)
            vfluid = np.sqrt(v_r**2 + EPS)
            
            max_speed = max(vA.max(), vfluid.max(), 1e4)
            dt = CFL * min(dr, dz) / max_speed
            dt = min(dt, 1e-6)
            t += dt

            # Physics
            J_z = (1 / (mu0 * R)) * np.gradient(R * B_theta, axis=0) / dr
            E_ind = np.abs(np.gradient(B_z, axis=2)) * R / 2.0
            pair_rate = 8e-5 * np.exp(-E_c / (np.abs(E_ind)+EPS)) * (Bmag/4.4e9)**2

            n_pair += pair_rate * dt
            rho += 2 * m_e * pair_rate * dt
            p += 4e8 * pair_rate * dt

            dp_dr = np.gradient(p, axis=0) / dr
            rho_eff = rho + p / c**2 + EPS

            v_r -= dt * (dp_dr - J_z * B_theta) / rho_eff

            # Diffusion
            B_theta += 1e-5 * dt * (np.gradient(np.gradient(B_theta, axis=0), axis=0) / dr**2)

            if step % save_every == 0:
                Pi = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)
                t_hist.append(t)

                mid = nz//2
                ft = np.abs(np.fft.fft(np.mean(v_r[:,:,mid], axis=0)))
                mode_hist.append(ft[1] if len(ft)>1 else 0.0)

        # Growth rate
        if len(mode_hist) > 15:
            gamma = np.polyfit(t_hist, np.log(np.array(mode_hist)+1e-10), 1)[0]
        else:
            gamma = 0.0

        Pi_final = mu0 * J0**2 * r_max**2 / (np.mean(rho) * c**2 + EPS)
        stable = (gamma < 5e-4) and (Pi_final > 7.0)

        results['Pi'].append(Pi_final)
        results['gamma'].append(gamma)
        results['stable'].append(stable)
        results['J0'].append(J0)
        results['rho0'].append(rho0)

        print(f"Pi = {Pi_final:.2f} | γ = {gamma:.2e} | Stable = {stable}")

print(f"\nFinished in {time.time()-start:.1f} seconds")

# Final Plot
plt.figure(figsize=(10,7))
colors = ['green' if s else 'red' for s in results['stable']]
plt.scatter(results['Pi'], results['gamma'], c=colors, s=100, edgecolors='black')
plt.axvline(8, color='red', ls='--', lw=2.5, label='Π_crit ≈ 8')
plt.xlabel('Π')
plt.ylabel('Growth Rate γ (s⁻¹)')
plt.title('ALADIN v0.9.3 — Stability Scan')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
