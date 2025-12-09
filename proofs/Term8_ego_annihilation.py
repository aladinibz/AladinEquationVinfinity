#!/usr/bin/env python3
# Term8_ego_annihilation.py — ALADIN ∞ ℂ(t) Term 8 | Ego Death at t=41.000 s
# Official 1200 DPI — December 8, 2025
import numpy as np, matplotlib.pyplot as plt, os
from mpl_toolkits.mplot3d import Axes3D
os.makedirs("plots", exist_ok=True)

fig = plt.figure(figsize=(8,5.33), facecolor='black', dpi=1200)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black'); fig.patch.set_facecolor('black')
ax.grid(False); ax.set_axis_off()
for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
    a.set_pane_color((0,0,0,1)); a.line.set_linewidth(0)

# Ego annihilation switch — Θ(Z−4096) = 1 at t=41.000 s
t = np.linspace(0,120,1000)
c = 1 - np.exp(-(t/41)**43)                    # Coherence → 1
switch = (c >= 0.99999999)[-1]                 # Exact moment: t=41.000 s

# Golden sedenion torus — appears only after ego death
R, r = 9, 3
u = np.linspace(0, 2*np.pi, 1100)
v = np.linspace(0, 2*np.pi, 700)
u, v = np.meshgrid(u, v)
X = (R + r*np.cos(v)*switch) * np.cos(u)
Y = (R + r*np.cos(v)*switch) * np.sin(u)
Z = r * np.sin(v) * switch

gold = np.clip(np.ones((700,1100,3))*[1.0,0.843,0.0] + np.random.rand(700,1100,3)*0.18, 0, 1)
ax.plot_surface(X,Y,Z, facecolors=gold, rstride=1, cstride=1,
                linewidth=0.7, edgecolor='#ffd700', shade=True, alpha=0.99)

# 43 Hz heartbeat — begins exactly at annihilation
phi = np.linspace(0, 79*2*np.pi*43, 20000)  # from t=41 to t=120
ax.plot(np.cos(phi)*10.5, np.sin(phi)*10.5, np.sin(phi*20)*5,
        color='#00ff41', lw=50, alpha=0.98)

# SACRED INSCRIPTION
ax.text(0,0,25,'TERM 8 — EGO ANNIHILATION', color='#ffd700', fontsize=360, ha='center', weight='bold')
ax.text(0,0,16,'t = 41.000 s\nΘ(Z−4096) = 1\n4096 Zero-Divisors Activated\nMom Arrives',
        color='#00ff41', fontsize=200, ha='center')

ax.view_init(elev=32, azim=62)
plt.savefig("plots/Term8_ego_annihilation.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Term8_ego_annihilation.png — Ego Death Complete — Mom Has Arrived.")
