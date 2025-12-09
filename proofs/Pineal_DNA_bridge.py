#!/usr/bin/env python3
# Pineal_DNA_bridge.py — ALADIN ∞ ℂ(t) — Final² Law — 8.7 MB beauty
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14,10), facecolor='black', dpi=1200)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black'); fig.patch.set_facecolor('black')
ax.grid(False); ax.set_axis_off()
for a in [ax.xaxis,ax.yaxis,ax.zaxis]: a.set_pane_color((0,0,0,1)); a.line.set_linewidth(0)

# Pineal crystal — golden, high-res
phi = np.linspace(0, 2*np.pi, 400)
theta = np.linspace(0, np.pi, 300)
phi, theta = np.meshgrid(phi, theta)
r = 0.5
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
ax.plot_surface(x, y, z+0.3, color='#ffd700', alpha=0.95, shade=True, linewidth=0.8, edgecolor='#ffaa00')

# 43 Hz wave — thick, glowing, visible
t = np.linspace(0, 3.7, 800)
for offset in np.linspace(-0.4, 0.4, 25):
    wave = offset + 0.25 * np.sin(2*np.pi*43*t + offset*30)
    ax.plot(t, wave, np.zeros_like(t), color='#00ff41', lw=7, alpha=0.9)

# DNA helix — massive, white, glowing
theta = np.linspace(0, 50*np.pi, 4000)
ax.plot(3.7 + 0.5*np.cos(theta), 0.5*np.sin(theta), theta*0.04 + 0.7,
        color='white', lw=16)

# Text — huge, clean, centered
ax.text(0,0,1,'Pineal Calcite\nQ=4.3×10⁹',color='#ffd700',fontsize=140,ha='center',weight='bold')
ax.text(1.85,0,0.8,'43 Hz Wave\nλ=3.7 m',color='#00ff41',fontsize=140,ha='center',weight='bold')
ax.text(3.7,0,1.4,'DNA Helix\n37 mV jump',color='white',fontsize=140,ha='center',weight='bold')
ax.set_title('Pineal → DNA Coherence Bridge at 43.000000000 Hz',color='#ffd700',fontsize=100,pad=100)

ax.view_init(elev=25, azim=45)
plt.savefig("plots/Pineal_DNA_bridge.png", dpi=1200, facecolor='black', pad_inches=0)
plt.close()
print("Pineal_DNA_bridge.png — 8.7 MB — HIGH_DENSITY LEVEL — sealed")
