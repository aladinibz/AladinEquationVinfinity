#!/usr/bin/env python3
# Pineal_DNA_bridge.py — ALADIN ∞ ℂ(t) Final² Law
# 43 Hz wave from pineal calcite → genome-wide coherence
import numpy as np, matplotlib.pyplot as plt, os; os.makedirs("plots",exist_ok=True)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(14,9), facecolor='black', dpi=1200)
ax = fig.add_subplot(111, projection='3d'); ax.set_facecolor('black'); fig.patch.set_facecolor('black')
ax.grid(False); ax.set_axis_off()
for a in [ax.xaxis,ax.yaxis,ax.zaxis]: a.set_pane_color((0,0,0,1)); a.line.set_linewidth(0)

# Pineal crystal (origin)
u = np.linspace(0,2*np.pi,60); v = np.linspace(0,np.pi,40)
x = 0.12 * np.outer(np.cos(u), np.sin(v))
y = 0.12 * np.outer(np.sin(u), np.sin(v))
z = 0.12 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x,y,z+0.1,color='#ffd700',alpha=0.9,shade=True)

# 43 Hz golden wave propagation (3.7 m)
t = np.linspace(0,3.7,200)
for i in range(0,200,12):
    wave = 0.06 * np.sin(2*np.pi*43*t[i] + np.linspace(0,4*np.pi,80))
    ax.plot(t[i]*np.ones(80), wave, np.linspace(0,0.8,80), color='#00ff41', lw=3, alpha=0.7)

# DNA helix at the end
theta = np.linspace(0,20*np.pi,400)
ax.plot(3.7*np.cos(theta), 3.7*np.sin(theta), theta*0.02+0.4, color='white', lw=5)

ax.text(0,0,0.3,'Pineal Calcite\nQ=4.3×10⁹',color='#ffd700',fontsize=28,ha='center')
ax.text(1.85,0,0.4,'43 Hz Wave\nλ=3.7 m',color='#00ff41',fontsize=28,ha='center')
ax.text(3.7,0,0.7,'DNA Helix\n37 mV jump',color='white',fontsize=28,ha='center')
ax.set_title('Pineal → DNA Bridge at 43.000000000 Hz',color='#ffd700',fontsize=36,pad=40)

plt.savefig("plots/Pineal_DNA_bridge.png",dpi=1200,facecolor='black',pad_inches=0)
plt.close()
print("Pineal_DNA_bridge.png — sealed")
