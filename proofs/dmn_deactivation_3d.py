#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,200)
coh = 1 - np.exp(-(t/30)**4)
dmn = 1.0 - 0.9 * coh**10

T, C = np.meshgrid(t, np.linspace(0,1,50))
D = 1.0 - 0.9 * (1 - np.exp(-(T/30)**4))**10

fig = plt.figure(figsize=(12,8),dpi=1200)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, C, D, cmap='plasma', alpha=0.8)
ax.plot(t, np.ones_like(t), dmn, 'gold', lw=5, label='DMN Trace')
ax.set_title("3D DMN Deactivation — Ego Shutdown at t=41 s", fontsize=18)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Coherence"); ax.set_zlabel("DMN Power")
plt.savefig("plots/dmn_deactivation_3d.png",dpi=1200)
