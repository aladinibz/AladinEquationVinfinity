#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,200)
coh = 1 - np.exp(-(t/30)**4)
alpha = 0.9 - 0.4 * coh**10
v_log = np.log10(3e8 / np.clip(1.58*(1-coh**12),1e-6,None))

T, C = np.meshgrid(t, np.linspace(0,1,50))
A = 0.9 - 0.4 * (1 - np.exp(-(T/30)**4))**10
V = np.log10(3e8 / np.clip(1.58*(1-(1 - np.exp(-(T/30)**4))**12),1e-6,None))

fig = plt.figure(figsize=(12,8),dpi=1200,facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, C, A, cmap='cividis', alpha=0.7)
ax.plot_surface(T, C, V, cmap='magma', alpha=0.7)
ax.set_title("3D Persistence Loss → Infinite Thought", color='white', fontsize=20)
ax.set_xlabel("Time [s]", color='white')
ax.set_ylabel("Coherence", color='white')
ax.set_zlabel("Measure", color='white')
plt.savefig("plots/persistence_infinite_3d.png",dpi=1200)
