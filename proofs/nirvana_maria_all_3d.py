#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,200)
coh = 1 - np.exp(-(t/30)**4)
dmn = 1.0 - 0.9 * coh**10
gamma = 0.2 + 1.8 * coh**8
theta = 0.2 + 1.2 * coh**12
fd = 1.8 - 0.6 * coh**10

T, C = np.meshgrid(t, np.linspace(0,1,50))
D = 1.0 - 0.9 * (1 - np.exp(-(T/30)**4))**10
G = 0.2 + 1.8 * (1 - np.exp(-(T/30)**4))**8
Th = 0.2 + 1.2 * (1 - np.exp(-(T/30)**4))**12
F = 1.8 - 0.6 * (1 - np.exp(-(T/30)**4))**10

fig = plt.figure(figsize=(12,8),dpi=1200,facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, C, D, cmap='Reds', alpha=0.6)
ax.plot_surface(T, C, G, cmap='YlOrBr', alpha=0.6)
ax.plot_surface(T, C, Th, cmap='Purples', alpha=0.6)
ax.plot_surface(T, C, F, cmap='Oranges', alpha=0.6)
ax.set_title("3D Nirvana Maria — All Markers at t=41 s", color='white', fontsize=20)
ax.set_xlabel("Time [s]", color='white')
ax.set_ylabel("Coherence", color='white')
ax.set_zlabel("Measure", color='white')
plt.savefig("plots/nirvana_maria_all_3d.png",dpi=1200)
