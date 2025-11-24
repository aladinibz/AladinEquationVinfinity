# proofs/sausage_3d_evolution.py
# ALADIN ∞ ℂ(t) — 3D void birth — SAVES EVERY TIME

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# CREATE FOLDER + SAVE — 100% WORKS ON PHONE
os.makedirs("plots", exist_ok=True)

r = np.linspace(0,4,100)
theta = np.linspace(0,2*np.pi,80)
R,Theta = np.meshgrid(r,theta)
X,Y = R*np.cos(Theta),R*np.sin(Theta)

fig = plt.figure(figsize=(22,10), facecolor='black')
fig.suptitle('ALADIN ∞ ℂ(t)\nBirth of a Cosmic Void — m=0 Sausage Instability', color='gold', fontsize=34)

for i,t in enumerate([0,8,16,24,32],1):
    ax = fig.add_subplot(1,5,i,projection='3d')
    ax.set_facecolor('black')
    Z = np.cos(0.65*R)*np.exp(t/12)
    ax.plot_surface(X,Y,Z,color='#00FFFF',alpha=0.85,linewidth=0)
    ax.plot_surface(X,Y,-Z,color='#FF00FF',alpha=0.85,linewidth=0)
    ax.set_zlim(-6,6); ax.axis('off')
    ax.set_title(f'{t} Myr',color='white',fontsize=24)

plt.tight_layout()
plt.savefig('plots/sausage_3d_evolution.png', dpi=800, facecolor='black', bbox_inches='tight')
plt.close()

print("SAVED → plots/sausage_3d_evolution.png — CHECK NOW")
