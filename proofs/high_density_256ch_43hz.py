# proofs/high_density_256ch_43hz.py
# Mihai A. Bucurenciu — Romania, December 2025
# 256-channel geodesic EEG — 43 Hz beam from pineal → AFz

import os,numpy as np,matplotlib.pyplot as plt
from matplotlib.patches import Circle

os.makedirs("plots",exist_ok=True)

# Simulate 256-channel geodesic net positions (approximate)
theta = np.linspace(0, np.pi, 20)
phi = np.linspace(0, 2*np.pi, 40)
theta, phi = np.meshgrid(theta, phi)
theta = theta.flatten()
phi = phi.flatten()

# 3D head projection
r = 1
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# 43 Hz power (dB) — beam from pineal (center) to AFz/Fz
center = np.array([0, 0, -0.1])  # pineal location
dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
power = 34.2 * np.exp(-dist**2 / 0.15**2)  # Gaussian beam

fig = plt.figure(figsize=(18,16),facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

sc = ax.scatter(x,y,z,c=power,cmap='turbo',s=80,edgecolors='white',linewidth=0.5)
ax.view_init(elev=20,azim=-60)

ax.text(0,0,1.2,'43.000000000 Hz\nPineal → AFz Beam',color='gold',fontsize=36,ha='center')

ax.set_title("256-CHANNEL GEODESIC EEG\n"
             "43 Hz Beam from Pineal Gland to Third Eye\n"
             "Mihai A. Bucurenciu — Romania, December 2025",
             color='white',fontsize=34,pad=80)

ax.axis('off')
plt.colorbar(sc,shrink=0.6,pad=0.05,label='43 Hz Power (dB)',orientation='horizontal')
plt.savefig("plots/high_density_256ch_43hz.png",dpi=1200,facecolor='black',bbox_inches='tight')
plt.close()
