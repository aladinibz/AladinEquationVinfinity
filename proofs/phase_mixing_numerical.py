import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

a=1e6; L=100e6; k_z=np.pi/L; v_Ai=1000e3; rho_r=10.0
r=np.linspace(0,a,200); dr=r[1]-r[0]
trans=0.1*a
rho=(rho_r+1)/2-(rho_r-1)/2*np.tanh((r-0.8*a)/trans)
omega_A=k_z*v_Ai*np.sqrt(1/rho)
xi_0=np.where(r<0.9*a,1.0,0.0)

t=np.arange(0,2001,1)
xi_t=np.zeros((len(t),len(r)))
for i,ti in enumerate(t):
    xi_t[i]=xi_0*np.cos(omega_A*ti)

area=2*np.pi*r*dr
global_amp=np.abs(np.sum(xi_t*area[None,:],axis=1)/np.sum(area))
global_amp/=global_amp[0]

plt.figure(figsize=(10,6),facecolor='black')
plt.plot(t,global_amp,color='gold',lw=3)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Time (s)',color='white')
plt.ylabel('Norm. Amplitude',color='white')
plt.title('Phase Mixing Decay',color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/phase_mixing_numerical.png',dpi=300,facecolor='black')
plt.close()
