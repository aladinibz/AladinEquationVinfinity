import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

f0=43.0; omega0=2*np.pi*f0; t=np.logspace(-1,3,200)

# Resonant absorption
rho_r=10.0; dL=0.1; c=(rho_r-1)/(rho_r+1)
gamma_ra=-(np.pi/2)*c*dL*omega0; A_ra=np.exp(gamma_ra*t)
A_ra/=A_ra[0]

# Phase mixing
dwr=0.4; A_pm=(t/t[0])**(-1/3)

# Nonlinear coupling (amplitude-dependent, fixed A_s=10 km)
A_s=10; coupling=0.005; delta_f_nl=coupling*A_s**2
A_nl=np.exp(-delta_f_nl*2*np.pi*t/f0)  # simple exp for illustration

plt.figure(figsize=(12,7),facecolor='black')
plt.plot(t,A_ra,color='gold',lw=3,label='Resonant Absorption (exp)')
plt.plot(t,A_pm,color='lime',lw=3,label='Phase Mixing (t^{-1/3})')
plt.plot(t,A_nl,color='cyan',lw=3,label='Nonlinear Coupling')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Time (s)',color='white')
plt.ylabel('Norm. Amplitude',color='white')
plt.title('MHD Damping Mechanisms Master',color='gold')
plt.legend(loc='upper right',framealpha=0.2)
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/mhd_damping_master.png',dpi=300,facecolor='black')
plt.close()
