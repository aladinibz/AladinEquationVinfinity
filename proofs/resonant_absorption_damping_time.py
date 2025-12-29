import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

rho_r=10.0; dL=0.1; c=(rho_r-1)/(rho_r+1)
gamma_over_omega=-(np.pi/2)*c*dL

f=np.logspace(-2,2,200)
tau=1/(np.abs(gamma_over_omega)*2*np.pi*f)

plt.figure(figsize=(10,6),facecolor='black')
plt.plot(f,tau,color='gold',lw=3)
plt.axvline(43,color='lime',ls='--',label='43 Hz')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Frequency (Hz)',color='white')
plt.ylabel('Damping Time τ (s)',color='white')
plt.title('Resonant Absorption Damping Time vs Freq',color='gold')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/resonant_absorption_damping_time.png',dpi=300,facecolor='black')
plt.close()
