import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

rho_r=10.0; dL=0.1; c=(rho_r-1)/(rho_r+1)
gamma_over_omega=-(np.pi/2)*c*dL

f=np.logspace(-2,2,200)  # 0.01 Hz to 100 Hz
omega=2*np.pi*f
gamma=gamma_over_omega*omega
tau=1/np.abs(gamma)

plt.figure(figsize=(10,6),facecolor='black')
plt.plot(f,tau,color='gold',lw=3)
plt.axvline(43,color='lime',ls='--',lw=2,label='43 Hz')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Frequency (Hz)',color='white')
plt.ylabel('Damping Time τ (s)',color='white')
plt.title('Kink Damping Time vs Frequency',color='gold')
plt.legend(loc='upper right',framealpha=0.2)
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/kink_damping_vs_frequency.png',dpi=300,facecolor='black')
plt.close()
