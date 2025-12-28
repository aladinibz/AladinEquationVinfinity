import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

rho_ratio=10.0; delta_over_L=0.1; f0=43.0
contrast=(rho_ratio-1)/(rho_ratio+1)
gamma_over_omega=-(np.pi/2)*contrast*delta_over_L
tau=1/(np.abs(gamma_over_omega)*2*np.pi*f0)

print(f"γ/ω₀ ≈ {gamma_over_omega:.3f}")
print(f"Damping time τ ≈ {tau:.3f} s")

t=np.linspace(0,100,200)
A=np.exp(gamma_over_omega*2*np.pi*f0*t)

plt.figure(figsize=(10,6),facecolor='black')
plt.plot(t,A,color='gold',lw=3)
plt.xlabel('Time (s)',color='white')
plt.ylabel('Norm. Amplitude',color='white')
plt.title('Resonant Absorption Decay',color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/resonant_absorption_damping.png',dpi=300,facecolor='black')
plt.close()
