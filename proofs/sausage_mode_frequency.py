import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import jv, kv

os.makedirs("plots", exist_ok=True)

L=100e6; k_z=np.pi/L; v_Ai=1000e3; rho_r=10.0
a=1e6; ka=k_z*a

omega_s = k_z*v_Ai*np.sqrt(1 + (1/rho_r)*kv(1,ka)/kv(0,ka))

print(f"Fundamental sausage freq ≈ {omega_s/(2*np.pi):.3f} Hz")

# Plot vs contrast
rho_r_vals=np.logspace(0,3,100)
omega_s_vals=k_z*v_Ai*np.sqrt(1 + (1/rho_r_vals)*kv(1,ka)/kv(0,ka))

plt.figure(figsize=(10,6),facecolor='black')
plt.semilogx(rho_r_vals,omega_s_vals/(2*np.pi),color='gold',lw=3)
plt.axhline(43,color='lime',ls='--',label='43 Hz')
plt.xlabel('Density Ratio ρ_i/ρ_e',color='white')
plt.ylabel('Sausage Frequency (Hz)',color='white')
plt.title('Sausage Mode Frequency vs Density Contrast',color='gold')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/sausage_mode_frequency.png',dpi=300,facecolor='black')
plt.close()
