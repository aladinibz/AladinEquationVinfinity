import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

L=100e6; k_z=np.pi/L; v_Ai=1000e3; rho_r=10.0
omega_k = k_z * v_Ai * np.sqrt((rho_r + 1/rho_r)/(rho_r + 1))

print(f"Fundamental kink freq ≈ {omega_k/(2*np.pi):.3f} Hz")

# Plot vs contrast
rho_r_vals=np.logspace(0,3,100)
omega_k_vals=k_z*v_Ai*np.sqrt((rho_r_vals + 1/rho_r_vals)/(rho_r_vals + 1))

plt.figure(figsize=(10,6),facecolor='black')
plt.semilogx(rho_r_vals,omega_k_vals/(2*np.pi),color='gold',lw=3)
plt.axhline(43,color='lime',ls='--',label='43 Hz')
plt.xlabel('Density Ratio ρ_i/ρ_e',color='white')
plt.ylabel('Kink Frequency (Hz)',color='white')
plt.title('Kink Mode Frequency vs Density Contrast',color='gold')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/kink_mode_frequency.png',dpi=300,facecolor='black')
plt.close()
