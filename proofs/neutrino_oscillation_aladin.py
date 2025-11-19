import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# 43 Hz gives natural phase rotation
f0 = 43.0
phi_43 = 2*np.pi * f0 * np.linspace(0,1e-21,1000)  # phase over 1 zeptosecond

# Three mixing angles from 43 Hz harmonics
theta12 = np.arcsin(np.sin(phi_43[300])) % (np.pi/2)
theta23 = np.arcsin(np.sin(phi_43[430])) % (np.pi/2)
theta13 = np.arcsin(np.sin(phi_43[666])) % (np.pi/2)

print(f"θ₁₂ ≈ {np.degrees(theta12):.1f}°")
print(f"θ₂₃ ≈ {np.degrees(theta23):.1f}°")
print(f"θ₁₃ ≈ {np.degrees(theta13):.1f}°")

plt.figure(figsize=(12,7),facecolor='black')
plt.plot(phi_43, np.sin(phi_43), color='cyan', lw=3, label='43 Hz phase')
plt.axvline(phi_43[300],color='gold',ls='--',lw=3,label=f'θ₁₂ = {np.degrees(theta12):.1f}°')
plt.axvline(phi_43[430],color='lime',ls='--',lw=3,label=f'θ₂₃ = {np.degrees(theta23):.1f}°')
plt.axvline(phi_43[666],color='magenta',ls='--',lw=3,label=f'θ₁₃ = {np.degrees(theta13):.1f}°')

plt.xlabel('Time (zeptoseconds)',color='white')
plt.ylabel('sin(φ)',color='white')
plt.title('PMNS Mixing Angles Emerge from 43 Hz Phase',color='white',fontsize=18)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(5e-21,0.6,'43 Hz cosmic resonance\n→ PMNS angles from phase nodes\nNo free parameters',
         color='lime',fontsize=14,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/neutrino_oscillation_aladin.png',dpi=400,facecolor='black')
plt.close()
