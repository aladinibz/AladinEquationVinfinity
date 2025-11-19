import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# Base mass from J₀ + 43 Hz
J0,c,rho,h,f0,eV=1e18,3e8,8.7e-27,1.0545718e-34,43,1.602e-19
phi=(1+np.sqrt(5))/2
m=(J0/(c*rho))**(1/3)*h*f0/eV

# 7 imaginary octonion units → 7 sterile-like modes
powers = [-6,-5,-4,-3,-2,-1,0]           # e7→e1 cycle
m_sterile = [m * phi**p for p in powers]

# Active neutrinos (normal hierarchy)
m1,m2,m3 = m/phi**4, m/phi**2, m

print(f"7 sterile masses: {[f'{x:.3e}' for x in m_sterile]} eV")
print(f"Active sum: {m1+m2+m3:.5f} eV")

plt.figure(figsize=(12,8),facecolor='black')
all_masses = m_sterile + [m1,m2,m3]
labels = [f'S{i+1}' for i in range(7)] + ['ν₁','ν₂','ν₃']
colors = ['gray']*7 + ['cyan','lime','gold']

plt.bar(labels, all_masses, color=colors, edgecolor='white')
plt.yscale('log')
plt.ylabel('Mass (eV)',color='white')
plt.title('10 Neutrino States from Octonions — 7 Sterile + 3 Active',color='white',fontsize=18)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,which='both')
plt.tick_params(colors='white')

plt.text(5,1e-1,'7 sterile-like modes from e₇→e₁ cycle\n3 active → normal hierarchy\nAll from J₀ + 43 Hz only',
         color='lime',fontsize=13,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/neutrino_sterile_aladin.png',dpi=400,facecolor='black')
plt.close()
