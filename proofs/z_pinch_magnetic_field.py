import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

r = np.logspace(-3,3,500)
J0 = 1e18
mu0 = 4*np.pi*1e-7
R = 1  # arbitrary scale

B_in = (mu0 * J0 * r)/2
B_out = (mu0 * J0 * np.pi * R**2)/(2*np.pi*r)

plt.figure(figsize=(11,7),facecolor='black')
plt.loglog(r[r<R],B_in[r<R],color='gold',lw=8,label='Inside pinch')
plt.loglog(r[r>=R],B_out[r>=R],color='gold',lw=8)
plt.axvline(R,color='white',ls='--',lw=4)

plt.xlabel('Radius r',color='white')
plt.ylabel('B-field (T)',color='white')
plt.title('Z-Pinch Magnetic Field B(r) = μ₀ J₀ r / 2',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3); plt.tick_params(colors='white')
plt.text(0.1,1e-10,'B(r) ∝ r inside\n→ v_rot = constant\n→ Flat rotation curves',
         color='lime',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))
plt.tight_layout()
plt.savefig('plots/z_pinch_magnetic_field.png',dpi=400,facecolor='black')
plt.close()
