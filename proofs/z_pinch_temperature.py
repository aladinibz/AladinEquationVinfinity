import numpy as np,matplotlib.pyplot as plt
n,r=500,np.linspace(.05,5,500);dr=r[1]-r[0];I=5e6;mu0=4*np.pi*1e-7;n0=1e19;T0=1e6;kB=1.38e-23;m_i=1.67e-27
n_e,v_r,T=n0*np.ones(n),np.zeros(n),T0*np.ones(n);dt,steps=5e-10,800
for _ in range(steps):
 B=mu0*np.clip(np.cumsum(n_e*2*np.pi*r*dr)*1.6e-19,0,I)/(2*np.pi*r);B=np.nan_to_num(B)
 F=-n_e*B**2/(mu0*(r+1e-10));P=n_e*kB*T;dP=np.gradient(P,dr);dP[0]=dP[1]
 a=(F+dP)/(n_e*m_i+1e-20);v_r+=a*dt;v_r=np.clip(v_r,-1e6,1e6)
 div=np.gradient(v_r,dr);div[0]=div[1];n_e+=-dt*(n_e*div+v_r*np.gradient(n_e,dr));n_e=np.maximum(n_e,1e16)
 dT=(5/3-1)*T*div+(B**2/mu0)*1e-3/(n_e*kB+1e-20);T+=dT*dt;T=np.maximum(T,1e6)
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))
ax1.loglog(r,n_e/n0,'purple',lw=3);ax1.loglog(r,np.ones_like(r),'--',color='gray');ax1.set_ylabel('Density / n₀')
ax2.loglog(r,T/1e6,'red',lw=3);ax2.set_xlabel('Radius (cm)');ax2.set_ylabel('T (MK)')
print(f"Max density: {np.max(n_e)/n0:.1e}x, Max T: {np.max(T)/1e6:.1f} MK")
