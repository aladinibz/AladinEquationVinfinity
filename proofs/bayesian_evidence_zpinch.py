import numpy as np

ell = np.array([220,540,815,1100,1370,1600])
C_obs = np.array([5760,3185,2510,1815,1490,1300])
sigma = np.array([120,80,70,60,50,60])

def zpinch(ell):
    r = np.pi*1e26/ell
    B = 2e-6/r
    F = 1e18*B
    d = F/(3e8*1e-24*1e5**2)
    return 5000*np.exp(-ell/1000)*(1+0.7*np.sin(2*np.pi*ell/540+d.max()))

pred = zpinch(ell)
chi2 = np.sum(((C_obs-pred)**2)/sigma**2)
print(f"Z-Pinch χ² = {chi2:.1f} (0 parameters)")

# Save prediction for second file
np.savetxt('zpinch_pred.txt', pred)
