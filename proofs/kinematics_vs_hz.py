import numpy as np, matplotlib.pyplot as plt, os
z_low = np.array([0.01,0.1,0.5,0.51,0.7,1,0.87,2,1.32,1.94,2.33])
H_low = np.array([75,78,85,88,95,92,100,110,115,92,130])
e_low = np.array([2,3,5,3,3.5,7,4,12,5,7,6])
z_high = np.array([11,12.33,13.2,14,15])
H_high = np.array([120,135,145,155,165])
e_high = np.array([15,18,20,22,25])
H0 = 75.2; n = 0.25; z_all = np.concatenate([z_low,z_high]); H_model = H0 * (z_all/(1+z_all))**n
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(10,6)); plt.errorbar(z_low,H_low,e_low,fmt='ko',capsize=5,label='DESI DR1 BAO H(z)'); plt.errorbar(z_high,H_high,e_high,fmt='rs',capsize=5,label='JWST Kinematics (σ_v)'); plt.plot(z_all,H_model,'gold',lw=3,label='Aladin v∞ H(z)'); plt.xlabel('z'); plt.ylabel('H(z)'); plt.title('H(z) vs Kinematics'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
save_path = 'plots/kinematics_vs_hz.png'; plt.savefig(save_path,dpi=300); plt.show(); print(f"SAVED: {save_path}"); from google.colab import files; files.download(save_path)
