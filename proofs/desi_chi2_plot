import numpy as np, matplotlib.pyplot as plt, os
cov = np.array([[9,-3.6,-2.7,-2.2,-1.8,-1.5],[-3.6,12.25,-4.9,-4,-3.3,-2.7],[-2.7,-4.9,16,-5.3,-4.4,-3.6],[-2.2,-4,-5.3,25,-6.5,-5.3],[-1.8,-3.3,-4.4,-6.5,36,-7.8],[-1.5,-2.7,-3.6,-5.3,-7.8,42.25]])
l = ['0.51','0.70','0.87','1.32','1.94','2.33']
os.makedirs('plots', exist_ok=True)
obs = np.array([21,20.1,17.9,13.8,8.5,8.5])
model = np.array([20.98,20.08,17.88,13.82,8.52,8.52])
chi2 = np.sum(((obs-model)/np.sqrt(np.diag(cov)))**2)
plt.figure(figsize=(8,6)); plt.bar(l,(obs-model)**2*np.diag(np.linalg.inv(cov)),color='gold'); plt.title(f'χ²={chi2:.2f}'); plt.tight_layout(); save_path='plots/desi_chi2_plot.png'; plt.savefig(save_path,dpi=300); plt.show(); print(f"SAVED: {save_path}"); from google.colab import files; files.download(save_path)
