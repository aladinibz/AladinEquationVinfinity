import numpy as np, matplotlib.pyplot as plt, os
cov = np.array([[9,-3.6,-2.7,-2.2,-1.8,-1.5],[-3.6,12.25,-4.9,-4,-3.3,-2.7],[-2.7,-4.9,16,-5.3,-4.4,-3.6],[-2.2,-4,-5.3,25,-6.5,-5.3],[-1.8,-3.3,-4.4,-6.5,36,-7.8],[-1.5,-2.7,-3.6,-5.3,-7.8,42.25]])
l = ['0.51','0.70','0.87','1.32','1.94','2.33']
os.makedirs('plots', exist_ok=True)
corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
plt.figure(figsize=(8,6)); plt.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1); plt.colorbar(); plt.title('Correlation'); plt.xticks(range(6),l,rotation=45); plt.yticks(range(6),l); plt.tight_layout(); save_path='plots/desi_corr_matrix.png'; plt.savefig(save_path,dpi=300); plt.show(); print(f"SAVED: {save_path}"); from google.colab import files; files.download(save_path)
