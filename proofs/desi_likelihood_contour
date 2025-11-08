import numpy as np, matplotlib.pyplot as plt, os
cov = np.array([[9,-3.6,-2.7,-2.2,-1.8,-1.5],[-3.6,12.25,-4.9,-4,-3.3,-2.7],[-2.7,-4.9,16,-5.3,-4.4,-3.6],[-2.2,-4,-5.3,25,-6.5,-5.3],[-1.8,-3.3,-4.4,-6.5,36,-7.8],[-1.5,-2.7,-3.6,-5.3,-7.8,42.25]])
os.makedirs('plots', exist_ok=True)
obs = np.array([21,20.1,17.9,13.8,8.5,8.5])
z = np.array([0.51,0.7,0.87,1.32,1.94,2.33])
H0 = np.linspace(70,85,100)
n = np.linspace(0.1,0.5,100)
H0m, nm = np.meshgrid(H0, n)
chi2g = np.zeros_like(H0m)
for i in range(len(n)):
    for j in range(len(H0)):
        model = H0[j] * (z / (1 + z))**n[i]
        chi2g[i,j] = np.sum(((obs - model) / np.sqrt(np.diag(cov)))**2)
plt.figure(figsize=(8,6))
plt.contour(H0m, nm, chi2g, levels=[2.3,6.18,11.83], colors=['blue','green','red'])
plt.scatter(75.2,0.25,color='gold',s=100)
plt.xlabel('H0')
plt.ylabel('n')
plt.title('Likelihood')
plt.tight_layout()
save_path = 'plots/desi_likelihood_contour.png'
plt.savefig(save_path, dpi=300)
plt.show()
print(f"SAVED: {save_path}")
from google.colab import files
files.download(save_path)
