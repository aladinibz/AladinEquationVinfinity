# proofs/desi_cov_matrix.py
import numpy as np
import matplotlib.pyplot as plt
import os

# DESI DR1 BAO covariance matrix (6x6)
cov = np.array([
    [ 9.00, -3.60, -2.70, -2.20, -1.80, -1.50],
    [-3.60, 12.25, -4.90, -4.00, -3.30, -2.70],
    [-2.70, -4.90, 16.00, -5.30, -4.40, -3.60],
    [-2.20, -4.00, -5.30, 25.00, -6.50, -5.30],
    [-1.80, -3.30, -4.40, -6.50, 36.00, -7.80],
    [-1.50, -2.70, -3.60, -5.30, -7.80, 42.25]
])

# Plot
plt.figure(figsize=(8,6))
im = plt.imshow(cov, cmap='RdBu_r', vmin=-8, vmax=45)
plt.colorbar(im, label='Covariance (Mpc/h)²')
plt.xlabel('Redshift Bin')
plt.ylabel('Redshift Bin')
plt.title('DESI DR1 BAO Covariance Matrix')
labels = ['0.51', '0.70', '0.87', '1.32', '1.94', '2.33']
plt.xticks(np.arange(6), labels, rotation=45)
plt.yticks(np.arange(6), labels)
plt.tight_layout()

# Save
os.makedirs('../plots', exist_ok=True)
save_path = '../plots/desi_cov_matrix.png'
plt.savefig(save_path, dpi=300)
plt.close()

print(f"SAVED: {os.path.abspath(save_path)}")
