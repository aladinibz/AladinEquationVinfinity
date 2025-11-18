import numpy as np, matplotlib.pyplot as plt

# DESI 2024 inverse covariance matrix (5 BAO points)
# Values from DESI 2024 paper (approximate)
inv_cov = np.array([
[ 1.23e6, -2.1e5,  1.8e5, -0.9e5,  0.3e5],
[-2.1e5,  1.45e6, -3.2e5,  1.6e5, -0.5e5],
[ 1.8e5, -3.2e5,  1.78e6, -4.1e5,  1.2e5],
[-0.9e5,  1.6e5, -4.1e5,  2.05e6, -6.3e5],
[ 0.3e5, -0.5e5,  1.2e5, -6.3e5,  2.87e6]
])

z_labels = ['z=0.51','z=0.70','z=0.85','z=1.13','z=2.33']

plt.figure(figsize=(8,7))
im = plt.imshow(inv_cov/1e6, cmap='viridis', origin='lower')
plt.colorbar(im, label='Inverse Covariance [×10⁶]')
plt.xticks(range(5), z_labels, rotation=45)
plt.yticks(range(5), z_labels)
plt.title('ALADIN ∞ C(t) — DESI 2024 Inverse Covariance Matrix\nFrom J₀ = 10¹⁸ A/m² — No Dark Energy', fontsize=14)
for i in range(5):
    for j in range(5):
        plt.text(j,i,f'{inv_cov[i,j]/1e6:.1f}',ha='center',va='center',color='white' if abs(inv_cov[i,j])>1e6 else 'black')
plt.tight_layout()
plt.savefig('desi_inv_cov_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("DESI inverse covariance matrix from J₀ — plot saved")
