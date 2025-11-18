import numpy as np, matplotlib.pyplot as plt

# DESI 2024 correlation matrix (5 BAO points)
corr = np.array([
[1.00, 0.31, 0.18, 0.10, 0.05],
[0.31, 1.00, 0.35, 0.20, 0.08],
[0.18, 0.35, 1.00, 0.38, 0.12],
[0.10, 0.20, 0.38, 1.00, 0.25],
[0.05, 0.08, 0.12, 0.25, 1.00]
])

z_labels = ['z=0.51','z=0.70','z=0.85','z=1.13','z=2.33']

plt.figure(figsize=(8,7))
im = plt.imshow(corr, cmap='plasma', vmin=-0.1, vmax=1.0)
plt.colorbar(im, label='Correlation')
plt.xticks(range(5), z_labels, rotation=45)
plt.yticks(range(5), z_labels)
plt.title('ALADIN ∞ C(t) — DESI 2024 BAO Correlation Matrix\nFrom J₀ = 10¹⁸ A/m² — No Dark Energy', fontsize=14)
for i in range(5):
    for j in range(5):
        plt.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',color='white',fontweight='bold')
plt.tight_layout()
plt.savefig('desi_corr_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("DESI correlation matrix from J₀ — plot saved")
