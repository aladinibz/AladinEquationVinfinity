import numpy as np
import matplotlib.pyplot as plt

# Roman WFI: 512x512, 0.28"/pix, ~2.3 deg FOV
n = 512
image = np.zeros((n, n))

# 100 high-z galaxies (z=10-20)
z = np.linspace(10, 20, 100)
H0, n_exp = 75.2, 0.25
flux = 100 / (1 + z)**2

x = np.random.randint(50, n-50, 100)
y = np.random.randint(50, n-50, 100)

# Add galaxies (Gaussian PSF)
for i in range(100):
    gx, gy = np.meshgrid(np.arange(-10,11), np.arange(-10,11))
    psf = np.exp(-(gx**2 + gy**2)/(2*2**2))
    image[x[i]-10:x[i]+11, y[i]-10:y[i]+11] += flux[i] * psf

# Noise
image += np.random.poisson(0.01, image.shape) + np.random.normal(0, 5, image.shape)

# Plot
plt.figure(figsize=(8,8))
plt.imshow(image, cmap='gray', origin='lower')
plt.title('Simulated Roman WFI — 100 Galaxies z=10–20')
plt.xlabel('X (pixels)'); plt.ylabel('Y (pixels)')
plt.tight_layout()
plt.savefig('roman_wfi_simulation.png', dpi=300)
plt.close()

print("Roman WFI sim saved: roman_wfi_simulation.png")
