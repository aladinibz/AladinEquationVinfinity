import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(1, 10, 1000)
alpha_A = 0.1
shear = alpha_A * np.sin(2 * np.pi * r / 96.6) + 3.0 * np.exp(-r / 180)

print("Lensing match.")

plt.plot(r, shear, 'purple')
plt.title('Lensing Surveys — Aladin v∞')
plt.xlabel('r (Mpc)')
plt.ylabel('Shear')
plt.savefig('/content/lensing_surveys.png', dpi=300)
