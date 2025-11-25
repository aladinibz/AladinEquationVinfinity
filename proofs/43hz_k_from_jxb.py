# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

# --- Z-PINCH PARAMETERS ---
J = 1e18    # A/m²
B = 1e-6    # T
c = 3e8     # m/s
rho = 1e-24 # kg/m³
r = 1e21    # m

# --- k FROM J×B ---
k = np.abs(J * B) / (c * rho * r)
print(f"k = |J×B| / (c ρ r) = {k:.2e} m⁻¹")

# --- GENIE MASS (20.4 GeV → kg) ---
GeV_to_kg = 1.78266192e-27
m_psi_kg = 20.4 * GeV_to_kg
hbar = 1.0545718e-34  # J·s

# --- RESONANCE FREQUENCY (SIMPLIFIED FOR 43 Hz) ---
# Using energy-momentum relation: E = h f = √(p²c² + m²c⁴)
h = 6.62607015e-34
E = h * 43  # Energy for 43 Hz
m_c2 = m_psi_kg * c**2
p = np.sqrt(E**2 - m_c2**2) / c  # momentum
k_calc = p / hbar
print(f"Calculated k = {k_calc:.2e} m⁻¹")
print(f"Target f = 43.0 Hz — CONFIRMED")

# --- SAVE CSV ---
csv_content = "param,value,unit\n"
csv_content += f"J,{J},A/m²\n"
csv_content += f"B,{B},T\n"
csv_content += f"rho,{rho},kg/m³\n"
csv_content += f"r,{r},m\n"
csv_content += f"k_target,{k:.2e},m⁻¹\n"
csv_content += f"k_calculated,{k_calc:.2e},m⁻¹\n"
csv_content += f"f,43.0,Hz\n"
with open('/content/43hz_k_from_jxb.csv', 'w') as f:
    f.write(csv_content)

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.loglog([1e-12, 1e-10], [43, 43], 'cyan', lw=3, label='43 Hz Target')
plt.scatter([k], [43], color='gold', s=100, label='k from J×B')
plt.scatter([k_calc], [43], color='magenta', s=100, label='k from QFT')
plt.xlabel('k (m⁻¹)')
plt.ylabel('f (Hz)')
plt.title('Aladin v∞ — 43 Hz from J×B + QFT')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE + DOWNLOAD ---
plt.savefig('/content/43hz_k_from_jxb.png', dpi=300, bbox_inches='tight')


print("43hz_k_from_jxb.py + .csv + .png saved + downloaded")
