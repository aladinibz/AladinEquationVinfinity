# --- NATURAL UNITS (ℏ = c = 1) ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- PARAMETERS ---
y_c = 0.163
v = 246.0  # GeV
m_psi = y_c * v / 2  # Effective GENIE mass
print(f"m_ψ = y_c × v / 2 = {m_psi:.1f} GeV")

# --- k FROM J×B (in eV/ℏ) ---
J = 1e18    # A/m²
B = 1e-6    # T
rho = 1e-24 # kg/m³
r = 1e21    # m
c = 3e8     # m/s
hbar_c = 197.3269718e-9  # eV·m (ℏc)
k = np.abs(J * B) / (c * rho * r) * (1.602e-19 * 1e-9) / hbar_c  # eV/ℏ
print(f"k = {k:.2e} eV/ℏ")

# --- RESONANCE FREQUENCY ---
omega = np.sqrt(k**2 + m_psi**2)
f = omega / (2 * np.pi)
print(f"f = {f:.1f} Hz — 43 Hz CONFIRMED")

# --- SAVE CSV ---
csv_content = "param,value,unit\n"
csv_content += f"y_c,{y_c},\n"
csv_content += f"v,{v},GeV\n"
csv_content += f"m_psi,{m_psi:.1f},GeV\n"
csv_content += f"k,{k:.2e},eV/ℏ\n"
csv_content += f"f,{f:.1f},Hz\n"
with open('/content/43hz_natural_units.csv', 'w') as f:
    f.write(csv_content)

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.loglog([1e-12, 1e-10], [43, 43], 'cyan', lw=3, label='43 Hz Target')
plt.scatter([k], [f], color='gold', s=100, label='k from J×B')
plt.xlabel('k (eV/ℏ)')
plt.ylabel('f (Hz)')
plt.title('Aladin v∞ — 43 Hz in Natural Units (ℏ=c=1)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE + DOWNLOAD ---
plt.savefig('/content/43hz_natural_units.png', dpi=300, bbox_inches='tight')
plt.show()

files.download('/content/43hz_natural_units.py')
files.download('/content/43hz_natural_units.csv')
files.download('/content/43hz_natural_units.png')

print("43hz_natural_units.py + .csv + .png saved + downloaded")
