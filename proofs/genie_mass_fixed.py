# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- HIGGS-GENIE COUPLING ---
v = 246.0  # Higgs VEV in GeV
y_c = 0.163  # Coupling constant
m_psi = y_c * v / 2  # Effective GENIE mass

print(f"m_ψ = y_c × v / 2 = {y_c} × {v} / 2 = {m_psi:.1f} GeV")

# --- SAVE CSV ---
csv_content = "param,value,unit\n"
csv_content += f"y_c,{y_c},\n"
csv_content += f"v,{v},GeV\n"
csv_content += f"m_psi,{m_psi:.1f},GeV\n"
with open('/content/genie_mass_fixed.csv', 'w') as f:
    f.write(csv_content)

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.bar(['Higgs VEV', 'GENIE Mass'], [v, m_psi], color=['red', 'gold'])
plt.ylabel('Mass (GeV)')
plt.title('Aladin v∞ — m_ψ = 20.1 GeV (y_c = 0.163)')
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE + DOWNLOAD ---
plt.savefig('/content/genie_mass_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

files.download('/content/genie_mass_fixed.py')
files.download('/content/genie_mass_fixed.csv')
files.download('/content/genie_mass_fixed.png')

print("genie_mass_fixed.py + .csv + .png saved + downloaded")
