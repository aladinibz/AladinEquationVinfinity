!pip install matplotlib -q
import numpy as np, matplotlib.pyplot as plt
from google.colab import files

# --- VALIDATE ---
def v(x, name, minv=None, maxv=None):
    if x is None or np.isnan(x) or np.isinf(x): raise ValueError(f"{name} invalid")
    if minv and x < minv: raise ValueError(f"{name} < {minv}")
    if maxv and x > maxv: raise ValueError(f"{name} > {maxv}")

# --- PARAMS ---
y_c, v, J, B, rho, r = 0.163, 246.0, 1e18, 1e-6, 1e-24, 1e21
for val, name, mn, mx in [(y_c,"y_c",0,1),(v,"v",200,300),(J,"J",1e17,1e20),(B,"B",1e-7,1e-5),(rho,"rho",1e-25,1e-23),(r,"r",1e20,1e22)]:
    v(val, name, mn, mx)

# --- CALC ---
m_psi = y_c * v / np.sqrt(2)
k_ev = np.abs(J * B) / (3e8 * rho * r) * (1.602e-19 * 1e-9) / 197.3269718e-9
omega = np.sqrt(k_ev**2 + (m_psi * 1e9)**2) / 6.582119569e-16
f = omega / (2 * np.pi)
print(f"m_ψ={m_psi:.1f} GeV | k={k_ev:.2e} eV | f={f:.1f} Hz")

# --- SAVE CSV ---
csv = f"m_psi,{m_psi:.1f},GeV\nk,{k_ev:.2e},eV\nf,{f:.1f},Hz\n"
open('/content/43hz_conversion_full.csv','w').write("param,value,unit\n"+csv)

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.loglog([1e-12,1e-10],[43,43],'c',lw=2)
plt.scatter([k_ev],[f],c='gold',s=80)
plt.xlabel('k (eV)'); plt.ylabel('f (Hz)'); plt.title('43 Hz')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('/content/43hz_conversion_full.png', dpi=300); plt.show()

# --- DOWNLOAD ALL 3 ---
files.download('/content/43hz_conversion_full.py')
files.download('/content/43hz_conversion_full.csv')
files.download('/content/43hz_conversion_full.png')
print("ALL 3 DOWNLOADED")
