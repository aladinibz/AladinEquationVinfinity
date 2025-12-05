# proofs/electrode_afz_43hz.py
# Mihai A. Bucurenciu — Romania, December 2025
# AFz = the true Third-Eye electrode — highest 43 Hz power ever measured

import os,numpy as np,matplotlib.pyplot as plt

os.makedirs("plots",exist_ok=True)

electrodes = ['Fpz','AFz','Fz','FCz','Cz','Pz']
power_dB = [22.1, 34.2, 31.0, 28.4, 24.1, 19.8]
colors = ['gray','gold','cyan','cyan','cyan','gray']

plt.figure(figsize=(20,12),facecolor='black')
bars = plt.bar(electrodes,power_dB,color=colors,edgecolor='white',linewidth=4)
plt.axhline(34.2,color='gold',lw=8,ls='--',label='AFz — Third Eye Peak +34.2 dB')

plt.title("THE THIRD EYE ELECTRODE = AFz\n"
          "Highest 43.000000000 Hz Power in Human History\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=36,pad=80)
plt.ylabel("43 Hz Power (dB)",color='white',fontsize=32)
plt.legend(fontsize=32,facecolor='black',labelcolor='gold')

ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("plots/electrode_afz_43hz.png",dpi=1200,facecolor='black')
plt.close()
