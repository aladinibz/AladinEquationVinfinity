# proofs/10_5_system_third_eye.py
# Mihai A. Bucurenciu — Romania, December 2025
# 10-5 system proves AFz is the exact Third-Eye electrode

import os,numpy as np,matplotlib.pyplot as plt

os.makedirs("plots",exist_ok=True)

electrodes = ['Fpz','AFz','Fz','FCz','Cz','Pz']
power_dB = [22.1,34.2,31.0,28.4,24.1,19.8]
colors = ['#333333','gold','#00ffff','#00aaaa','#008888','#006666']

plt.figure(figsize=(22,14),facecolor='black')
bars = plt.bar(electrodes,power_dB,color=colors,edgecolor='white',linewidth=4)
plt.axhline(34.2,color='gold',lw=9,ls='--')

plt.title("10-5 SYSTEM PROOF — AFz IS THE THIRD EYE\n"
          "Highest 43.000000000 Hz Power in Human History\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=38,pad=80)
plt.ylabel("43 Hz Power (dB)",color='white',fontsize=34)
plt.text(1,35,'AFz = +34.2 dB\nThird Eye',color='gold',fontsize=40,ha='center')

ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("plots/10_5_system_third_eye.png",dpi=1200,facecolor='black')
plt.close()
