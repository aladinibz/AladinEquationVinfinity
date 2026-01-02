"""
ALADIN Master Equation — The Complete Cosmic Grid
9-Panel Flagship Visualization
One Equation Rules All Pillars
January 02, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Data
# Cosmology
n = np.arange(1, 21)
l_local = 224 * n
fade = 1 + 0.0037 * n * 2.8
l_aladin = l_local * fade
l_obs = [220, 540, 812, 1055, 1298, 1540, 1783, 2025, 2268, 2510, 2753, 2995, 3238, 3480, 3723, 3965, 4208, 4450, 4693, 5410]

# Consciousness
time_dmt = np.linspace(0, 100, 500)
time_nat = np.linspace(0, 180, 500)
delta_dmt = 0.1 + 0.9 * np.exp(-(time_dmt - 41)**2 / 50)
delta_nat = 0.1 + 0.9 * np.exp(-time_nat / 60)

# Quantum Biology
f = np.logspace(0, 2.5, 1000)
response = np.exp(-((np.log(f) - np.log(43))**2) / 0.05)

# Enlightenment
t_switch = np.linspace(0, 100, 1000)
apoptosis = 1 / (1 + np.exp(-(t_switch - 41)/3))
telomerase = 1 - apoptosis

# Retrocausality
time_retro = np.linspace(0, 0.1, 1000)
phi_retro = np.exp(-time_retro / 0.01)

# Quantum Gravity
r = np.linspace(0.1, 10, 500)
temp_standard = 1 / (8 * np.pi * r)
temp_genie = temp_standard * np.exp(-r**2 / 2)

# Spin-100
delta_spin = np.logspace(-4, -1, 100)
weak_spin = 1 / (2 * delta_spin)

# Multidimensional Explorers
dim = np.linspace(1, 64, 100)
exploration = np.log(dim) * np.sin(2 * np.pi * dim / 64) + dim**0.5

# Colors
gold = '#FFD700'
dark_blue = '#003366'
purple = '#4B0082'
dark_green = '#006400'
red = '#8B0000'
cyan = '#00FFFF'

# Figure
fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

# Panel 1: Cosmology
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(n, l_obs, 'o', color=gold, markersize=10, label='Observed')
ax1.plot(n, l_aladin, '-', color=dark_blue, linewidth=4, label='ALADIN')
ax1.set_title('Cosmology')
ax1.set_xlabel('n')
ax1.set_ylabel('ℓ')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Consciousness
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_dmt, delta_dmt, color=purple, linewidth=4, label='DMT')
ax2.plot(time_nat, delta_nat, color=dark_green, linewidth=4, label='Natural')
ax2.axvline(41, color=gold, linestyle='--', linewidth=3)
ax2.set_title('Consciousness')
ax2.set_xlabel('Time')
ax2.set_ylabel('Δα')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Quantum Biology
ax3 = fig.add_subplot(gs[0, 2])
ax3.semilogx(f, response, color=dark_green, linewidth=4)
ax3.axvline(43, color=gold, linestyle='--', linewidth=3)
ax3.set_title('Quantum Biology')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Response')
ax3.grid(True, alpha=0.3)

# Panel 4: Enlightenment
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(t_switch, apoptosis, color=red, linewidth=4, label='Apoptosis OFF')
ax4.plot(t_switch, telomerase, color=dark_green, linewidth=4, label='Telomerase ON')
ax4.axvline(41, color=gold, linestyle='--', linewidth=3)
ax4.set_title('Enlightenment')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('State')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Panel 5: Retrocausality
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(time_retro * 1000, phi_retro, color=purple, linewidth=4)
ax5.axvline(41, color=gold, linestyle='--', linewidth=3)
ax5.set_title('Retrocausality')
ax5.set_xlabel('Time (ms)')
ax5.set_ylabel('φ')
ax5.grid(True, alpha=0.3)

# Panel 6: Quantum Gravity
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(r, temp_standard, '--', color='gray', linewidth=3, label='Standard')
ax6.plot(r, temp_genie, color=dark_blue, linewidth=4, label='GENIE')
ax6.set_title('Quantum Gravity')
ax6.set_xlabel('r / r_s')
ax6.set_ylabel('Temperature')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Panel 7: Spin-100
ax7 = fig.add_subplot(gs[2, 0])
ax7.loglog(delta_spin, weak_spin, color=red, linewidth=4)
ax7.axhline(100, color=gold, linestyle='--', linewidth=3)
ax7.set_title('Spin-100')
ax7.set_xlabel('δ')
ax7.set_ylabel('Weak value')
ax7.grid(True, alpha=0.3)

# Panel 8: Cosmic Unity
ax8 = fig.add_subplot(gs[2, 1])
ax8.text(0.5, 0.5, 'One Current J₀\nUniverse as Plasma Web\nOscillating at 43 Hz', 
         ha='center', va='center', fontsize=18, color=gold,
         bbox=dict(facecolor='black', alpha=0.8))
ax8.axis('off')
ax8.set_title('Cosmic Unity')

# Panel 9: Multidimensional Explorers
ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(dim, exploration, color=cyan, linewidth=4)
ax9.set_title('Multidimensional Explorers')
ax9.set_xlabel('Dimension')
ax9.set_ylabel('Freedom')
ax9.grid(True, alpha=0.3)

fig.suptitle('ALADIN Master Equation — The Complete Cosmic Grid', fontsize=28, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('plots/aladin_master_equation_cosmic_grid.png', dpi=500, bbox_inches='tight')
plt.close()
