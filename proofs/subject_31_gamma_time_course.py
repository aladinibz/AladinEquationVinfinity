#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

raw = mne.io.read_raw_edf('subject_31_meditation_3h.edf', preload=True)
raw.filter(30, 100)  # gamma band

picks = ['AFz'] if 'AFz' in raw.ch_names else mne.pick_types(raw.info, eeg=True)[:1]

sfreq = raw.info['sfreq']
window_samples = int(2.0 * sfreq)  # 2 s window in samples
step_samples = int(0.5 * sfreq)    # 0.5 s step in samples

gamma_power = []
for start in range(0, len(raw.times) - window_samples, step_samples):
    data = raw.get_data(picks, start=start, stop=start+window_samples)[0]
    power = np.mean(np.abs(data)**2)
    gamma_power.append(power)

t_gamma = np.arange(len(gamma_power)) * 0.5

plt.figure(figsize=(12,8),dpi=1200)
plt.plot(t_gamma, gamma_power, 'gold', lw=3)
plt.axvline(41, color='darkred', ls='--', lw=4, label='t=41 s Burst')
plt.title("Subject 31 — Gamma Power Time Course\nBurst at 43 Hz Lock",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Gamma Power")
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/subject_31_gamma_time_course.png",dpi=1200)
print("Saved: plots/subject_31_gamma_time_course.png")
plt.show()
