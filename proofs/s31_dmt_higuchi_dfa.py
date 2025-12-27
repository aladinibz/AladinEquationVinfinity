# s31_dmt_higuchi_dfa.py - Higuchi FD & DFA for S31-DMT.bdf
import mne,numpy as np,matplotlib.pyplot as plt
from scipy.stats import linregress

def higuchi_fd(d,k_max=50):
 N=len(d);if N<100:return np.nan
 L=[]
 for k in range(1,k_max+1):
  Lk=[np.mean([np.sum(np.abs(d[m-1:N:k][1:]-d[m-1:N:k][:-1]))*(N-1)/((len(d[m-1:N:k])-1)*k)for m in range(1,k+1)if len(d[m-1:N:k])>=2])]
  if Lk[0]:L.append(Lk[0])
 if not L:return np.nan
 return -linregress(np.log(1/np.arange(1,len(L)+1)),np.log(L))[0]

def dfa(d,scales=np.logspace(1,3,20).astype(int)):
 N=len(d);if N<100:return np.nan
 a=[]
 for s in scales:
  if s>=N:continue
  y=np.cumsum(d-np.mean(d))
  rms=[np.sqrt(np.mean((seg-np.polyval(np.polyfit(np.arange(len(seg)),seg,1),np.arange(len(seg))))**2))for seg in np.array_split(y,N//s)if len(seg)>=2]
  a.append(np.mean(rms)if rms else 0)
 if not a:return np.nan
 return linregress(np.log(scales[:len(a)]),np.log(np.array(a)+1e-10))[0]

raw=mne.io.read_raw_bdf('data/raw_bdf/S31-DMT.bdf',preload=True)
raw.set_eeg_reference('average',projection=True)
raw.filter(1,100)
raw.notch_filter(50)
data=raw.get_data(picks='eeg')[0][:100000]

w=10000;s=1000
hfd_v,dfa_v,t=[],[],[]
for i in range(0,len(data)-w,s):
 win=data[i:i+w]
 h=higuchi_fd(win);d=dfa(win)
 if not np.isnan(h)and not np.isnan(d):
  hfd_v.append(h);dfa_v.append(d);t.append(i/raw.info['sfreq'])

fig,(a1,a2)=plt.subplots(2,1,figsize=(12,10),sharex=True)
a1.plot(t,hfd_v,color='red');a1.axvline(41,color='green',ls='--');a1.set_ylabel('HFD');a1.set_title('S31-DMT: Higuchi FD')
a2.plot(t,dfa_v,color='blue');a2.axvline(41,color='green',ls='--');a2.set_xlabel('Time (s)');a2.set_ylabel('DFA')
plt.tight_layout();plt.savefig('s31_dmt_higuchi_dfa.png',dpi=300,bbox_inches='tight');plt.close()
print("Plot saved: s31_dmt_higuchi_dfa.png")
