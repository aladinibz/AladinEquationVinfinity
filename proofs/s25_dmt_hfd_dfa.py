import mne,numpy as np,matplotlib.pyplot as plt,os,scipy.stats
from IPython.display import Image

B='data/raw_bdf/S25-DMT.bdf';C='Fz';T=41.0;W=30;D='plots'
os.makedirs(D,exist_ok=True)

r=mne.io.read_raw_bdf(B,preload=True)
d=r.get_data(picks=[C])[0] if C in r.ch_names else r.average().get_data()[0]
sf=r.info['sfreq'];s=int((T-W)*sf);e=int((T+W)*sf)
pre=d[s:int(T*sf)];post=d[int(T*sf):e]

def h(sig):
 N=len(sig);L=[]
 for k in range(1,81):
  Lk=[np.sum(np.abs(np.diff(sig[m-1:N:k])))*(N-1)/((len(sig[m-1:N:k])-1)*k)for m in range(1,k+1)if len(sig[m-1:N:k])>1]
  if Lk:L.append(np.mean(Lk))
 L=np.array(L);v=L>0
 if np.sum(v)<2:return np.nan
 return np.polyfit(np.log(1/np.arange(1,81)[v]),np.log(L[v]),1)[0]

def df(sig):
 s=np.logspace(1,np.log2(len(sig)/4),15).astype(int)
 c=np.cumsum(sig-np.mean(sig));a=[]
 for sc in s:
  segs=len(c)//sc
  if segs<1:continue
  rms=[np.sqrt(np.mean((c[i*sc:(i+1)*sc]-np.polyval(np.polyfit(range(sc),c[i*sc:(i+1)*sc],1),range(sc)))**2))for i in range(segs)]
  a.append(np.mean(np.log(rms))/np.log(sc))
 if len(a)<2:return np.nan
 return np.polyfit(np.log(s[:len(a)]),a,1)[0]

hp,ho=h(pre),h(post);dp,do=df(pre),df(post)
ph=scipy.stats.wilcoxon([hp],[ho]).pvalue if not np.isnan(hp)and not np.isnan(ho) else np.nan
pd=scipy.stats.wilcoxon([dp],[do]).pvalue if not np.isnan(dp)and not np.isnan(do) else np.nan

plt.figure(figsize=(6,4))
x=np.arange(2)
plt.bar(x-.2,[hp,dp],.4,label='Pre',color='blue')
plt.bar(x+.2,[ho,do],.4,label='Post',color='red')
plt.xticks(x,['HFD','DFA']);plt.ylabel('Value')
plt.title(f'S25 DMT p_HFD={ph:.3e} p_DFA={pd:.3e}')
plt.legend();plt.grid(True,ls='--',alpha=.5)

save=os.path.join(D,'s25_dmt_hfd_dfa.png')
plt.savefig(save,dpi=200,bbox_inches='tight')
plt.show()
display(Image(save))
