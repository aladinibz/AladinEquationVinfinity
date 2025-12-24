import mne,np as np,plt as plt,os

os.makedirs("plots",exist_ok=True)

raw_dmt=mne.io.read_raw_bdf('S35-DMT.bdf',preload=True)
raw_ec=mne.io.read_raw_bdf('S35-EC.bdf',preload=True)
raw_eo=mne.io.read_raw_bdf('S35-EO.bdf',preload=True)

raw_dmt.filter(1,100);raw_ec.filter(1,100);raw_eo.filter(1,100)
raw_dmt.notch_filter(np.arange(50,249,50));raw_ec.notch_filter(np.arange(50,249,50));raw_eo.notch_filter(np.arange(50,249,50))

picks=mne.pick_types(raw_dmt.info,eeg=True)[:1]

data_dmt=raw_dmt[picks,:][0].flatten()
data_ec=raw_ec[picks,:][0].flatten()
data_eo=raw_eo[picks,:][0].flatten()

def higuchi_fd(ts,k_max=20):
    ts=np.asarray(ts).flatten();N=len(ts);L=[]
    for k in range(1,k_max+1):
        Lk=[]
        for m in range(k):
            num=(N-m)//k
            if num<2:continue
            Lk.append(np.sum(np.abs(np.diff(ts[m::k])))*(N-1)/(num*k))
        L.append(np.mean(Lk))
    return -np.polyfit(np.log(np.arange(1,k_max+1)[:len(L)]),np.log(L),1)[0]

def dfa(ts):
    ts=np.asarray(ts).flatten();y=np.cumsum(ts-ts.mean())
    scales=np.logspace(np.log10(4),np.log10(len(ts)//4),15,dtype=int)
    F=[]
    for n in scales:
        segs=len(ts)//n;rms=[]
        for v in range(segs):
            seg=y[v*n:(v+1)*n]
            trend=np.polyval(np.polyfit(np.arange(n),seg,1),np.arange(n))
            rms.append(np.sqrt(np.mean((seg-trend)**2)))
        F.append(np.mean(rms))
    return np.polyfit(np.log(scales),np.log(F),1)[0]

hfd_dmt=higuchi_fd(data_dmt);hfd_ec=higuchi_fd(data_ec);hfd_eo=higuchi_fd(data_eo)
dfa_dmt=dfa(data_dmt);dfa_ec=dfa(data_ec);dfa_eo=dfa(data_eo)

labels=['HFD DMT','HFD EC','HFD EO','DFA α DMT','DFA α EC','DFA α EO']
values=[hfd_dmt,hfd_ec,hfd_eo,dfa_dmt,dfa_ec,dfa_eo]

plt.figure(figsize=(12,8),dpi=1200)
plt.bar(labels,values,color=['gold','blue','green','gold','blue','green'])
plt.title("S35 — DMT vs EC vs EO\nHFD + DFA α",fontsize=18)
plt.ylabel("Value");plt.grid(alpha=0.4);plt.tight_layout()
plt.savefig("plots/s35_dmt_ec_eo_hfd_dfa_combined.png",dpi=1200)
print("Saved: plots/s35_dmt_ec_eo_hfd_dfa_combined.png")
plt.show()
