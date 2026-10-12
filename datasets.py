"""Load public UCI archives; remove exact duplicate feature-label rows before splitting."""
from pathlib import Path
import io, zipfile, subprocess, tempfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
ROOT=Path(__file__).resolve().parent

def load(name):
    path=ROOT/'data'/f'{name}.zip'
    with zipfile.ZipFile(path) as z:
        if name=='letter':
            df=pd.read_csv(z.open('letter-recognition.data'),header=None); y=df.iloc[:,0];X=df.iloc[:,1:];official=None
        elif name=='spambase':
            df=pd.read_csv(z.open('spambase.data'),header=None);X=df.iloc[:,:-1];y=df.iloc[:,-1];official=None
        elif name=='drybean':
            df=pd.read_excel(z.open('DryBeanDataset/Dry_Bean_Dataset.xlsx'));X=df.iloc[:,:-1];y=df.iloc[:,-1];official=None
        elif name in ['pendigits','satellite']:
            a,b=('pendigits.tra','pendigits.tes') if name=='pendigits' else ('sat.trn','sat.tst')
            kw={'sep':','} if name=='pendigits' else {'sep':r'\s+'}
            tr=pd.read_csv(z.open(a),header=None,**kw);te=pd.read_csv(z.open(b),header=None,**kw)
            df=pd.concat([tr,te],ignore_index=True);official=np.r_[np.zeros(len(tr),int),np.ones(len(te),int)]
            X=df.iloc[:,:-1];y=df.iloc[:,-1]
        elif name=='waveform':
            # UCI distributes this legacy file in UNIX compress format.
            r=subprocess.run(['gzip','-dc'],input=z.read('waveform.data.Z'),capture_output=True,check=True)
            df=pd.read_csv(io.BytesIO(r.stdout),header=None);X=df.iloc[:,:-1];y=df.iloc[:,-1];official=None
        else:raise ValueError(name)
    keep=~df.duplicated();X=np.asarray(X.loc[keep],dtype=np.float32);y=LabelEncoder().fit_transform(y.loc[keep])
    if official is not None:official=official[np.asarray(keep)]
    if not np.isfinite(X).all():raise ValueError('nonfinite source values')
    meta={'dataset':name,'raw_n':len(df),'n':len(y),'features':X.shape[1],'classes':len(np.unique(y)),
          'duplicates_removed':int((~keep).sum()),'official_test':official is not None}
    return X,y,official,meta

def split(name,seed,mode="random"):
    X,y,official,meta=load(name);rng=np.random.default_rng(seed)
    if official is None or mode=="random":
        ix=rng.permutation(len(y));ntr=int(.4*len(y));nc=int(.3*len(y));tr=ix[:ntr];ca=ix[ntr:ntr+nc];te=ix[ntr+nc:]
    else:
        ix=rng.permutation(np.flatnonzero(official==0));cut=int(.6*len(ix));tr=ix[:cut];ca=ix[cut:];te=np.flatnonzero(official==1)
    meta.update(n_train=len(tr),n_cal=len(ca),n_test=len(te))
    return X,y,tr,ca,te,meta
