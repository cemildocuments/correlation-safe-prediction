"""Run held-out classification experiments with auditable moment uncertainty."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import argparse,concurrent.futures,json,time,platform,sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from core import masks,moments,action_statistics,solve_policy,worst_risk,minimum_coverage,pooled_policy
from datasets import split
ROOT=Path(__file__).resolve().parent
THRESHOLDS=np.r_[0.,np.arange(.3,1.,.05),1.01]

def features(X,m):
    # Round-robin grouping is a controlled block-missingness benchmark.
    d=m.shape[-1]; group=np.arange(X.shape[1])%d
    return np.concatenate([X*(1-m[...,group]), np.broadcast_to(m,(len(X),d))],axis=1).astype(np.float32)

def fit_predict(name,seed,family,d=4,mode="random"):
    X,y,tr,ca,te,meta=split(name,seed,mode);sc=StandardScaler().fit(X[tr]);X=sc.transform(X)
    rng=np.random.default_rng(seed+12345);xx=np.repeat(X[tr],4,axis=0);yy=np.repeat(y[tr],4)
    mm=(rng.random((len(xx),d))<.25).astype(np.float32);xx=features(xx,mm)
    if family=='extra_trees':model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,max_features=1.,n_jobs=1,random_state=seed)
    elif family=='logistic':model=LogisticRegression(C=1.,max_iter=600,solver='lbfgs',random_state=seed)
    elif family=='neural':
        from neural import NeuralClassifier
        model=NeuralClassifier(seed=seed)
    else:raise ValueError(family)
    start=time.perf_counter();model.fit(xx,yy);train_seconds=time.perf_counter()-start
    out=[]
    for ix in [ca,te]:
        cc=[];ee=[]
        for m in masks(d):
            proba=model.predict_proba(features(X[ix],m));cc.append(proba.max(axis=1));ee.append(model.classes_[proba.argmax(axis=1)]!=y[ix])
        out.extend([np.array(cc,dtype=np.float32).T,np.array(ee,dtype=bool).T])
    meta.update(seed=seed,model=family,groups=d,split_mode=mode,train_seconds=train_seconds)
    return out,meta

def run(job):
    name,seed,family,d,mode=job;threadpool_limits(limits=1)
    tag=f'{name}_{family}_{seed}_d{d}_{mode}';outdir=ROOT/'results';outdir.mkdir(exist_ok=True)
    cache=ROOT/'cache';cache.mkdir(exist_ok=True);cp=cache/f'{tag}.npz'
    if cp.exists() and not os.environ.get("CSP_RETRAIN"):
        z=np.load(cp);data=[z[k] for k in ['cal_conf','cal_wrong','test_conf','test_wrong']];meta=json.loads(str(z['meta']))
    else:
        data,meta=fit_predict(name,seed,family,d,mode);np.savez_compressed(cp,**dict(zip(['cal_conf','cal_wrong','test_conf','test_wrong'],data)),meta=json.dumps(meta))
    cc,ce,tc,te=data;M=2**d;rows=[];policies={};p=.25
    q=(p**masks(d).sum(1))*((1-p)**(d-masks(d).sum(1)))
    B1,b1=moments(d,1,p);B2,b2=moments(d,2,p)
    for alpha in [.1,.2]:
        ac,ec,A,H=action_statistics(cc,ce,THRESHOLDS,alpha)
        at,et,_,_=action_statistics(tc,te,THRESHOLDS,alpha)
        pol={}
        pol['pooled']=(pooled_policy(cc,ce,THRESHOLDS,alpha,.05,p,seed+700),float('nan'))
        for method,B,b in [('nominal',np.eye(M),q),('marginal',B1,b1),('pairwise',B2,b2)]:
            start=time.perf_counter();x,c=solve_policy(A,H,B,b);pol[method]=(x,c)
            policies[f'{alpha}_{method}']={'weights':x.tolist(),'certified_min_coverage':c,'solve_seconds':time.perf_counter()-start}
        for method,B,b in [('marginal_shared',B1,b1),('pairwise_shared',B2,b2)]:
            pol[method]=solve_policy(A,H,B,b,shared=True)
        from scipy.optimize import linprog
        xm=np.zeros_like(A)
        for mi in range(M):
            rr=linprog(-A[mi],A_ub=H[mi][None,:],b_ub=[0.],A_eq=np.ones((1,len(THRESHOLDS))),b_eq=[1.],bounds=(0,1),method='highs')
            if not rr.success:raise RuntimeError(rr.message)
            xm[mi]=rr.x
        pol['maskwise']=(xm,float(np.min((A*xm).sum(1))))
        # Finite-sample ablation: same robust optimization, raw empirical inputs.
        pol['pairwise_plugin']=solve_policy(ac,ec-alpha*ac,B2,b2)
        for method,(x,c) in pol.items():
            aa=(at*x).sum(1);ee=(et*x).sum(1)
            aac=(ac*x).sum(1);eec=(ec*x).sum(1)
            rec={**meta,'alpha':alpha,'method':method,'certified_coverage':c,'iid_coverage':float(q@aa),'iid_risk':float((q@ee)/(q@aa)) if q@aa>1e-12 else None}
            for order,B,b in [(1,B1,b1),(2,B2,b2)]:
                risk,qt=worst_risk(aa,ee,B,b);_,qc=worst_risk(aac,eec,B,b)
                rec[f'worst_risk_{order}']=risk;rec[f'min_coverage_{order}']=minimum_coverage(aa,B,b)
                rec[f'cal_adversary_risk_{order}']=float((qc@ee)/(qc@aa)) if qc@aa>1e-12 else None
                rec[f'cal_adversary_coverage_{order}']=float(qc@aa)
            rows.append(rec)
    (outdir/f'{tag}.json').write_text(json.dumps({'rows':rows,'policies':policies},indent=2,allow_nan=True))
    return tag,meta['train_seconds']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['letter','pendigits','drybean','spambase','satellite','waveform']);p.add_argument('--seeds',type=int,default=10);p.add_argument('--models',nargs='+',default=['extra_trees','logistic']);p.add_argument('--workers',type=int,default=8);p.add_argument('--groups',type=int,default=4);p.add_argument('--split-mode',default='random',choices=['random','official']);args=p.parse_args()
    jobs=[(n,s,m,args.groups,args.split_mode) for n in args.datasets for s in range(args.seeds) for m in args.models]
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(run,j):j for j in jobs}
        for f in concurrent.futures.as_completed(fs):print(f.result(),flush=True)
    print('Total wall seconds',time.perf_counter()-start,flush=True)
