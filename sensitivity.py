"""Sensitivity analyses from held-out prediction caches; no model reselection."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,concurrent.futures
from pathlib import Path
import numpy as np
from core import masks,moments,action_statistics,solve_policy,worst_risk,minimum_coverage,solve_interval_policy,cp_lower,cp_upper
from run_experiments import THRESHOLDS
ROOT=Path(__file__).resolve().parent

def one(file):
    z=np.load(file);meta=json.loads(str(z['meta']));cc,ce,tc,te=[z[k] for k in ['cal_conf','cal_wrong','test_conf','test_wrong']]
    d=meta['groups'];M=2**d;rows=[]
    at,et,_,_=action_statistics(tc,te,THRESHOLDS,.1)
    for p in [.1,.25,.4]:
        B,b=moments(d,1,p)
        for frac in [.25,.5,1.]:
            n=int(len(cc)*frac);_,_,A,H=action_statistics(cc[:n],ce[:n],THRESHOLDS,.1)
            x,c=solve_policy(A,H,B,b);a=(at*x).sum(1);e=(et*x).sum(1);risk,_=worst_risk(a,e,B,b)
            rows.append({**meta,'analysis':'calibration','p':p,'n_cal_used':n,'fraction':frac,'worst_risk':risk,'minimum_coverage':minimum_coverage(a,B,b),'certified_coverage':c})
    B,b=moments(d,2,.25);F=B[1:];rng=np.random.default_rng(meta['seed']+44000)
    _,_,A,H=action_statistics(cc,ce,THRESHOLDS,.1,.025)
    for ntele in [200,1000,10000,100000]:
        obs=(rng.random((ntele,d))<.25).astype(int);ids=obs@(2**np.arange(d-1,-1,-1));counts=F[:,ids].sum(1);tail=.025/(2*len(F))
        lo=cp_lower(counts,ntele,tail);hi=cp_upper(counts,ntele,tail)
        x,c=solve_interval_policy(A,H,F,lo,hi);a=(at*x).sum(1);e=(et*x).sum(1);risk,_=worst_risk(a,e,B,b)
        rows.append({**meta,'analysis':'telemetry','n_telemetry':ntele,'contains_true_moments':bool(np.all((lo<=b[1:])&(b[1:]<=hi))),'worst_risk':risk,'minimum_coverage':minimum_coverage(a,B,b),'certified_coverage':c,'mean_interval_width':float(np.mean(hi-lo))})
    return rows

if __name__=='__main__':
    files=[f for f in (ROOT/'cache').glob('*extra_trees_*_d4_random.npz') if int(f.name.split('_')[-3])<5]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one,files):rows+=r
    (ROOT/'results/sensitivity.json').write_text(json.dumps(rows,indent=2,allow_nan=True));print('sensitivity rows',len(rows))
