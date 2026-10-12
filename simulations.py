"""Exact counterexamples and repeated finite-sample validation on known populations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,time,argparse,concurrent.futures
from pathlib import Path
import numpy as np
from core import masks,moments,action_statistics,solve_policy,worst_risk,minimum_coverage
ROOT=Path(__file__).resolve().parent

def one(job):
    n,rep=job;d=4;m=masks(d);k=m.sum(1);alpha=.1
    hi=.8-.12*k;rh=.01+.012*k+.25*(k==4);rl=.20+.05*k
    A=np.c_[np.ones(16),hi,np.zeros(16)]
    E=np.c_[hi*rh+(1-hi)*rl,hi*rh,np.zeros(16)]
    rng=np.random.default_rng(900000+n*1000+rep)
    high=rng.random((n,1))<hi;conf=np.where(high,.8,.4);wrong=rng.random((n,1))<np.where(high,rh,rl)
    a,e,low,h=action_statistics(conf,wrong,[0.,.6,1.01],alpha)
    rows=[]
    for order in [1,2]:
        B,b=moments(d,order,.25)
        for mode in ['certified','plugin','population']:
            aa,hh=(low,h) if mode=='certified' else (a,e-alpha*a) if mode=='plugin' else (A,E-alpha*A)
            x,c=solve_policy(aa,hh,B,b);acta=(A*x).sum(1);acte=(E*x).sum(1);risk,_=worst_risk(acta,acte,B,b)
            rows.append(dict(n=n,rep=rep,order=order,mode=mode,true_worst_risk=risk,true_min_coverage=minimum_coverage(acta,B,b),certified_coverage=c))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=200);p.add_argument('--workers',type=int,default=8);args=p.parse_args();start=time.perf_counter()
    jobs=[(n,r) for n in [100,250,500,1000,2500,5000] for r in range(args.repetitions)]
    out=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for z in ex.map(one,jobs):out.extend(z)
    (ROOT/'results/simulation.json').write_text(json.dumps(out,indent=2,allow_nan=True));print('completed',len(jobs),'repetitions in',time.perf_counter()-start,'seconds')
