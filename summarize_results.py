"""Aggregate experiment outputs into machine-readable numerical summaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent

def main():
 rows=[]
 for f in sorted((R/'results').glob('*_d[46]_*.json')):rows.extend(json.loads(f.read_text())['rows'])
 d=pd.DataFrame(rows);d.to_csv(R/'results/all_results.csv',index=False)
 out=[]
 for key,s in d.groupby(['split_mode','groups','model','alpha','method']):
  rec=dict(zip(['split_mode','groups','model','alpha','method'],key));rec['runs']=len(s)
  for k in [1,2]:
   for metric in ['worst_risk','min_coverage','cal_adversary_risk']:rec[f'{metric}_{k}']=s[f'{metric}_{k}'].mean()
   rec[f'violations_{k}']=int((s[f'worst_risk_{k}']>s.alpha+1e-8).sum())
  out.append(rec)
 pd.DataFrame(out).to_csv(R/'results/summary.csv',index=False)
 m=d[(d.split_mode=='random')&(d.groups==4)&(d.model!='neural')]
 out=[]
 for (alpha,method),s in m.groupby(['alpha','method']):
  rec=dict(alpha=alpha,method=method,runs=len(s))
  for k in [1,2]:
   rec[f'coverage_{k}']=s[f'min_coverage_{k}'].mean();rec[f'risk_{k}']=s[f'worst_risk_{k}'].mean();rec[f'violations_{k}']=int((s[f'worst_risk_{k}']>alpha+1e-8).sum())
  out.append(rec)
 pd.DataFrame(out).to_csv(R/'results/main_summary.csv',index=False)
 print('Aggregated',len(d),'benchmark policy evaluations.')
if __name__=='__main__':main()
