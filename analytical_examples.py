"""Reproduce exact moment-indistinguishable mask examples and sharp frontiers."""
import json
from pathlib import Path
import numpy as np
from core import masks,moments,solve_policy,worst_risk
m=masks(3);A=np.c_[np.ones(8),np.zeros(8)];E=np.zeros((8,2));E[-1,0]=.5
qe=(m.sum(1)%2==0)/4;qo=(m.sum(1)%2==1)/4;qi=np.ones(8)/8
rows=[]
for order in [0,1,2,3]:
 B,b=moments(3,order,.5)
 for alpha in [.025,.05,.075,.1,.125,.15,.2]:
  x,c=solve_policy(A,E-alpha*A,B,b);r,q=worst_risk((A*x).sum(1),(E*x).sum(1),B,b)
  rows.append(dict(order=order,alpha=alpha,minimum_coverage=c,worst_risk=r,all_failed_acceptance=float(x[-1,0])))
B,b=moments(3,2,.5)
assert np.allclose(B@qe,B@qo)
np.testing.assert_allclose(B@(qo-qe),0)
# Under the marginal MCAR model, accepting the all-failed mask with probability .25
# controls alpha=.1. Making exactly those failures occur when Y=1 raises risk to .2.
out={'parity':{'masks':m.tolist(),'even':qe.tolist(),'odd':qo.tolist(),'independent':qi.tolist(),'risk_even':float(qe@E[:,0]),'risk_odd':float(qo@E[:,0]),'risk_independent':float(qi@E[:,0])},'frontier':rows,'label_dependent':{'failure_probability':.5,'all_failed_acceptance':.25,'mcar_risk':.1,'label_dependent_risk':.2,'coverage':.625}}
(Path(__file__).resolve().parent/'results/analytical.json').write_text(json.dumps(out,indent=2))
print('Exact analytical results verified and saved')
