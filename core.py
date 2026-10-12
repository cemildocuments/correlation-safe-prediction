"""Moment-constrained selective prediction; probabilities use mask bit 1 = absent."""
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta


def masks(d):
    return np.array(list(itertools.product([0, 1], repeat=d)), dtype=float)


def moments(d, order=1, p=.25):
    m = masks(d)
    rows = [np.ones(len(m))]
    targets = [1.]
    for k in range(1, order+1):
        for ix in itertools.combinations(range(d), k):
            rows.append(m[:, ix].prod(axis=1)); targets.append(p**k)
    return np.array(rows), np.array(targets)


def cp_upper(k, n, tail):
    k, n = np.broadcast_arrays(k, n)
    return np.where(k >= n, 1., beta.ppf(1-tail, k+1, np.maximum(n-k, 1)))


def cp_lower(k, n, tail):
    k, n = np.broadcast_arrays(k, n)
    return np.where(k <= 0, 0., beta.ppf(tail, np.maximum(k, 1), n-k+1))


def action_statistics(conf, wrong, thresholds, alpha=.1, delta=.05):
    """Independent rows, potentially dependent mask evaluations. Familywise CP bounds.

    Thresholds must be fixed independently of these calibration rows.
    Final action is deterministic reject. First threshold 0 is accept-all.
    """
    n, M = conf.shape; J = len(thresholds)
    accept = conf[:, :, None] >= np.array(thresholds)[None, None, :]
    ka = accept.sum(axis=0); ke = (accept * wrong[:, :, None]).sum(axis=0)
    kc = ka-ke
    # Three one-sided families: error upper, accepted-correct lower, acceptance lower.
    tail = delta/(3*M*(J-1))
    a = ka/n; e = ke/n
    au = cp_lower(ka, n, tail)
    eu = cp_upper(ke, n, tail)
    cl = cp_lower(kc, n, tail)
    h = (1-alpha)*eu-alpha*cl
    # Accept-all has known acceptance exactly 1, so use tighter same error event.
    au[:, 0] = 1.; h[:, 0] = eu[:, 0]-alpha
    au[:, -1] = 0.; h[:, -1] = 0.
    return a, e, au, h


def solve_policy(A, H, B, b, shared=False):
    """Maximize worst-case certified coverage using LP duality.

    Variables: x[M,J], y[r], z[r], c. B q=b; q>=0.
    B.T y >= Hx, b.T y<=0, B.T z<=Ax, b.T z>=c.
    """
    M, J = A.shape; r = len(b); N=M*J
    obj=np.zeros(N+2*r+1); obj[-1]=-1
    ub=[]; rhs=[]
    for m in range(M):
        row=np.zeros(len(obj));row[m*J:(m+1)*J]=H[m];row[N:N+r]=-B[:,m]
        ub.append(row);rhs.append(0)
        row=np.zeros(len(obj));row[m*J:(m+1)*J]=-A[m];row[N+r:N+2*r]=B[:,m]
        ub.append(row);rhs.append(0)
    row=np.zeros(len(obj));row[N:N+r]=b;ub.append(row);rhs.append(0)
    row=np.zeros(len(obj));row[N+r:N+2*r]=-b;row[-1]=1;ub.append(row);rhs.append(0)
    eq=np.zeros((M,len(obj)))
    for m in range(M):eq[m,m*J:(m+1)*J]=1
    eq_rhs=np.ones(M)
    if shared:
        rows=[]
        for m in range(1,M):
            for j in range(J):
                row=np.zeros(len(obj));row[m*J+j]=1;row[j]=-1;rows.append(row)
        eq=np.vstack([eq,rows]);eq_rhs=np.r_[eq_rhs,np.zeros(len(rows))]
    out=linprog(obj,A_ub=np.array(ub),b_ub=np.array(rhs),A_eq=eq,b_eq=eq_rhs,
                bounds=[(0,1)]*N+[(None,None)]*(2*r)+[(0,1)],method='highs')
    if not out.success:raise RuntimeError(out.message)
    x=np.maximum(out.x[:N].reshape(M,J),0);x/=x.sum(axis=1)[:,None]
    # Numerically verify primal worst cases independently of the optimization dual.
    worst=linprog(-(H*x).sum(1),A_eq=B,b_eq=b,bounds=(0,None),method='highs')
    low=linprog((A*x).sum(1),A_eq=B,b_eq=b,bounds=(0,None),method='highs')
    if not worst.success or not low.success:raise RuntimeError('audit LP failed')
    if -worst.fun>1e-7 or low.fun < out.x[-1]-1e-7:raise AssertionError('invalid certificate')
    return x,float(out.x[-1])


def worst_risk(a,e,B,b):
    """Sharp empirical ratio supremum via Charnes--Cooper transformation."""
    if a.max()<1e-12:return float('nan'), np.zeros(len(a))
    M=len(a)
    eq=np.zeros((len(b)+1,M+1));eq[:-1,:M]=B;eq[:-1,-1]=-b;eq[-1,:M]=a
    rhs=np.zeros(len(b)+1);rhs[-1]=1
    res=linprog(np.r_[-e,0],A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs')
    if not res.success:
        # No positive coverage in the feasible ambiguity set.
        cov=linprog(-a,A_eq=B,b_eq=b,bounds=(0,None),method='highs')
        if cov.success and -cov.fun<1e-10:return float('nan'),np.zeros(M)
        raise RuntimeError(res.message)
    q=np.maximum(res.x[:M]/res.x[-1],0);q/=q.sum()
    return float(-res.fun),q


def minimum_coverage(a,B,b):
    res=linprog(a,A_eq=B,b_eq=b,bounds=(0,None),method='highs')
    if not res.success:raise RuntimeError(res.message)
    return float(res.fun)


def pooled_policy(conf,wrong,thresholds,alpha,delta,p,seed):
    n,M=conf.shape;d=int(np.log2(M));rng=np.random.default_rng(seed)
    ids=(rng.random((n,d))<p).astype(int) @ (2**np.arange(d-1,-1,-1))
    cc=conf[np.arange(n),ids];ee=wrong[np.arange(n),ids]
    valid=[]
    for j,t in enumerate(thresholds[:-1]):
        keep=cc>=t;k=int(keep.sum())
        if k and cp_upper(int(ee[keep].sum()),k,delta/(len(thresholds)-1))<=alpha:valid.append((k,j))
    chosen=max(valid)[1] if valid else len(thresholds)-1
    x=np.zeros((M,len(thresholds)));x[:,chosen]=1
    return x


def solve_interval_policy(A,H,F,lower,upper):
    """Certified maximin selector when mask moments are known only in intervals."""
    M,J=A.shape;R=np.vstack([np.ones(M),F,-F]);v=np.r_[1.,upper,-lower];r=len(v);N=M*J
    obj=np.zeros(N+2*r+1);obj[-1]=-1;ub=[];rhs=[]
    for m in range(M):
        row=np.zeros(len(obj));row[m*J:(m+1)*J]=H[m];row[N:N+r]=-R[:,m];ub.append(row);rhs.append(0.)
        row=np.zeros(len(obj));row[m*J:(m+1)*J]=-A[m];row[N+r:N+2*r]=R[:,m];ub.append(row);rhs.append(0.)
    row=np.zeros(len(obj));row[N:N+r]=v;ub.append(row);rhs.append(0.)
    row=np.zeros(len(obj));row[N+r:N+2*r]=-v;row[-1]=1;ub.append(row);rhs.append(0.)
    eq=np.zeros((M,len(obj)))
    for m in range(M):eq[m,m*J:(m+1)*J]=1
    bounds=[(0,1)]*N+[(None,None)]+[(0,None)]*(r-1)+[(None,None)]+[(None,0)]*(r-1)+[(0,1)]
    res=linprog(obj,A_ub=np.array(ub),b_ub=rhs,A_eq=eq,b_eq=np.ones(M),bounds=bounds,method='highs')
    if not res.success:raise RuntimeError(res.message)
    x=np.maximum(res.x[:N].reshape(M,J),0);x/=x.sum(1)[:,None]
    kwargs=dict(A_eq=np.ones((1,M)),b_eq=[1.],A_ub=R[1:],b_ub=v[1:],bounds=(0,None),method='highs')
    check=linprog(-(H*x).sum(1),**kwargs);checkc=linprog((A*x).sum(1),**kwargs)
    if not check.success or -check.fun>1e-7 or not checkc.success or checkc.fun<res.x[-1]-1e-7:raise AssertionError('interval certificate invalid')
    return x,float(res.x[-1])
