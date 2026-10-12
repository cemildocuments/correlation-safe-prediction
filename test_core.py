import unittest
import numpy as np
from core import masks,moments,solve_policy,worst_risk,minimum_coverage,action_statistics,solve_interval_policy

class CoreTests(unittest.TestCase):
    def test_parity_counterexample(self):
        m=masks(3);B,b=moments(3,2,.5);qe=(m.sum(1)%2==0)/4;qo=(m.sum(1)%2==1)/4
        np.testing.assert_allclose(B@qe,b);np.testing.assert_allclose(B@qo,b)
        e=np.zeros(8);e[-1]=.5;a=np.ones(8)
        self.assertAlmostEqual(qe@e,0);self.assertAlmostEqual(qo@e,.125)
        r,q=worst_risk(a,e,B,b);self.assertAlmostEqual(r,.125)
        A=np.c_[a,np.zeros(8)];E=np.c_[e,np.zeros(8)];x,c=solve_policy(A,E-.1*A,B,b)
        self.assertAlmostEqual(c,.9375);self.assertAlmostEqual(x[-1,0],.75)
        rr,_=worst_risk((A*x).sum(1),(E*x).sum(1),B,b);self.assertAlmostEqual(rr,.1)
    def test_marginal_example(self):
        B,b=moments(3,1,.5);A=np.c_[np.ones(8),np.zeros(8)];E=np.zeros((8,2));E[-1,0]=.5
        x,c=solve_policy(A,E-.1*A,B,b);self.assertAlmostEqual(c,.625);self.assertAlmostEqual(x[-1,0],.25)
    def test_reject_all(self):
        B,b=moments(2,1,.25);A=np.ones((4,2));A[:,-1]=0;H=A*.4
        x,c=solve_policy(A,H,B,b);self.assertAlmostEqual(c,0)
    def test_moment_monotonicity(self):
        rng=np.random.default_rng(7);A=np.c_[np.ones(8),rng.uniform(.2,.7,8),np.zeros(8)];E=A*rng.uniform(.01,.3,(8,3))
        vals=[]
        for k in [0,1,2,3]:
            B,b=moments(3,k,.25);_,c=solve_policy(A,E-.1*A,B,b);vals.append(c)
        self.assertTrue(np.all(np.diff(vals)>=-1e-8))
    def test_exact_actions(self):
        cc=np.ones((20,4))*.8;ee=np.zeros_like(cc,dtype=bool)
        a,e,A,H=action_statistics(cc,ee,[0,.9,1.01]);np.testing.assert_equal(A[:,0],1);np.testing.assert_equal(H[:,-1],0)

    def test_interval_equalities_match(self):
        rng=np.random.default_rng(33);A=np.c_[np.ones(8),rng.uniform(.2,.8,8),np.zeros(8)]
        E=A*rng.uniform(.01,.3,(8,3));B,b=moments(3,2,.25)
        _,c=solve_policy(A,E-.1*A,B,b)
        _,ci=solve_interval_policy(A,E-.1*A,B[1:],b[1:],b[1:])
        self.assertAlmostEqual(c,ci,places=8)
    def test_interval_widening(self):
        rng=np.random.default_rng(31);A=np.c_[np.ones(8),rng.uniform(.2,.8,8),np.zeros(8)]
        E=A*rng.uniform(.01,.3,(8,3));B,b=moments(3,2,.25)
        values=[]
        for width in [0.,.05,.15]:
            _,c=solve_interval_policy(A,E-.1*A,B[1:],np.maximum(0,b[1:]-width),np.minimum(1,b[1:]+width));values.append(c)
        self.assertTrue(np.all(np.diff(values)<=1e-8))
    def test_ratio_against_all_vertices(self):
        import itertools
        rng=np.random.default_rng(45);B,b=moments(3,1,.25)
        a=rng.uniform(.1,1,8);e=a*rng.uniform(.01,.4,8);vertices=[]
        for ix in itertools.combinations(range(8),len(b)):
            BB=B[:,ix]
            if abs(np.linalg.det(BB))<1e-10:continue
            q=np.zeros(8);q[list(ix)]=np.linalg.solve(BB,b)
            if q.min()>=-1e-10:vertices.append((q@e)/(q@a))
        r,_=worst_risk(a,e,B,b);self.assertAlmostEqual(r,max(vertices),places=9)

if __name__=='__main__':unittest.main()
