"""Replay the complete experiment protocol, optionally refitting all predictors."""
from pathlib import Path
import argparse,os,subprocess,sys
R=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--retrain',action='store_true');a=p.parse_args()
 if a.workers<1:p.error('--workers must be positive')
 env=os.environ.copy()
 if a.retrain:env['CSP_RETRAIN']='1'
 else:env.pop('CSP_RETRAIN',None)
 def run(*args):subprocess.run([sys.executable,*map(str,args)],cwd=R,env=env,check=True)
 run('validate_results.py','--inputs-only')
 run('-m','unittest','-v','test_core')
 run('run_experiments.py','--workers',a.workers)
 run('run_experiments.py','--models','neural','--seeds',3,'--workers',1)
 run('run_experiments.py','--models','extra_trees','--groups',6,'--seeds',3,'--workers',a.workers)
 run('run_experiments.py','--datasets','pendigits','satellite','--split-mode','official','--workers',a.workers)
 run('analytical_examples.py')
 run('simulations.py','--workers',a.workers)
 run('sensitivity.py')
 run('summarize_results.py')
 if not a.retrain:run('validate_results.py')
 print('Complete protocol finished. Numerical summaries are in results/summary.csv.',flush=True)
if __name__=='__main__':main()
