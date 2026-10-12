"""Verify source hashes and compare replayed numerical results with frozen results."""
from pathlib import Path
import argparse,hashlib,json,math
R=Path(__file__).resolve().parent

def normalize(x):
 if isinstance(x,dict):return {k:normalize(v) for k,v in x.items() if not k.endswith('_seconds')}
 if isinstance(x,list):
  z=[normalize(v) for v in x]
  if z and all(isinstance(v,dict) for v in z):z.sort(key=lambda v:json.dumps(v,sort_keys=True))
  return z
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def compare(a,b,path=''):
 if isinstance(a,dict):
  assert isinstance(b,dict) and a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9),(path,a,b)
 else:assert a==b,(path,a,b)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs-only',action='store_true');p.add_argument('--freeze',action='store_true');a=p.parse_args()
 manifest=json.loads((R/'data_manifest.json').read_text())
 for key,z in manifest.items():assert hashlib.sha256((R/'data'/f'{key}.zip').read_bytes()).hexdigest()==z['sha256'],key
 if a.inputs_only:
  print('Verified six original dataset archives.');return
 files=sorted((R/'results').glob('*.json'))
 actual={f.name:normalize(json.loads(f.read_text())) for f in files}
 if a.freeze:
  (R/'expected_results.json').write_text(json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False));print('Reference results frozen:',len(actual));return
 expected=json.loads((R/'expected_results.json').read_text());compare(expected,actual)
 print('Verified all',len(actual),'result files against frozen numerical results (timings excluded).')
if __name__=='__main__':main()
