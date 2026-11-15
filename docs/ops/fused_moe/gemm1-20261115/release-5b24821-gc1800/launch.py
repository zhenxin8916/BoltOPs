from pathlib import Path
import subprocess,os,json,concurrent.futures,hashlib
root=Path(__file__).parent
triton='/zhenxin/worktrees/triton-feature-20261102'
repo='/zhenxin/BoltOPs'
assert subprocess.check_output(['git','-C',triton,'rev-parse','HEAD'],text=True).strip()=='5b24821ee57eeb7560074cc7979ce7b59604079f'
assert not subprocess.check_output(['git','-C',triton,'status','--porcelain'],text=True).strip()
def run(pair):
 stage,bn,bk=pair;d=root/f'{stage}-fp8-bn{bn}';d.mkdir(exist_ok=False)
 env=dict(os.environ,PYTHONPATH=triton+'/python:'+repo,TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',GPU_ARGS=str(['--gpu-clock=1.8GHz']),GPU_DFLAGS='["StatLog","Perf"]',PMD_PATH='/opt/rocm/pmd',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync',TRITON_CACHE_DIR=str(d/'cache'))
 cmd=['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python']
 cmd+= [str(root/'probe_top.py'),'fp8',str(bn)] if stage=='top' else [str(root/f'probe_bottom_{bn}.py'),'--bn',str(bn),'--bk',str(bk),'--tiles','1536' if bn==256 else '384']
 (d/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k in ['GPU_CHIP','GPU_ARGS','GPU_DFLAGS','PYTHONPATH','TRITON_HIP_CLANG_PATH','TRITON_CACHE_DIR']},indent=2))
 with (d/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=21600)
 (d/'exit-code.txt').write_text(str(r.returncode));print(stage,bn,'exit',r.returncode,flush=True)
 if r.returncode:print((d/'run.log').read_text()[-2500:],flush=True)
 return r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,[('top',128,128),('top',256,128)]))
assert not any(results),results
