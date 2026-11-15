import concurrent.futures,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
cases=[('top',bn,kind) for bn in (512,128,256) for kind in ('fp8','int8')]+[('bottom',bn,kind) for bn in (256,512) for kind in ('fp8','int8')]
jobs=[(version,stage,bn,kind,gc,False) for version,gcs in [('new',(1800,2200,1500)),('previous',(1800,))] for stage,bn,kind in cases for gc in gcs]
jobs += [(v,'top',512,'fp8',1800,True) for v in ('new','previous')]
def run(job):
 version,stage,bn,kind,gc,warm=job;name=f'{version}-{stage}-bn{bn}-{kind}-gc{gc}'+('-warm' if warm else '')
 d=root/'perf'/name;d.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',PMD_PATH='/opt/rocm/pmd',GPU_ARGS=json.dumps([f'--gpu-clock={gc/1000:g}GHz']),GPU_DFLAGS='["StatLog","IT0","Perf"]',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH=str(root/version/'clang-control-sync'),TRITON_CACHE_DIR=str(d/'cache'))
 cmd=['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(root/'probe.py'),stage,str(bn),kind]+(['--warm'] if warm else [])
 with (d/'run.log').open('w') as log:
  try:code=subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
  except subprocess.TimeoutExpired:code=124
 (d/'exit-code.txt').write_text(str(code)+'\n');print(name,code,flush=True)
 return name,code
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:results=list(pool.map(run,jobs))
(root/'perf-results.json').write_text(json.dumps(dict(results),indent=2)+'\n')
