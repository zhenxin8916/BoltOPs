import os,sys,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;d=Path(sys.argv[1]);d.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',PMD_PATH='/opt/rocm/pmd',GPU_CHIP='sb',GPU_ARGS=json.dumps(['--gpu-clock=1.8GHz']),GPU_DFLAGS=json.dumps(['StatLog','IT0','Perf']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync',TRITON_CACHE_DIR=str(d/'cache'))
cmd=['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(r/'check.py'),*sys.argv[2:]]
with (d/'run.log').open('w') as f:p=subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
(d/'exit-code.txt').write_text(str(p.returncode));print(d.name,p.returncode,flush=True);print((d/'run.log').read_text()[-2000:]);sys.exit(p.returncode)
