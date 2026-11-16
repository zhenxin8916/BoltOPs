import os,sys,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
case=Path(sys.argv[1]).resolve();bm=sys.argv[2] if len(sys.argv)>2 else '64'
env=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',PMD_PATH='/opt/rocm/pmd',GPU_CHIP='sb',GPU_ARGS=json.dumps(['--gpu-clock=1.8GHz']),GPU_DFLAGS=os.environ.get('BENCH_DFLAGS',json.dumps(['StatLog','IT0','Perf'])),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync',TRITON_CACHE_DIR=str(case/'cache'),TRITON_ALWAYS_COMPILE='1',BM_INPUT_ROOT=str(r/('inputs-bm'+bm+'-seed3209')),BENCH_BM=bm)
Path(env['BM_INPUT_ROOT']).mkdir(exist_ok=True)
cmd=['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(r/'run-seed3209.py')]+sys.argv[3:]
with (case/'run.log').open('w') as f:p=subprocess.run(cmd,cwd=case,env=env,stdout=f,stderr=subprocess.STDOUT)
(case/'exit-code.txt').write_text(str(p.returncode));print(case.name,p.returncode,flush=True);print((case/'run.log').read_text()[-3500:]);sys.exit(p.returncode)
