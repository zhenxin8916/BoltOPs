from pathlib import Path
import subprocess,os,json,concurrent.futures
root=Path(__file__).parent
work='/zhenxin/worktrees/BoltOPs-gemm1-config-20261115'
def run(pair):
 kind,bn=pair;d=root/f'top-{kind}-bn{bn}';d.mkdir(exist_ok=False)
 env=dict(os.environ,PYTHONPATH='/zhenxin/temp/moe-fullgrid-tune-20261112/mls-bps1/triton/python:'+work,TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',GPU_ARGS=str(['--gpu-clock=1.8GHz']),GPU_DFLAGS='["StatLog","Perf"]',PMD_PATH='/opt/rocm/pmd',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync')
 (d/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k in ['GPU_CHIP','GPU_ARGS','GPU_DFLAGS','PYTHONPATH','TRITON_HIP_CLANG_PATH','PYTHONHASHSEED']},indent=2))
 with (d/'run.log').open('w') as log:p=subprocess.run(['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(root/'probe_top.py'),kind,str(bn)],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=21600)
 (d/'exit-code.txt').write_text(str(p.returncode));print(kind,bn,'exit',p.returncode,flush=True)
 if p.returncode:print((d/'run.log').read_text()[-2500:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,[(kind,bn) for kind in ('fp8','int8') for bn in (128,256)]))
