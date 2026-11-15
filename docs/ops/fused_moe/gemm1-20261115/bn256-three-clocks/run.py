from pathlib import Path
import subprocess,os,json,sys,concurrent.futures
root=Path(__file__).parent
jobs=[tuple(x.split(':')) for x in sys.argv[1:]]
def run(job):
 version,gc=job;d=root/f'{version}-gc{round(float(gc)*1000)}';d.mkdir(exist_ok=False)
 env=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',GPU_ARGS=str(['--gpu-clock='+gc+'GHz']),GPU_DFLAGS='["StatLog","IT0","Perf"]',PMD_PATH='/opt/rocm/pmd',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync',TRITON_CACHE_DIR=str(d/'cache'),GEMM1_SOURCE=str(root/(version+'.py')),GEMM1_GC_GHZ=gc)
 (d/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('GEMM1','GPU_','TRITON_')) or k=='PYTHONPATH'},indent=2))
 with (d/'run.log').open('w') as log:r=subprocess.run(['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(root/'probe.py'),'fp8','256'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=21600)
 (d/'exit-code.txt').write_text(str(r.returncode));print(d.name,r.returncode,flush=True)
 if r.returncode:print((d/'run.log').read_text()[-2500:])
 return r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
assert not any(codes),codes
