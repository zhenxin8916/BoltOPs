import concurrent.futures,json,os,subprocess
from pathlib import Path
root=Path('/zhenxin/temp/llvm-sync-fd18-20261117')
probe='/zhenxin/temp/bm64-gemm1-port-20261116/perf_probe.py'
def run(v):
 d=root/'gemm1-bm64'/v;d.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',PMD_PATH='/opt/rocm/pmd',GPU_ARGS='["--gpu-clock=1.8GHz"]',GPU_DFLAGS='["StatLog","IT0","TT0","Perf"]',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH=str(root/f'{v}-control'),TRITON_CACHE_DIR=str(d/'cache'),MOE_BN='512',MOE_BK='256')
 with (d/'run.log').open('w') as log:
  code=subprocess.run(['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',probe],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
 (d/'exit-code.txt').write_text(str(code)+'\n');print(v,code,flush=True);return dict(version=v,code=code)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['old','new']))
(root/'gemm1-bm64/results.json').write_text(json.dumps(results,indent=2))
