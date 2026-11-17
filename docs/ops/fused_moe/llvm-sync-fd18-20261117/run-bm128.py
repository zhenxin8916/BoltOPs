import concurrent.futures, json, os, subprocess
from pathlib import Path
root=Path('/zhenxin/temp/llvm-sync-fd18-20261117')
repo=Path('/zhenxin/BoltOPs')
probe=repo/'docs/ops/fused_moe/llvm-fd18eddb-20261115/probe.py'
for version, binary in [('old',root/'old-bin/clang-18'),('new',root/'new-bin/clang-18')]:
    wrapper=root/f'{version}-control'
    wrapper.write_text(f'#!/bin/sh\nexec {binary} -mllvm -hcu-control-only-ebarrier-sync "$@"\n')
    wrapper.chmod(0o755)
jobs=[(v,s,n) for v in ['old','new'] for s,n in [('top',128),('top',256),('top',512),('bottom',256),('bottom',512)]]
def run(job):
    v,s,n=job; d=root/'bm128'/f'{v}-{s}-bn{n}';d.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',PMD_PATH='/opt/rocm/pmd',GPU_ARGS='["--gpu-clock=1.8GHz"]',GPU_DFLAGS='["StatLog","IT0","TT0","Perf"]',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH=str(root/f'{v}-control'),TRITON_CACHE_DIR=str(d/'cache'))
    with (d/'run.log').open('w') as log:
        try: code=subprocess.run(['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(probe),s,str(n),'fp8'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
        except subprocess.TimeoutExpired: code=124
    (d/'exit-code.txt').write_text(str(code)+'\n');print(d.name,code,flush=True)
    return dict(name=d.name,code=code)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,jobs))
(root/'bm128-results.json').write_text(json.dumps(results,indent=2))
