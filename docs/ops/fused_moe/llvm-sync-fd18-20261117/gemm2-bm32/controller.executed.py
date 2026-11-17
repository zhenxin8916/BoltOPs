import os,sys,json,subprocess,hashlib,concurrent.futures
from pathlib import Path
r=Path(__file__).parent;py='/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python';repo=Path('/zhenxin/BoltOPs')
base=dict(os.environ,PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:'+str(repo),TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',PMD_PATH='/opt/rocm/pmd',GPU_DFLAGS='["StatLog","TT0","IT0","Perf"]',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync',BENCH_BM='32',BM_INPUT_ROOT='/zhenxin/temp/moe-gemm2-bm32-M128-BN512-20261116/inputs-bm32',KERNEL_SOURCE=str(r/'kernel_snapshot.py'),RUN_GLUON_GEMM2_BM32_BN512_EQUIVALENCE='1')
jobs=[('perf','old'),('perf','new')]
def run(job):
 typ,name=job;d=r/name;d.mkdir(exist_ok=True)
 env=dict(base,GPU_ARGS='["--gpu-clock=1.8GHz"]',TRITON_CACHE_DIR=str(d/'cache'),TRITON_HIP_CLANG_PATH=str(r/('clang-'+name+'-control-sync')))
 cmd=['setarch','x86_64','-R',py]
 if typ=='perf':cmd +=[str(r/'run.py'),'--kind','fp8']
 else:cmd += ['-m','pytest',str(repo/'test/fused_moe/test_channel_gemm2_bm32_bn512_equivalence.py')+'::test_gluon_gemm2_matches_triton['+name+']','-v','-s','--junitxml='+str(d/'junit.xml')]
 with (d/'run.log').open('w') as f:
  try:code=subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=2400).returncode
  except subprocess.TimeoutExpired:code=124
 (d/'exit-code.txt').write_text(str(code)+'\n');out=dict(type=typ,name=name,exit_code=code,path=str(d));print(json.dumps(out),flush=True);return out
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,jobs))
(r/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
