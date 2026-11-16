"""Run frozen M1024 inputs, 48 TGs, two M64xN512 outputs per TG."""
import json, os, subprocess, sys
from pathlib import Path
clock, kind = sys.argv[1:3]
root = Path('/zhenxin/temp/bm64-gemm2-bn512-port-20261116')
case = root / f'{kind}-gc{clock}'
case.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH='/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs', TRITON_BACKENDS_IN_TREE='1', GPU_CHIP='sb', PMD_PATH='/opt/rocm/pmd', GPU_ARGS=json.dumps([f'--gpu-clock={int(clock)/1000}GHz']), GPU_DFLAGS=json.dumps(['StatLog','TT0','IT0','Perf']), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TRITON_HIP_CLANG_PATH='/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync', TRITON_CACHE_DIR=str(case/'cache'), BM_INPUT_ROOT=str(root/'inputs'))
with (case/'run.log').open('w') as log:
    code = subprocess.call(['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python', str(Path(__file__).with_name('run.py')), '--kind', kind], cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
(case/'exit-code.txt').write_text(str(code))
print(case, code, flush=True)
sys.exit(code)
