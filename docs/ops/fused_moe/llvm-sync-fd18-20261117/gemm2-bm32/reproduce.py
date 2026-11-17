"""Replay old/new compilers on the archived fixed-input benchmark."""
import argparse, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--old-clang',required=True);p.add_argument('--new-clang',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
inputs=Path(a.inputs).resolve();inputs.mkdir(parents=True,exist_ok=True)
repo=here.parents[4];triton=Path(os.environ.get('TRITON_ROOT','/zhenxin/worktrees/triton-feature-20261102'))
env=dict(os.environ,PYTHONPATH=str(triton/'python')+':'+str(repo),TRITON_BACKENDS_IN_TREE='1',GPU_CHIP='sb',PMD_PATH=os.environ.get('PMD_PATH','/opt/rocm/pmd'),GPU_ARGS='["--gpu-clock=1.8GHz"]',GPU_DFLAGS='["StatLog","TT0","IT0","Perf"]',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',BENCH_BM='32',BM_INPUT_ROOT=str(inputs),KERNEL_SOURCE=str(here/'kernel_snapshot.py'))
if not (inputs/'frozen-fp8.pt').exists():subprocess.run([sys.executable,str(here/'run.py'),'--freeze'],cwd=out,env=env,check=True)
for name,raw in [('old',a.old_clang),('new',a.new_clang)]:
 clang=Path(raw).resolve();assert clang.exists() and (clang.parent/'ld.lld').exists()
 case=out/name;case.mkdir(exist_ok=True);wrapper=case/'clang-control-sync';wrapper.write_text('#!/bin/sh\nexec "'+str(clang)+'" -mllvm -hcu-control-only-ebarrier-sync "$@"\n');wrapper.chmod(0o755)
 runenv=dict(env,TRITON_HIP_CLANG_PATH=str(wrapper),TRITON_CACHE_DIR=str(case/'cache'))
 with (case/'run.log').open('w') as log:subprocess.run(['setarch','x86_64','-R',sys.executable,str(here/'run.py'),'--kind','fp8'],cwd=case,env=runenv,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([sys.executable,str(here/'analyze.py'),str(case)],check=True)
