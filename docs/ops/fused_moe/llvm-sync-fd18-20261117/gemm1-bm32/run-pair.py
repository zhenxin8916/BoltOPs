from pathlib import Path
import os,subprocess,json,concurrent.futures
r=Path(__file__).parent
def run(name):
 d=r/name;d.mkdir(exist_ok=True)
 env=dict(os.environ,PYTHONPATH="/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs",TRITON_BACKENDS_IN_TREE="1",TRITON_CACHE_DIR=str(d/"cache"),TRITON_HIP_CLANG_PATH=str(r/f"clang-{name}-control"),GPU_CHIP="sb",GPU_DFLAGS=json.dumps(["StatLog","IT0","TT0","Perf"]),GPU_ARGS=json.dumps(["--gpu-clock=1.8GHz"]),PMD_PATH="/opt/rocm/pmd",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",GC_GHZ="1.8")
 with (d/"run.log").open("w") as f:code=subprocess.run(["setarch","x86_64","-R","/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python",str(r/"probe.py")],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 (d/"exit-code.txt").write_text(str(code)+"\n");print(name,code,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,["old","new"]))
