import argparse,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument("kind",choices=("fp8","int8"));p.add_argument("--warm",action="store_true");a=p.parse_args()
for clock in ((1800,) if a.warm else (1800,2200,1500)):
    suffix = "-warm" if a.warm else ""
    directory=root/f"{a.kind}{suffix}-gc{clock}";directory.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH="/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs",TRITON_BACKENDS_IN_TREE="1",
             PMD_PATH="/opt/rocm/pmd",GPU_CHIP="sb",GPU_ARGS=json.dumps([f"--gpu-clock={clock/1000:g}GHz"]),
             GPU_DFLAGS=json.dumps(["StatLog","IT0","Perf"]),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
             TRITON_HIP_CLANG_PATH="/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync")
    command=["setarch","x86_64","-R","/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python",str(root/"run.py"),"--kind",a.kind]
    if a.warm:command.append("--warm")
    with (directory/"run.log").open("w") as log:
        result=subprocess.run(command,env=env,cwd=directory,stdout=log,stderr=subprocess.STDOUT)
    (directory/"exit-code.txt").write_text(str(result.returncode)+"\n")
    print(directory.name,result.returncode,flush=True)
    if result.returncode:sys.exit(result.returncode)
