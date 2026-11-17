"""Launch the three clocks with identical inputs and separate PMD output roots."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--clang", required=True, help="frozen clang/control wrapper")
args = parser.parse_args()
probe = Path(__file__).with_name("probe.py")
repo = probe.resolve().parents[4]
def run(gc):
    directory = args.output.resolve() / f"gc{gc}-final"
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GPU_CHIP="sb", PMD_PATH=os.environ.get("PMD_PATH", "/opt/rocm/pmd"),
               GPU_DFLAGS=json.dumps(["StatLog", "IT0", "TT0", "Perf"]),
               GPU_ARGS=json.dumps([f"--gpu-clock={gc / 1000}GHz"]),
               GC_GHZ=str(gc / 1000), TRITON_BACKENDS_IN_TREE="1",
               TRITON_HIP_CLANG_PATH=args.clang,
               TRITON_CACHE_DIR=str(directory / "cache"),
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (directory / "run.log").open("w") as log:
        code = subprocess.run(["setarch", "x86_64", "-R", sys.executable, str(probe)],
                              cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (directory / "exit-code.txt").write_text(str(code) + "\n")
    print(gc, code, flush=True)
    return code
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    codes = list(pool.map(run, [2200, 1800, 1500]))
raise SystemExit(any(codes))
