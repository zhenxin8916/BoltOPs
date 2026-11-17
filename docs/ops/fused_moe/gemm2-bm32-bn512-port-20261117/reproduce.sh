#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../../.." && pwd)
work=${1:?Usage: reproduce.sh /absolute/output/directory}
mkdir -p "$work"
work=$(cd -- "$work" && pwd)
triton_root=${TRITON_ROOT:-/zhenxin/worktrees/triton-feature-20261102}
python_bin=${PYTHON_BIN:-/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python}
export PYTHONPATH="$triton_root/python:$repo${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_BACKENDS_IN_TREE=1 GPU_CHIP=sb PMD_PATH=${PMD_PATH:-/opt/rocm/pmd}
export TRITON_HIP_CLANG_PATH="$here/clang-control-sync"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export BM_INPUT_ROOT="$work/inputs" BENCH_BM=32
export GPU_DFLAGS='["StatLog","TT0","IT0","Perf"]'
export GPU_ARGS='["--gpu-clock=1.8GHz"]'
if [[ ! -f "$BM_INPUT_ROOT/frozen-fp8.pt" ]]; then
  (cd "$work" && "$python_bin" "$here/run.py" --freeze)
fi
for frequency in 2.2 1.8 1.5; do
  name=gc${frequency/./}
  mkdir -p "$work/$name"
  export GPU_ARGS="[\"--gpu-clock=${frequency}GHz\"]"
  export TRITON_CACHE_DIR="$work/$name/cache"
  (cd "$work/$name" && setarch x86_64 -R "$python_bin" "$here/run.py" --kind fp8 >run.log 2>&1)
  "$python_bin" "$here/analyze.py" "$work/$name"
done
