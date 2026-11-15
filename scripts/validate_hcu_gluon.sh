#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Build and validate the paired worktrees inside zx-perf-node60 only.
set -euo pipefail

if [[ ! -d /zhenxin/worktrees ]]; then
  echo "Run inside zx-perf-node60, where /home/zhenx is mounted at /zhenxin." >&2
  exit 2
fi

TRITON_SOURCE=${TRITON_SOURCE:-/zhenxin/worktrees/triton-feature-20261102}
BOLTOPS_SOURCE=${BOLTOPS_SOURCE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
GLUON_ENV=${GLUON_ENV:-/zhenxin/temp/hcu-gluon-upstream-venv-20261103}
GLUON_LOG_DIR=${GLUON_LOG_DIR:-/zhenxin/temp/hcu-gluon-validation-20261103}
export LLVM_SYSPATH=${LLVM_SYSPATH:-/zhenxin/llvm-triton-staging/build}
# Pin the matching MLIR/LLVM package together, including on incremental rebuilds.
# The older llvm-188a510c binary package lacks this baseline's FP4 ROCDL ops.
export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DLLVM_DIR=$LLVM_SYSPATH/lib/cmake/llvm -DMLIR_DIR=$LLVM_SYSPATH/lib/cmake/mlir -DLLD_DIR=$LLVM_SYSPATH/lib/cmake/lld"
export MAX_JOBS=${MAX_JOBS:-8}
export TRITON_OFFLINE_BUILD=1
export TRITON_BUILD_PROTON=OFF
export GPU_CHIP=sb
export PMD_PATH=${PMD_PATH:-/opt/rocm/pmd}
export GPU_DFLAGS=${GPU_DFLAGS:-"['StatLog', 'TT', 'Perf']"}
# The deployed September clang drops the VMEM wait for asyncmark-based LDS
# transfers. Use the existing October backend, which preserves this dependency.
export TRITON_HIP_CLANG_PATH=${TRITON_HIP_CLANG_PATH:-/zhenxin/llvm-project/build-dtk-llvm-6.3.3/bin/clang-18}

mkdir -p "$GLUON_LOG_DIR"
if [[ ! -f "$GLUON_ENV/bin/activate" ]]; then
  python3 -m venv --system-site-packages "$GLUON_ENV"
fi
source "$GLUON_ENV/bin/activate"

# BoltOPs uses PEP 621 metadata and requires setuptools>=77.
# These installs are confined to the activated validation venv.
python -m pip install "setuptools>=77" wheel

# Isolated editable installations; preserve the existing global Triton install.
python -m pip install --no-deps --no-build-isolation -e "$TRITON_SOURCE" 2>&1 | tee "$GLUON_LOG_DIR/build-triton.log"
python -m pip install --no-deps --no-build-isolation -e "$BOLTOPS_SOURCE" 2>&1 | tee "$GLUON_LOG_DIR/build-boltops.log"

python - "$TRITON_SOURCE" "$BOLTOPS_SOURCE" <<'PY'
import pathlib
import sys
import triton
import boltops
expected_triton = pathlib.Path(sys.argv[1]).resolve() / 'python/triton'
expected_boltops = pathlib.Path(sys.argv[2]).resolve() / 'boltops'
assert pathlib.Path(triton.__file__).resolve().is_relative_to(expected_triton), triton.__file__
assert pathlib.Path(boltops.__file__).resolve().is_relative_to(expected_boltops), boltops.__file__
print('Using Triton:', triton.__file__)
print('Using BoltOPs:', boltops.__file__)
PY

python -m pytest -q \
  "$TRITON_SOURCE/python/test/unit/language/test_hcu_gluon_pipeline.py" -k test_hcu_pipeline \
  --junitxml="$GLUON_LOG_DIR/hcu-api-tests.xml" 2>&1 | tee "$GLUON_LOG_DIR/hcu-api-tests.log"

python -m pytest -q \
  "$TRITON_SOURCE/third_party/hcu/test/test_scaled_compile.py" \
  "$BOLTOPS_SOURCE/test/fused_moe/test_gluon_fused_experts.py" \
  "$BOLTOPS_SOURCE/test/fused_moe/test_channel_gemm1_persistent.py" \
  "$BOLTOPS_SOURCE/test/tools/test_gluon_moe_stats.py" \
  --junitxml="$GLUON_LOG_DIR/numerical-tests.xml" 2>&1 | tee "$GLUON_LOG_DIR/numerical-tests.log"

# Single-TG channel performance probe after numerical validation:
# python -m boltops.tools.gluon_moe_gemm1 --case qwen35-large-m4096 \
#   --recipe fp8_channel --modes wdra --out /zhenxin/temp/gluon-channel-m4096
