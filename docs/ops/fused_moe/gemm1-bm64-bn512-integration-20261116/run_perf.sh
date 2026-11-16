#!/bin/bash
set -eu
suffix=$1
freq=$2
docker exec -w /zhenxin/temp/bm64-gemm1-port-20261116/$suffix -e PYTHONPATH=/zhenxin/worktrees/triton-feature-20261102/python:/zhenxin/BoltOPs -e TRITON_BACKENDS_IN_TREE=1 -e GPU_CHIP=sb -e PMD_PATH=/opt/rocm/pmd -e "GPU_ARGS=[\"--gpu-clock=${freq}GHz\"]" -e 'GPU_DFLAGS=["StatLog","IT0","TT0","Perf"]' -e TRITON_HIP_CLANG_PATH=/zhenxin/temp/bm64-gemm1-port-20261116/clang-control-frozen -e TRITON_CACHE_DIR=/zhenxin/temp/bm64-gemm1-port-20261116/$suffix/cache -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e MOE_BN=512 -e MOE_BK=256 zx-perf-node60 /zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python /zhenxin/temp/bm64-gemm1-port-20261116/perf_probe.py > /home/zhenx/temp/bm64-gemm1-port-20261116/$suffix/run.log 2>&1
