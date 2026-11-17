# BM32 GEMM1 update integration

Source: `experiment/moe-bm32-scale-bank-20261117`, commit
`4f60fcf5` (including its independent-scale, early-store, early-A1 and static-VGPR predecessors).
Merged into `dev/gluon-moe-gemm-opt`, preserving the existing GEMM2 and capacity checks.
The 66-KiB allocation replaces the old 64-KiB scale-alias layout; one hardware TG
is resident per CU, with two sequential logical output tiles per TG.

Validation after migration: 14/14 numerical checks passed (12 API comparisons
against ordinary Triton plus two inactive/capacity cases), no skips or failures.
The generated ELF `.text` is byte-identical to the source branch's measured
scalar-count version. The target probe reserves 82 VGPRs, with zero spills and scratch.
The two production files differ from the source branch only in explanatory comments.

Retained performance evidence: 1.8 GHz, M128/N512/K2048, grid48, 96 outputs:
36.177808 us dispatch, 31.764280 us wave active, 1.584536 TB/s B bandwidth;
CU0 whole-active MMAC 14.3392%, K-hot MMAC 16.5488%.
These timings come from the source branch's validated repeat, not a new timing run.
See [full report](../bm32-scalar-count-20261117.md) for exact provenance and scope.
Existing three-frequency CSV rows remain historical measurements of their recorded
kernel; this migration has not remeasured the new code at 2.2 or 1.5 GHz.

Tests and code-equivalence compilation explicitly used the frozen control clang
`/home/zhenx/temp/llvm-ds-read2-wrap-20261115/clang-after-control`;
the concurrent LLVM 7.2.1 build is independent of these validation results.
