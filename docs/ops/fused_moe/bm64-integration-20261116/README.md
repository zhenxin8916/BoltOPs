# BM64 channel MoE integration

Three single-file kernels are connected to the existing Gluon-first / ordinary
Triton fallback. The sorting configuration owns BM; each kernel selects BN/BK.
Both GEMM2's sorted-activation bridge and its launch use the same BM selection.

| Stage | BM/BN/BK | Source commit | Verification |
|---|---|---|---|
| GEMM1 | 64/512/256 | `86d3db7` | [Evidence](../gemm1-bm64-bn512-integration-20261116/) |
| GEMM2 | 64/256/256 | `23c77c41f1f4cab3652700da33b2e29ee6e67d7e` | [Evidence](../bm64-bn256-integration-20261116/) |
| GEMM2 | 64/512/64 | `98a1e1e` | [Evidence](../bm64-gemm2-bn512-port-20261116/) |

The GEMM1 port inlines its old kernel dependencies. A six-state B ticket replaces
three live FIFO tokens, and metadata/scales use ordinary global loads to avoid
SGPR spills with the current compiler. Slot/phase ordering is unchanged. The
host supplies the known persistent grid as a constexpr loop step. GEMM2 BN256
adds wide C scatter and full-width B-scale addressing, including large channel
strides. GEMM2 BN512 retains its existing algorithm.

Compatibility includes CK sorting, FP8 and INT8 accumulation, BF16/FP16 output,
independent constexpr A/B/C address widths and disabled instruction warmup by
default. Queries reject unsupported cases before compilation. Forced wide-path
tests use small allocations; they are not physical allocations beyond 4 GiB.

The three kernel owners supplied numerical/assembly/trace evidence, followed by
cross-review of FIFO generations, barrier credits, address expressions and host
dispatch. CPU dispatch/CSV regressions cover BM64 and existing BM128 selections,
caller CU budgets, GPU work counts, independent address flags and large scale
strides. Numerical tests compare identical inputs against ordinary non-Gluon
Triton; PMD evidence does not certify hardware numerical behavior.

## Performance records

BM64 uses **M1024, E256, top-k8, 48 CUs, two tiles per persistent TG**. Frozen
inputs are reused across 2.2/1.8/1.5 GHz. GEMM2's two tiles use independent M
blocks and a bounded N traversal; full-N correctness is checked separately.
The public top/bottom CSVs contain all three clocks and CU0 whole-kernel MMAC
percentages, including startup and tail. The denominator is CU0's own wave
lifetime, not the dispatch window. Two BN512 tiles contain twice the output
work of two BN256 tiles; their elapsed times are not equal-work comparisons.

GEMM1 uses LLVM `89bee503` with matrix-store VGPR hazard protection. GEMM2's
performance objects predate that rebuild and retain LLVM `fd18eddb` provenance.
The BN256 final-source recompile has identical `.text` to its measured objects.
Binary hashes and the rebuild boundary are recorded with each kernel's evidence.
