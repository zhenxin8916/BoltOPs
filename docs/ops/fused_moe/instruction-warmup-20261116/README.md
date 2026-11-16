# Instruction warmup validation

Warmup fetches real consumer MMAC blocks once with EXEC=0 before tile consumption. It remains opt-in through the `INSN_WARMUP` constexpr, default false. GEMM1 already implements this path; this change adds it to the four GEMM2 kernels. No additional LDS storage or barrier IDs are allocated. An opaque scalar loop start prevents cloning named assembly labels.

PMD `GPU_CHIP=sb`, 1.8 GHz, grid 48, two output tiles per persistent TG. Routing is fixed; each off/on pair uses identical input manifests. Times cover GEMM dispatch only, excluding initialization and output checking. One model run per variant; these are not repeated hardware measurements.

| Kernel | Input | Off (us) | On (us) | Time reduction |
|---|---|---:|---:|---:|
| GEMM1 BM128 BN256 BK128 | FP8, M4096/N1024/K2048 | 41.603812 | 41.215724 | 0.93% |
| GEMM1 BM128 BN512 BK128 | FP8, M4096/N1024/K2048 | 63.701476 | 62.189156 | 2.37% |
| GEMM2 BM64 BN512 BK64 | FP8, M1024/N2048/K256, one N panel | 20.206708 | 19.413852 | 3.92% |

All listed off/on cases passed bit-exact output checks with zero nonfinite values. All exported GEMM2 shapes/dtypes compiled with zero SGPR/VGPR spills and zero private bytes; GEMM1 final-object audit is included here. GEMM2 BM64 BN256 has compile coverage only.

Initial GEMM2 BM128 FP8 probes (before the final explicit direct-MFMA default-off branch) passed numerically but showed no useful benefit: BN256 16.512088 -> 16.580476 us; BN512 22.188848 -> 22.147704 us. They are retained as candidate evidence, not final-source performance claims. A final-source INT8 BM128 BN256 correctness comparison also passed (0/2,263,040 errors each); INT8 performance CSVs are intentionally removed. Future INT8 performance estimates may reference FP8 and must be labeled estimates.

A reduced warmup candidate stalled in GEMM2 and was discarded. The retained implementation is the fully visited block version, with the opaque loop start preserved.

Local raw logs, source snapshots, HSACO and CU0 SQTT perf:
- `/home/zhenx/temp/gemm1-warmup-20261116/pilot-runs/{off,on}/<case>/m5out/0/1/`
- `/home/zhenx/temp/gemm2-warmup-20261116/bm64-agent/{off_direct,on_direct}/m5out/0/1/`
- GEMM2 full compile metadata: `/home/zhenx/temp/gemm2-warmup-20261116/{off_direct,on_direct}/cases/`.

Triton: `5b24821ee57eeb7560074cc7979ce7b59604079f`. LLVM: `fd18eddbaf4a662ae5ce86c516d28e5b16952513`. Per-case compiler binary hashes and options are preserved in local compile.json files.
