# BM64 / BN512 / BK256 GEMM1 integration

Source: `86d3db7` from `BoltOPs-gemm1-medium-20261114`. The production kernel is now one file, with the common GEMM1 launch/query interface, CK sorting, FP8 E4M3FN/E5M2/mixed inputs, INT8 accumulation, BF16/FP16 sorted output and independent A/B/C address-width constants. CPU selection retains the source's medium-M performance range, `512 <= M <= 2048`; unsupported inputs fall back to ordinary Triton.

## Pipeline and changes

```text
P0: B-left0 -> B-right0 -> B-left1 -> B-right1      shared three-slot FIFO
P1:         A[k]                                  two A slots; EB3/EB4 issue grant
C0: read A + B-left0 -> MMAC -> retained A + B-left1 -> MMAC
C1: read A + B-right0 -> MMAC -> retained A + B-right1 -> MMAC
```

Each A slot is 16 KiB and each B slot 32 KiB: exactly 128 KiB LDS. A data remains in registers across the two N128 panels. The final K tile returns A after the first panel's complete read; ordinary K tiles return it after the second panel. Routes/expert IDs alias retired A0 and channel scales alias retired A1, with the original EMPTY/EB5 ownership preserved. B always uses MLS. Instruction-cache warmup defaults off; FP8's active zero-BF16 MMAC accumulator initialization is separate and remains enabled.

This required more than concatenating source files. The original kernel also failed the current compiler's SGPR no-spill check. During the first comparison its generated metadata reported 12 SGPR spills; the retained rejection log is in `source-compiler-regression/`. The integration uses ordinary global loads for routes/scales and replaces three live slot/phase tokens with one bounded ticket:

```text
ticket: 0 1 2 3 4 5 0 ...
slot:   0 1 2 0 1 2 0 ...
phase:  0 0 0 1 1 1 0 ...
P0 advances by 1; C0/C1 advance by 2, starting at 0/1.
```

Conditional subtraction keeps tickets in 0..5. P0/C0/C1 each matched the original `_fifo3_rotate` slot and phase sequence for 100,000 steps. FIFO order, READY/EMPTY counts and issue grants are unchanged. Wide A copies use packed dword loads/stores into the same layout; the narrow path remains buffer-to-LDS. B folds expert/N into a 64-bit base before local MLS addressing. C widens the row base when requested.

## Numerical and resource validation

`test/fused_moe/test_channel_gemm1_bm64_bn512_equivalence.py`: **6 passed**. Every case compares all 65,536 outputs bitwise against both CPU and a direct ordinary Triton kernel, with fixed seed/routing. Coverage includes CK rank1 and padded tokens, a GPU work counter smaller than capacity, E4M3FN/E5M2/mixed FP8, INT8, BF16/FP16, forced wide A/B/C and non-unit scale strides. An additional **2 passed** run includes a real `_try_gluon_channel_gemm1` launch; only the discovered CU count is set to 1, while dispatch, compilation and GPU execution remain real.

The successful emitted variants from numerical, framework and performance runs were independently audited: **20 assemblies, zero SGPR spills, zero VGPR spills, zero private scratch and no scratch instructions**. See `audit.json` and the archived performance assemblies/HSACO.

Three-frequency CU0 IT traces verify:

- Four startup waves each initialize IDs 0..15 in order, with A EMPTY4/5 count8 and other IDs count4.
- All 16 waves begin their try-wait sequence with a nonzero ID.
- Every wait ID is within 0..15, with no consecutive repeated try-wait ID per wave.

These are trace checks for the tested two-output workload, alongside numerical validation; they are not a proof for arbitrary future pipeline changes.

## Performance

M1024 / E256 / actual GEMM1 output N1024 / K2048 / topk8, seed3101, grid48, two output tiles per persistent TG: **96 BN512 tiles**. The `prepare_cpu` case uses `N=512` because GEMM1 doubles that intermediate dimension. All three runs load the same frozen CPU tensors and routing. Each run checks **1,580,032 valid elements**, padded rows and untouched output regions: zero errors.

| GC | Dispatch time (μs) | CU0 full-span MMAC | Hot K-loop FP8 MMAC |
|---|---:|---:|---:|
| 2.2 GHz | 67.961984 | 23.766652% | 26.281241% |
| 1.8 GHz | 69.284272 | 28.507549% | 31.870182% |
| 1.5 GHz | 71.671816 | 33.702086% | 38.104293% |

Time comes from the HYGPERF XCD dispatch section. CU0 efficiency uses the union of active MMAC issue intervals per SIMD, divided by CU0's own first-wave-start through last-wave-end span, including startup and tail. The model's `mmop-f16-i8-f8=8` supplies an eight-cycle interval for FP8 and BF16. Per CU, raw MMOP count **16,512 = 16,384 mathematical FP8 + 128 active BF16 zero-initialization instructions**. Full-span efficiency includes both families; the JSON also provides the FP8-only ratio. Hot K-loop efficiency measures only mathematical FP8 instructions. The denominator is not the whole dispatch window.

Triton is clean `5b24821ee57eeb7560074cc7979ce7b59604079f`; LLVM is `89bee503d8e9d0bdcfbb4d97e204d5e346f02550`, with the matrix-store source-VGPR hazard protection. Clang and linker were frozen before final tests; their hashes and the control-only ebarrier flag are in `provenance.json`. Three frequencies have identical `.text` SHA256 `d0a9244a7a36fef12caaaccb5002c77c13ae20d6d3b64b8a40998de7c7d5838b` and identical input hashes. Earlier source-performance records use different workloads/toolchains and are not a direct performance comparison.

## Reproduction and evidence

With the gfx946 environment and compiler described in `provenance.json`:

```bash
RUN_GLUON_GEMM1_BM64_BN512_EQUIVALENCE=1 python -m pytest -q -s \
  test/fused_moe/test_channel_gemm1_bm64_bn512_equivalence.py
```

`probe.py`, `run_perf.sh`, `analyze.py` and `perf_dispatch_span.py` retain the local performance commands and parsing. The fixed 516-MiB input fixture and full IT traces remain under `/home/zhenx/temp/bm64-gemm1-port-20261116/`; their hashes/paths are recorded, rather than adding large temporary data to Git. `gc1500`, `gc1800` and `gc2200` contain the small reproducibility artifacts: launch/result/metrics, assembly, LLVM IR, HSACO and CU0 perf. Public CSV references are in `public-csv-records.json`.

All results here are PMD validation. They do not establish real-card reliability, including the separately investigated rare BM128 GEMM1 BN512 hardware failures.

Generated assembly/LLVM IR and retained logs are stored as lossless `.gz` files; `archive-format.json` records their uncompressed SHA256 values. Raw originals remain in the recorded temporary run directories.
