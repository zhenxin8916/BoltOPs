# BM64 GEMM2 BN256 integration — 2026-11-16

Source: `23c77c41f1f4cab3652700da33b2e29ee6e67d7e`, one self-contained
BM64/BN256/BK256 kernel. The retained row-persistent algorithm is unchanged.
The integration adds a CPU support query, constexpr C64 scatter, and a 64-bit
B-scale addresses. Wide B scales use 64-bit global loads and an explicit
LDS completion wait; narrow B scales retain direct async buffer-to-LDS copies.
Narrow output addresses keep packed buffer stores.
Warmup is absent; no inline assembly was added.

## Correctness

PMD gfx946, Triton `5b24821ee57eeb7560074cc7979ce7b59604079f`,
The initial LLVM source HEAD was `fd18eddbaf4a662ae5ce86c516d28e5b16952513`.
A concurrent build replaced clang during validation; final source HEAD is
`89bee503d8e9d0bdcfbb4d97e204d5e346f02550`, while its embedded version remains
fd18. Historical binary SHA was not captured, so these two identifiers must
not be presented as interchangeable build evidence. See `provenance.json`.
A forced final-source narrow compile and all three E256 perf code objects have
identical 6364-byte ELF `.text` (SHA256
`52c3b17c0bc556ef7572680c67f3a4a58a585eb2a2691b30b60e841e3400ce5c`).
The final SB64 fix and concurrent rebuild therefore do not change the measured
narrow-path instructions. `final-isa-equivalence.json` records binary hash,
mtime, embedded version, source HEAD, and each compared HSACO.

| Case | Compared elements | mismatches |
| --- | ---: | ---: |
| FP8 / BF16 vs CPU | 197632 | 0 |
| INT8 / BF16 vs CPU | 197632 | 0 |
| FP8 / FP16, A64+B64+C64 vs CPU | 197632 | 0 |
| FP8 / BF16 vs non-Gluon Triton | 197632 | 0 |
| INT8 / BF16 vs non-Gluon Triton | 197632 | 0 |

These full-N cases use ten M blocks, grid 2, two N panels and CK packed routes,
covering five A generations and ten B generations per TG. The non-Gluon
comparison uses the same sorted input/scales/weights with BK128. Both FP8
and INT8 comparisons additionally found **zero bitwise-unequal BF16 values**
(197632 elements each); raw counts are in each triton-comparison.json.
Wide paths were executed on small allocations; this does not claim an allocated
>4 GiB tensor test. Both output dtypes are selected by the C pointer type.

Final assembly: zero SGPR/VGPR spills, zero private segment and zero scratch
instructions for all three standalone correctness variants (`no-spill.json`).
CU0 trace verifies 16 live waves, four per SIMD, zero exec=0 MMAC warmup,
barrier IDs in [0,15], nonzero first waits and no adjacent repeated wait ID.
The full-N generation audit verifies A, B and metadata phases through wrap.
This audits software protocol and traces, not every hardware transaction.

## Performance scope

`performance.csv` initially records an E16 diagnostic, not the shared E256
comparison; its frequency subdirectories are retained as evidence.
M1024/N2048/K256, grid48, topk8 frozen seed3101, each persistent TG computes
96 / 48 = two independent M64 × N256 outputs (`N_TILES_LIMIT=1`). Full-N
correctness is separate. MMAC uses CU0's first wave start to last wave end
including one final cycle; startup and tail are included. Dispatch time is read
from HYGPERF section15. No CPU initialization/checker kernel is timed.

The standardized comparison is `performance-e256.csv`, with E256 and the
same fixed topk8/seed3101. All three frequencies checked 785920 selected
values with zero tolerance failures; untouched outputs stay NaN and hashes
match across frequencies. Every TG still computes two BN256 output tiles.
A BN512 two-tile run performs twice the N work; compare throughput or work
counts rather than raw latency across BN256 and BN512.

| GC GHz | time μs | CU0 whole-lifetime MMAC |
| ---: | ---: | ---: |
| 2.2 | 11.877320 | 11.324302% |
| 1.8 | 13.561396 | 12.644317% |
| 1.5 | 14.897445 | 14.289701% |

Reproduce inside `zx-perf-node60` with the venv Python and `launch.py`:
create a case directory containing this kernel as `kernel.py`; invoke
`launch.py CASE 64 --freeze` once, then `BENCH_GHZ=1.8
BENCH_DFLAGS='["StatLog","IT0","TT0","Perf"]' launch.py CASE 64 --kind fp8`.
The frozen manifest contains every tensor hash; subsequent runs never
regenerate routing. `analyze.py CASE` computes the documented CU0 metric.
The checker uses rtol=0.02 and atol=0.002, not a claim of bitwise identity.

 Input tensors
and raw traces live under `/home/zhenx/temp/bm64-bn256-integration-20261116`.

Generated assembly/LLVM IR and retained logs are stored as lossless `.gz` files; `archive-format.json` records their uncompressed SHA256 values. Raw originals remain in the recorded temporary run directories.
