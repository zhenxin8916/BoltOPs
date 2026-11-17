# BM32 independent scale bank, 2026-11-17

Base: 598f0ec (API integration), following tuning baseline 639b287. Single TG per CU is the intended resource policy; two resident TGs are not a requirement.

Fixed workload: GEMM1 M128/N512/K2048, BM32/BN256/BK256, grid48, two logical outputs per persistent TG (96 total), cold input, 1.8 GHz. Two outputs processed sequentially are distinct from two resident hardware TGs.

| Version | Dispatch (us) | Wave active (us) | B active bandwidth (TB/s) | Merged handoff (cycles) | Same-consumer handoff mean/max (cycles) |
|---|---:|---:|---:|---:|---:|
| Baseline 639b287 | 36.580352 | 32.166824 | 1.564706 | 916 | 1441 / 1532 |
| Independent single scale bank + early A return | 36.569788 | 32.156260 | 1.565222 | 849 | 1332.5 / 1432 |

The candidate repeated identically in two PMD runs. Dispatch improves only 0.0289%; this is a small improvement, not achievement of the 1.6 TB/s target. The bandwidth numerator is 48 MiB B bytes. Target wave-active duration is at most 31.457280 us. Merged handoff combines both consumers; individual consumer statistics are included to avoid hiding stalls.

A keeps two 8-KiB slots, B keeps its three 16-KiB FIFO slots. A is released after panel0 LDS reads complete and reused from registers for panel1. A scale bank of 2 KiB is allocated independently instead of aliasing A1: total LDS is 66 KiB. P1 no longer waits for prior scale readers before issuing the next output A1. EB5 count12 protects scale overwrite immediately before the next scale publication. Consumers signal after completing their old scale reads. Producer mutual exclusion, B slot tickets/phases, priorities and disabled consumer ping-pong remain intact.

Actual CU0 traces verified all four producer wave sets and eight consumers: scale overwrite follows prior scale-read completion; next output A issue precedes the scale reuse wait; B ring tickets and returns match; A/B issue windows do not overlap. Both fixed-workload runs had zero numeric, padding and untouched-output errors (101376 checked valid outputs). This is not a claim of full dtype/shape regression coverage. Worktree source compiles to exactly the measured ELF .text (see evidence JSON).

Measured artifacts remain at `/home/zhenx/temp/moe-gemm1-bm32-M128-20261116/`:

- Baseline: `bm32-best-no-pingpong-repeat-output96-gc1800/`.
- Candidate: `bm32-independent-scale-onebank-earlyA-output96-gc1800/`.
- Repeat: `bm32-independent-scale-onebank-earlyA-repeat-output96-gc1800/`.
- Perf basename: `M128_N512_K2048_BM32_BN256_BK256_GRID48_OUTPUT96_GC1800_CU0.perf`.

Compiler/runtime provenance and local-change snapshots are preserved in the adjacent evidence directory. No LLVM or Triton changes were made for this optimization. Compile-only checks are not performance measurements at another frequency.

## Single-resident follow-up and first-quarter early store

The follow-up keeps grid48 and two sequential logical outputs per TG. It does not impose two resident TGs/CU or a 64-KiB shared-memory cap.

| Algorithm | LDS KiB | Dispatch us | B active TB/s | Merged handoff cycles | Per-consumer mean handoff cycles |
|---|---:|---:|---:|---:|---:|
| B3+A2 independent scales (398b787) | 66 | 36.569788 | 1.565221 | 849 | 1332.5 |
| B4+A2 | 82 | 37.005136 | 1.544313 | 946 | not extracted |
| B5+A2 | 98 | 36.980116 | 1.545500 | 1602 | not extracted |
| B5+A3 | 106 | 36.847232 | 1.551832 | 1876 | not extracted |
| Epilogue priority0 | 66 | 36.567564 | 1.565329 | 1269 | 1833.5 |
| Grouped consumer B panels | 66 | 36.587024 | 1.564382 | 1434 | 2333.0 |
| First quarter early store | 66 | 36.551440 | 1.566114 | 677 | 1434.0 |
| Early store after last B read | 66 | 36.576460 | 1.564896 | 582 | 1495.5 |
| Zero-MMAC only on first output | 66 | 36.707676 | 1.558538 | 1228 | 1924.0 |
| Vector output store | 74 | 37.475512 | 1.522342 | 1701 | 2679.5 |

First-quarter early store repeated identically twice: 36.551440 us, 0.0502% faster than 398b787, 0.0790% faster than the original 36.580352-us reference. This is the fastest measured follow-up. It remains below 1.6 TB/s. Each owning consumer writes its completed first N64 quarter after final panel0 MMAC and before reading/computing final panel1. The scale publication wait moves to that first store, with the later scale-read reuse still protected by EB5. The actual CU0 trace confirms the first store precedes final panel1 MMAC on all eight consumers, both outputs; numerical checks pass. The production-source ELF .text equals the measured binary.

There is a tradeoff: merged handoff improves 849 -> 677 cycles, but individual-consumer mean worsens 1332.5 -> 1434 and max worsens 1432 -> 1520 cycles. Thus this is saved as a small total-time improvement on the experimental branch, not a claim that every consumer transition or the full bandwidth goal is solved. Kernel hot-loop MMAC is 16.79% on CU0; no claim of 70% utilization.

Increasing buffers alone did not help. A3+B5 verified slot order, phase and release and almost removed A EMPTY issue gaps (median296 ->4 cycles), yet dispatch worsened. Baseline B READY issue gaps are mostly after all four producer waves have issued their associated MLS: 256 samples, only0.9532% of accumulated gap precedes the last producer issue; median post-issue gap1056 cycles. Gaps include scheduling and completion latency, not pure HBM latency. The scalar/issue timing evidence is in the adjacent JSONs; wait sums across waves are not additive wall time.

Negative trials are not applied to the production source. A syntax failure before model launch and a PMD trace-output open failure before kernel execution are excluded from performance results. The latter was retried in a fresh directory without changing the kernel; the successful run and independent repeat agree. The compile-only source-equivalence run uses gfx946/sb and is not a 2.2-GHz performance result.

Best follow-up perf: [Wavefront perf](/home/zhenx/temp/moe-gemm1-bm32-M128-20261116/bm32-early-first-quarter-store-repeat-output96-gc1800/M128_N512_K2048_BM32_BN256_BK256_GRID48_OUTPUT96_GC1800_CU0.perf). Scope remains the fixed E4M3/BF16 cold workload, not the full dtype/shape suite. No compiler changes.
