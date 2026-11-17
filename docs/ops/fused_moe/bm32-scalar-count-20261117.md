# BM32 single-resident scalar work-count read, 2026-11-17

Implementation: the commit adding this report, parent `8fd70e37`. Exact source and machine-code hashes are in [production-equivalence.json](bm32-scalar-count-20261117/production-equivalence.json). Only the uniform GPU padded-row count changes to native `s_load_dword` with an explicit owned LGKM wait, independently inside each role. Keep 2P2C, four waves per role, static VGPR allocation, A2/B3, producer exclusion, consumer priorities, disabled consumer ping-pong, independent scales and early first-quarter stores. Single resident hardware TG per CU; two sequential logical outputs per TG.

All timings below are PMD, GEMM1 M128/N512/K2048, BM32/BN256/BK256, grid48, 96 logical output tiles, cold B, 1.8 GHz. B active bandwidth is 50,331,648 bytes divided by the global first-to-last wave interval.

| Code | Dispatch μs | Wave-active μs | B active TB/s | CU0 per-consumer handoff mean / max cycles |
|---|---:|---:|---:|---:|
| `8fd70e37` | 36.217284 | 31.803756 | 1.582569 | 1269.5 / 1640 |
| This commit, initial scalar-count experiment | 36.177808 | 31.764280 | 1.584536 | 1123 / 1352 |
| This commit, production-source repeat | 36.177808 | 31.764280 | 1.584536 | 1123 / 1352 |

Improvement is **0.039476 μs / 0.1090%** over `8fd70e37`. This is a small gain. The 1.6-TB/s goal remains unfinished: required active time is 31.457280 μs, another 0.307000 μs below this result. CU0 weighted K-hot MMAC is 16.5488%; whole-active MMAC is 14.3392%. These MMAC measurements describe CU0, not all 48 CUs.

## Handoff and critical CUs

Handoff means each consumer wave's output0 last valid MMAC end to output1 first MMAC. Merged C0/C1 gaps are a different metric: CU0 merged gap increases 141 → 296 cycles, even though individual mean/max improve. Do not present the merged gap as an individual stall reduction.

| CU | `8fd70e37` mean / max | This commit mean / max | Previous / current last valid MMAC end |
|---|---:|---:|---:|
| 0 | 1269.5 / 1640 | 1123 / 1352 | 55745 / 55525 |
| 17 | 1719 / 1896 | 1730 / 2108 | 56669 / 56280 |
| 28 | 1847.5 / 2336 | 1717.5 / 2124 | 56233 / 56379 |

Cycles are relative to the global first wave. CU17 has a worse individual handoff but finishes its work earlier; CU28 has a better handoff but finishes later. Global wave end improves 57201 → 57130 cycles. The optimization does not improve every CU or every transition.

In the retained control's CU17, next-output B/A issue begins at 28616/28780 cycles. C0's next-output B READY wait starts at30224 and reaches the following instruction at31028; A READY then starts at31080 and advances at31372. These instruction gaps include scheduling and are not pure memory latency. Full B6 advances next-output B much earlier in CU28 (28287 → 25931) while its consumer still waits for READY; more slots alone did not fix the transition. The B6 FIFO/phase/return checker passed, including all six slots. CU0 LDS bank conflicts are zero in the retained control.

## Other trials excluded from production

| Algorithm change from `8fd70e37` | LDS KiB | Dispatch μs |
|---|---:|---:|
| A2/B4 | 82 | 36.247864 |
| A2/B5 | 98 | 36.307356 |
| A2/B6 | 114 | 36.196156 |
| A3/B5 | 106 | 36.398540 |
| A4/B4 | 98 | 36.437460 |
| Store-stage priority0 or1 | 66 | 36.243972 |
| Metadata issued in partitions before shared initialization | 66 | 36.273996 |
| Producer work count prefetched first | 66 | 36.254536 |
| Partition initialization plus work-count-first | 66 | 36.266212 |
| Relax consumer scheduling fences | 66 | 36.249532 |
| A2/B6 plus relaxed consumer scheduling | 114 | 36.312916 |

B6 is slightly faster than the control, but slower than the retained scalar-count change. Other trial source changes are not retained in production. [comparisons.json](bm32-scalar-count-20261117/comparisons.json) gives exact cases, source hashes, numerical checks and bandwidth. Failed trace-file opens before kernel execution at tick1668 are excluded; fresh-directory retries succeeded. No PMD execution defect or workaround is claimed.

## Verification and reproduction

- Initial and production-source repeat both check101376 valid CPU-reference values: zero numerical, padding or untouched-region errors. Each CU0 consumer executes512 valid MMACs across both outputs.
- Production-source ELF `.text` equals the initial measured experiment byte-for-byte. Zero scratch, zero SGPR/VGPR spills,82 VGPR; metadata SGPR count56. LDS remains67584 bytes.
- Existing numerical API suite:12/12 pass, no skips/failures/errors, covering E4/E5 and mixed FP8, INT8, FP16/BF16, address-width flags, grouped-M, warmup and K1024. XML accompanies this report.
- Thirteen compile-only resource configurations, including no-counter, compile without spills/scratch. Compile-only PMD initialization may print a default2.2-GHz context; it is not a performance measurement.
- All CU0 producer/consumer waves retain FIFO slot/phase/EMPTY order and initialization credits. Actual A128/B1024 memory instructions; zero A/B overlapping issue windows. Scale overwrite follows prior consumer completion, consumer priority sequences match the control, and first-quarter stores retain their validated placement.
- Embedded Wavefront matches raw TT for CU0, CU16, CU17 and CU28 in the production repeat. Raw stats/IT/TT/ISA/HSACO and input snapshots remain in the measured case directory.

[Production perf, including four traced CUs](/home/zhenx/temp/moe-gemm1-bm32-M128-20261116/bm32-static-scalar-count-production-retry-output96-gc1800/M128_N512_K2048_BM32_BN256_BK256_GRID48_OUTPUT96_GC1800_CU0.perf).

Reproduce in container `zx-perf-node60` with `/zhenxin/temp/moe-gemm1-bm32-M128-20261116/launch.py`, `LIMIT_BLOCKS=48`, `TEST_GRID=48`, `GC_GHZ=1.8`, `EXTRA_TRACE_CUS=16,17,28`, a fresh `CASE` and `VARIANT_SCRIPT` pointing at this commit's BM32 kernel. Launcher uses the frozen clang control wrapper and Triton environment; exact input hashes/arguments are in the adjacent input.json.

No LLVM/Triton changes in this commit. [provenance.json](bm32-scalar-count-20261117/provenance.json) records branches/commits, current clean local status, frozen binary hashes and the original local-change snapshot. The live LLVM checkout has since advanced; performance uses the unchanged frozen clang binary, not a rebuild from the current checkout. The scalar load uses a literal zero offset and does not trigger the documented PMD SGPR-plus-immediate offset defect.
