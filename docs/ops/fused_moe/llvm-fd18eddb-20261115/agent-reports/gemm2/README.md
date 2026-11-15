# GEMM2 FP8 rebuilt LLVM audit

No kernel, CSV or Git state was modified. No new INT8 task was launched.

| BN / BK | GHz | Full dispatch (us) | CU0 lifetime MMAC (%) | Per-output MMAC span (%) |
|---|---:|---:|---:|---:|
| 256 / 128 | 1.8 | 23.499340 | 24.7889 | 38.4155 |
| 256 / 128 | 2.2 | 21.187530 | 21.6456 | 33.4150 |
| 256 / 128 | 1.5 | 25.987654 | 28.4860 | 45.0325 |
| 512 / 64 | 1.8 | 21.144124 | 28.3245 | 45.3056 |
| 512 / 64 | 2.2 | 19.034470 | 24.9285 | 38.0265 |
| 512 / 64 | 1.5 | 24.004664 | 30.7508 | 46.5757 |

Same frozen input and output hashes across clocks and old/new compilers within each BN. All eight measurements verified zero mismatches and untouched unselected output. All active FP8 MMAC counts are 4096 per CU0; 16 waves only on CU0; all 48 CUs complete one TG; no SGPR/VGPR spill or scratch.

BN256 new/previous 1.8 GHz: 23.499340 / 23.470428 us (+0.1232%). BN512: 21.144124 / 20.994004 us (+0.7151%). These small differences include independent compilation variation.

BN512 .text is identical across all three clocks; HSACO differs only in .debug_line and .debug_line_str. BN256 1.8 and 1.5 GHz have identical .text; 2.2 GHz has identical TTGIR but differing LLVM IR order of hoisted LDS GEPs, register assignments and scheduling. Therefore the BN256 2.2 GHz point is not a strict identical-binary frequency comparison.

Correctness logs already completed 4 cases per kernel, including 2 FP8 cases: BF16 with routed weights and FP16 with forced independent A/B/C 64-bit addressing plus non-unit scale strides. Each launches 4 M blocks x 2 N tiles with grid 1 (persistent 8 outputs), checks CPU and bitwise ordinary Triton and actual framework dispatch. This supplies full-N evidence separately from bounded equal-work performance (N_TILES_LIMIT=1).

Hardware trace audit covers: every wait ID in [0,15], first wait nonzero on every wave, no adjacent waits on the same ID. All pass; actual per-wave sequences remain in metrics.json.

Full details and direct perf paths: audit.json.
