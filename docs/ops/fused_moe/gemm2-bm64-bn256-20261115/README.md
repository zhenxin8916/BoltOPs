# GEMM2 BM64×BN256 perf analysis — 2026-11-15

This report, the retained operator and their measured evidence are saved in the same commit.

Current retained operator: `boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn256xbk256.py`.
Current best is **12.168060 μs**, repeated with identical machine `.text`, frozen inputs, GPU addresses, result hash and elapsed time. Target **≤9.602 μs is not met**.

## Measurement scope

PMD gfx946/HCU, 1.8 GHz (556 ps/cycle), 48 CU. Launch grid48, four roles × four waves. Each physical persistent TG processes two independent logical M64 output tiles: M=pid and pid+48, N0 only. Actual GEMM2 N2048/K256; BM64/BN256. The originating MoE preparation is M4096/E256/topk8. This is **one physical TG per CU processing two outputs**, not two simultaneous physical TGs.

MMAC utilization in the table is all 48 CUs averaged over the **whole dispatch**, including startup and tail. Canonical CU0 wave-lifetime utilization is 14.994%, while whole-dispatch is 9.358%; do not interchange them. CU0 has 1024 active MMACs, all 48 CUs have the same count, and active 8-cycle intervals do not overlap on a SIMD. There are 16 live waves per CU, four per SIMD, no exec=0 MMACs, and no compiler spills.

The BM128 two-output reference performs twice the padded work of BM64 two-output runs. 15.364/12.168 is only a bounded two-output latency ratio, not equal-work throughput improvement. Old raw result.json files call this `bounded_equal_padded_output_work`; that inherited label is incorrect **across BM sizes**. It is corrected in the archived harness and repeat result. Within BM64 all candidate comparisons use identical work, input hashes and GPU addresses.

## Algorithm and LDS ownership

2P2C remains intact. P0 issues A and SA/weight/route/expert metadata. P1 issues left/right B and channel SB. C0/C1 each compute M64×N128×K256, apply strict `((acc*SA)*SB)*weight`, convert, then scatter directly with packed `buffer_store_dword` in the natural accumulator layout. Generic x4 output conversion was slower and needed additional layout-transpose LDS. Neither consumer performs global metadata loads.

LDS99KiB: A2×64×256 =32KiB, B left/right each256×128 =64KiB, two row-metadata generations =2KiB, SB left/right =1KiB. No output mailbox. B uses native `matrix_load_64x32_b8` selected by `(128,32)` interleave2 panels and loader `(1,4)`; A uses native64x16. Requested WDRA quotas48/16/224/224, total512, no bypass.

A slots alternate `epoch&1`, READY1/2, EMPTY3/4 (eight consumer-wave credits), phase `(epoch>>1)&1`. Metadata READY12/0 has the same two-slot phase. B left/right use READY5/7, EMPTY6/8, phase `N_epoch&1`. Consumers issue A/B/SB/row-metadata reads, wait LDS completion, **then** return both A+metadata and B credits. Thus metadata cannot be overwritten while a consumer still reads it. A remains in registers for later N tiles. `protocol-audit.json` verifies actual trace register values for all eight consumer waves, both in two-output timing and five-M/two-N full traversal. This audits generation sequences, not every hardware-internal transaction.

## Perf identifies two different gaps

The following figures are CU0 SIMD0 and **issued-instruction intervals**, not isolated hardware instruction latencies. Whole-dispatch time remains the acceptance metric.

| Segment (GPU cycles) | C0 | C1 |
|---|---:|---:|
| first MMAC absolute time (μs) | 8.222684 | 9.419196 |
| last MMAC M0 → first store issue | 1052 | 528 |
| first → last store issue, M0 | 1644 | 1056 |
| last store issue → first MMAC M1 | 792 | 752 |
| complete M0→M1 MMAC gap | 3488 | 2336 |
| final MMAC → wave end | 2116 | 1260 |

In the M0→M1 interval, A READY, B READY and metadata READY each proceed to the next issued instruction in four cycles. The final LDS wait-to-next-instruction interval is318/326 cycles. Therefore this particular gap is **not a long B READY stall**. It includes 48 packed FP32 multiplies,16 packed BF16 conversions,16 packed dword stores, repeated route/mask/address operations, and24 matrix DS reads plus metadata loads for the next output. C0's output work also overlaps the still-active C1 computation, so issue contention matters; instruction count alone cannot predict speed.

Startup has a different bottleneck: C1 first MMAC is1.197μs later than C0. P1 serializes left/right publication around SB read/store completion. Hoisting SB reads, publishing SB before MLS, splitting B into K128 stages, raising B priority, and deferring next-A prefetch were therefore measured separately. None improved total dispatch. Moving route computation to P0 also reduced consumer work but delayed metadata and lost overall. These are measured negative results, not retained optimizations.

## Same-scope comparisons

| Algorithm | whole dispatch μs | 48-CU whole-dispatch MMAC |
|---|---:|---:|
| BM128 reference; two independent M tiles | 15.363948 | 14.823% |
| BM64: paired M32 mailbox; P0 scatter; B64x16 MLS | 12.761312 | 8.923% |
| BM64: native B64x32 MLS; two A slots; consumer scatter | 12.168060 | 9.358% |
| SB global read hoisted before B MLS | 12.267028 | 9.283% |
| SB LDS publication before B MLS | 12.337084 | 9.230% |
| P0/P1 each drain one consumer mailbox | 13.078788 | 8.706% |
| dual scatter plus SB prefetch | 13.020964 | 8.745% |
| P0 computes routed C row offsets and validity | 13.407384 | 8.493% |
| next-M A prefetch waits current A consumption | 12.591176 | 9.044% |
| B split into two K128 stages; native64x32 MLS | 12.710716 | 8.958% |
| P1 B send window uses priority1 | 12.252016 | 9.294% |

Only canonical source is retained in the repository. Rejected implementations remain under the raw temporary root for diagnosis. Current best improves BM64 initial baseline12.761312→12.168060μs (4.65% less time). The remaining work is to overlap next-output DS reads with current scaling/store without extending register lifetimes into spills, and reduce cold B-right publication latency without serializing the two producers. These are future work, not claimed gains.

## Correctness and reproducibility

Bounded timing checks1,315,072 selected BF16 elements: zero mismatches at rtol0.02/atol0.002; unselected rows/columns remain NaN. Comparison uses the original strict scale/weight operation order. Independent full-N tests use ten M blocks, grid2 (five M per persistent TG), two N outputs per M: **197,632 elements, zero mismatches for both FP8 and INT8**, exercising slot wrap/reuse and N traversal. These full-N tests are correctness checks, not performance acceptance runs. They do not exhaustively validate every optional wide-address/FP16 mode.

[Best SQTT perf](canonical-repeat-sqtt/GEMM2_BM64_BN256_K256_GRID48_OUTPUT96_2M_PER_CU_GC1800_CU0.perf), [metrics](canonical/metrics.json), [phase split](canonical/phase-analysis.json), [repeat proof](repeat-proof.json), [toolchain provenance](provenance.json), and archived assembly/HSACO accompany the source. Raw IT0 traces and frozen tensor payloads are retained at `/home/zhenx/temp/moe-gemm2-bm64bn256-20261115`; perf SHA/path is recorded beside the exported perf.

Use container `zx-perf-node60`, Python `/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python`. Launch harness sets clean Triton source PYTHONPATH and clang wrapper with `-mllvm -hcu-control-only-ebarrier-sync`. Reproduce by copying retained operator to a case directory as `kernel.py`, setting `BENCH_BK=256`, and running `launch.py <case> 64 --kind fp8`; frozen input root is resolved relative to the harness. `--freeze` regenerates frozen inputs at seed3101 if needed. Set `BENCH_DFLAGS='["StatLog","IT0","TT0","Perf"]'` for SQTT. Analyze with `analyze.py`, `phase_analysis.py`, `audit_barriers.py`, `audit_protocol.py`; full-N checks use `launch_check.py <case> <source> fp8` or `int8`, with BENCH_BK256. Toolchain commits/diff states and actual clang/PMD binary hashes are recorded in provenance; LLVM's two untracked files are not compiler inputs. No compiler source changes were introduced for this operator.

## Follow-up after retained-code commit `973bc63`

| Perf-guided candidate | whole dispatch μs | result |
|---|---:|---|
| Consumer metadata/address preparation before B wait | 12.670684 | zero numeric errors; rejected (slower) |
| Remove scheduling barrier after MMAC | 12.372668 | zero numeric errors; rejected (slower) |
| Register/lane DPP transpose; buffer_store_dwordx2 | 13.232244 | zero numeric errors; rejected (slower) |
| Two DPP swaps without canonical register ordering; scalar stores | 15.787064 | zero numeric errors; rejected (slower) |
| Two DPP swaps with canonical registers; buffer_store_dwordx4 | 13.378472 | zero numeric errors; rejected (slower) |

Consumer address preparation reduced M0 store-issue spans C0/C1 from1644/1056 to1100/700 cycles, but last-store→next-MMAC intervals rose792/752→1140/1832 cycles. It added a separate metadata LDS completion wait and delayed the next B read. Thus a shorter store segment did not imply shorter dispatch. The DPP variants truly produced dwordx2/x4 in assembly after register ordering was canonicalized, but the extra cross-lane selection and data rearrangement cost more overall. Removing the post-MMAC scheduling barrier also did not beat the retained code. None of these candidate source changes were promoted; **code commit973bc63 remains the BM64×BN256 best**. Measurements use the identical frozen BM64 workload, GPU addresses and1.8GHz setting.
