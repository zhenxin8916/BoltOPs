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

## Continuation: LDS overlap, epilogue distribution and fetch evidence

All following passing numerical runs use the same48-CU/1.8GHz/two-output workload and frozen input/output hashes. None was promoted into the retained operator.

| Candidate | dispatch μs | observation |
|---|---:|---|
| complete consumer lookahead |13.391260|last-store→next-MMAC is shorter, but pre-store work and total inter-output gap grow|
| A-only lookahead |12.396020|still slower|
| adjacent M mapping |13.638680|same first96 M output set, different TG assignment|
| adjacent M plus same-expert B reuse |13.813820|correct but slower; reuse condition applies to single-N traversal|
| A priority0 or1 |12.168060|elapsed time and outputs exactly equal canonical; no observed scheduling benefit|
| SB prefetch, producer quotas32/32 |12.267028|same time as original SB-prefetch variant|
| SB direct LDS, dword |12.306504|native copy removes the explicit register-to-LDS store but is slower|
| SB direct LDS, dwordx4 |12.216988|close, but slower|
| row metadata direct LDS |12.537244|valid permuted row layout, no numeric errors, slower|
| metadata LDS read overlaps MMAC |12.452732|safe delayed A/metadata credit return, but not faster overall|
| metadata overlap plus SB direct LDS x4 |12.164724|only6 cycles below retained code; not treated as a material/generic gain, no full-N promotion validation|
| two-producer FP32 epilogues |23.434288|correct115KiB pipeline, substantial extra mailbox/handshake and producer work|
| FP32 epilogues, separate B and SB readiness |23.039528|slightly improves that pipeline but far behind retained consumer-scatter algorithm|

For the combined metadata-overlap/direct-SB diagnostic, CU0 SIMD0 C0 first MMAC moves8.222684→8.002508μs and C1 moves9.419196→8.938812μs. However, C0 has two new **consecutive-MMAC** issue gaps508/472 cycles at next-PC `0x…2b00`/`0x…2c00`, exactly256B boundaries; retained C0 has one92-cycle gap at `0x…2c00`. No intervening instructions are issued by that wave in these gaps. Other waves also exhibit some long gaps away from an exact boundary, so this is not sufficient to classify every stall as fetch.

The older independent PMD fetch probe at `/home/zhenx/moe-gemm1-tuning/icache-characterization-20261110/README.html` measured256B frontend requests and300-cycle cold boundary gaps. Together these support an **instruction-fetch hypothesis** for part of the regression, not proof of a model/hardware defect. `fetch_boundary_audit.py` records the current raw PCs and issued-instruction intervals. `numCyclesWaitIFetchAfterBranchTaken` is11330 wave-cycles for canonical CU0, average46.245 per taken branch; these are summed wave-cycles, not a dispatch-time component. The metadata-overlap variant has11327 such wave-cycles, so this branch-only statistic does not explain its additional straight-line MMAC gaps.

A shared noinline K128 math body was also attempted to reduce duplicated MMAC code, but clean Gluon explicitly rejects **non-scalar arguments to noinline functions**. It has no performance result. No compiler checks or native WDRA requirements were bypassed. Native WDRA currently requires four waves per partition (`ConvertWarpSpecializeToLLVM.cpp`), so changing to two-wave roles was not used as a workaround.

The retained code commit remains`973bc63` at12.168060μs with its verified full-N FP8/INT8 tests. Target9.602μs remains unmet. Further work should reduce static epilogue/fetch footprint or uncouple B publication from scale work **without** creating a costly FP32 mailbox pipeline or holding the current store behind a complete next-output read.


## Continuation: independent B and scale lifetimes

The retained source hash remains `0f1178d1c8d4ff2836d2f5bac17746b034b6e5a95d45960c62caf1273fc0852a` (code commit `973bc63`). Same frozen BM64 workload, GPU addresses, 48 CU, two independent output tiles per TG, 1.8 GHz, Triton `5b24821e` and LLVM `fd18eddb`; no compiler source changes. Every numerical PASS below checks 1,315,072 elements, zero mismatches, untouched output tail and the canonical output hash. These are bounded two-output validations, not new full-N protocol proofs.

| Candidate | dispatch μs | whole 48-CU dispatch MMAC | decision |
|---|---:|---:|---|
| retained |12.168060|9.358%|keep|
| B READY separate from SB READY; row metadata after MMAC |12.658452|8.995%|reject|
| same, issue row metadata LDS reads before MMAC |12.533908|9.085%|reject|
| separate B EMPTY and SB EMPTY credits |12.349872|9.220%|reject|
| same, SB buffer-to-LDS copy |12.279260|9.273%|reject|
| separate credits, complete metadata reads before MMAC |12.860836|8.854%|reject|
| full-K loads, two static K128 descriptor slices |12.647888|9.003%|reject|
| K128 tensor-phi loop |12.601740|9.036%|reject|
| same with Triton range `loop_unroll_factor=1` |12.601740|9.036%|reject|
| early route load and row-base/valid preparation in P0 |13.143284|8.664%|reject|

Perf establishes a tradeoff rather than a retained improvement. Splitting B READY from SB READY and issuing row metadata before MMAC advances C1's first MMAC from **9.419196 to8.816492 μs**, but its inter-output MMAC gap grows from **2336 to3680 cycles**. That first candidate returns B EMPTY only after MMAC plus SB read, delaying the next B generation. A subsequent variant returns B EMPTY after A/B LDS completion, with a distinct eight-reader SB EMPTY barrier9 and SB READY10, so P1 can reuse B while the old SB remains protected. Initial SB EMPTY gets eight credits from four initializer waves arriving with count2. P1 waits that SB credit only after issuing both B halves. Both consumers return SB credit only after their SB read completes. This reduces dispatch12.533908→12.349872 μs; direct SB buffer-to-LDS further reduces it to12.279260 μs, still slower than retained.

For the direct-SB candidate C1 starts MMAC at8.649692 μs, **0.769504 μs earlier** than retained, but its inter-output gap is3548 cycles; C0's is3652 cycles. In these inter-output intervals the A/B/metadata READY wait-to-next-issued-instruction intervals remain four cycles, while the final A/B LDS completion waits cost322–330 cycles. Therefore the remaining inter-output loss cannot be attributed to a long READY wait alone. Delayed scale reads/epilogue work and straight-line issued-instruction gaps remain relevant. The direct-SB trace still has436–440-cycle gaps between consecutive MMAC instructions with no intervening issued instructions, at PC mod256=192; this is evidence of an issue gap, **not proof of a fetch miss or model defect**.

The K128 slicing experiment also uncovered two frontend constraints. Reinterpreting MLS storage as a new leading K-phase dimension loses its physical allocation geometry and produced1,302,985 mismatches; that invalid candidate has no accepted timing. Keeping original descriptors and using constant `.slice` views passes numerical validation. Selecting shared descriptors with a tensor ternary fails compilation because shared descriptors cannot be converted to tensors. Branching on the two tensor loads instead compiles and passes. The final assembly still contains128 static MMAC instructions across both consumer bodies, including with `loop_unroll_factor=1`; neither variant shrinks the math code or improves dispatch. No compiler checks were bypassed and no PMD issue is claimed.

[Per-case metrics, phase evidence, hashes and raw-case paths](ready-and-k128-controls/summary.json) accompany this report. Rejected source remains only in the scratch experiment directories; the retained operator is unchanged. The original≤9.602 μs gate remains unmet.

A separate larger-A MLS control used tile32×128/interleave2 and loader2×2, a geometry present in the native MLS table. End-to-end compilation failed at LLVM translation with `builtin.unrealized_conversion_cast` at the MMAC result; **no numerical or performance result exists**. Native table geometry alone does not prove support for this A-to-MMAC layout combination. The public `MmacInterleave2Layout` explicitly wraps B or the result and rejects operand A. This candidate did not change compiler checks or enter the retained source. [Compile-control record](ready-and-k128-controls/a-interleave2-compile-control.json).


## Continuation: independent M pairs and pending FP32 output

These controls preserve BM64/BN256/K256, the same frozen 96-output set and GPU addresses, grid48, native four-wave-per-role 2P2C, and1.8GHz. Each run has1024 active MMAC instructions per CU, matching all48 CUs; numerical checks pass with1,315,072 elements, zero mismatches, untouched tail and the canonical output hash. No control is promoted.

| Work distribution / output scheduling | dispatch μs | whole48-CU dispatch MMAC |
|---|---:|---:|
| retained split-N consumers |12.168060|9.358%|
| two independent producer/consumer M pairs, K128×2 |13.970056|8.151%|
| pairs, global metadata prefetch before first B, LDS publish after second B |13.632564|8.353%|
| same, issue metadata LDS reads before second MMAC phase |13.778236|8.264%|
| retained split-N roles, one pending FP32 result per consumer |13.963940|8.155%|
| same, remove unnecessary current-accumulator loop carry |13.866640|8.212%|

The independent-pair control changes which data the roles own, while retaining2P2C: P0/C0 process M=pid, pid+2×grid,…; P1/C1 process M=pid+grid, pid+3×grid,… . Each consumer computes the complete64×256 output in two K128 phases. A is one full64×256 slot per pair and is retained in LDS across all N; B is one128×256 slot per pair. LDS is100KiB:32KiB A,64KiB B,2KiB row metadata,2KiB column scale. Requested/final quotas32/32/224/224; actual producer22 (prefetch25) registers and consumer153 (metadata-overlap193), no spills.

Each pair uses six barrier IDs: A READY/EMPTY, B READY/EMPTY and metadata READY/EMPTY (IDs1–6 and9–14, four waves each). A EMPTY is returned only after the final N's second-half A read completes. B EMPTY is returned after each K128 A/B LDS completion. Metadata EMPTY is returned after metadata read completion. Consequently producers may prefetch future B/A during compute/store without overwriting data still being read. Both the N loop and persistent M progression remain supported: FP8 and INT8 full-N checks use10 M blocks, grid2, two N tiles per M and unequal per-role M counts (three versus two). Each checks197,632 elements with zero mismatches. This validates this control's baseline protocol, not the other candidates' full-N behavior.

The baseline pair control has K128 boundary gaps C0/C1=2028/1952 cycles. The phase2 B READY wait-to-next-issued-instruction in that gap costs1584/1356 cycles. Moving row metadata global loads outside N and prefetching SB before the two B issues, while publishing LDS metadata after both issues, reduces those READY intervals to36/208 cycles and total K-boundary gaps to956/1048 cycles. This confirms that metadata publication between B phases was delaying phase2. Remaining gaps contain20 matrix LDS reads plus completion waits; first MMAC also starts later (C0 8.961052→9.165660 μs). Consumer tails remain4992/4428 cycles. Thus removing most of that READY wait does not beat the retained dispatch.

The pending-FP32 control loads and computes the next output before applying the previous output's strict `((acc*SA)*SB)*weight` epilogue. One extra drain iteration emits the final output; one common store body handles both outputs. It passes the bounded two-output test and truly places the first store after the final MMAC. However, the original control adds68/69 `v_mov_b32` instructions at the M boundary and extends final tails to6084/5356 cycles. Resetting the temporary current accumulator/SB each iteration eliminates those spurious accumulator loop-carrier copies (zero/one such moves in that boundary), but dispatch only improves13.963940→13.866640 μs. Final native WDRA quotas are28/20/232/232, adjusted by the compiler from requested48/16/224/224; actual consumer225/226 registers fit those final quotas, sum512, scratch0. There is no quota bypass.

The evidence rejects these specific alternatives: concurrent independent-M pairs add a K-boundary and longer single-output epilogues, while moving all stores after both MMAC bodies shifts work into a longer drain. It does not prove that all compute/store overlap is ineffective. Further work should preserve the retained split-N roles and avoid delaying the entire epilogue until the end.

[Metrics, launch records, phase analyses, source hashes and full-N results](persistent-output-controls/summary.json) accompany this report. Rejected sources remain scratch-only. Completed full-N pair IT traces are gzip-compressed with decompressed SHA verification to recover space; their `.gz` paths and hashes are recorded in the accompanying compression manifest. The best source remains code commit973bc63,12.168060 μs;≤9.602 μs is still unmet.
