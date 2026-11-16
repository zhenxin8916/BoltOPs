# GEMM2 BM64×BN256 perf analysis — 2026-11-15

This report, the retained operator and their measured evidence are saved in the same commit.

Current retained operator: `boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn256xbk256.py`.
Current best is **11.794984 μs**, with **9.654002% whole-dispatch MMAC**, repeated with identical machine `.text`, frozen inputs, GPU addresses and elapsed time. The operator, report and row32 epilogue evidence are retained together in this commit. Target **≤9.602 μs is not met**.

## Measurement scope

PMD gfx946/HCU, 1.8 GHz (556 ps/cycle), 48 CU. Launch grid48, four roles × four waves. Each physical persistent TG processes two independent logical M64 output tiles: M=pid and pid+48, N0 only. Actual GEMM2 N2048/K256; BM64/BN256. The originating MoE preparation is M4096/E256/topk8. This is **one physical TG per CU processing two outputs**, not two simultaneous physical TGs.

MMAC utilization in the table is all 48 CUs averaged over the **whole dispatch**, including startup and tail. Current CU0 wave-lifetime utilization is 15.317%, while whole-dispatch is 9.526%; do not interchange them. CU0 has 1024 active MMACs, all 48 CUs have the same count, and active 8-cycle intervals do not overlap on a SIMD. There are 16 live waves per CU, four per SIMD, no exec=0 MMACs, and no compiler spills.

The BM128 two-output reference performs twice the padded work of BM64 two-output runs. 15.364/11.954 is only a bounded two-output latency ratio, not equal-work throughput improvement. Old raw result.json files call this `bounded_equal_padded_output_work`; that inherited label is incorrect **across BM sizes**. It is corrected in the archived harness and repeat result. Within BM64 all candidate comparisons use identical work, input hashes and GPU addresses.

## Algorithm and LDS ownership

2P2C remains intact. P0 issues A and SA/weight/route/expert metadata. P1 issues left/right B and channel SB. C0/C1 each compute M64×N128×K256, apply strict `((acc*SA)*SB)*weight`, then scale, convert and scatter two32-row groups with packed `buffer_store_dword` in the natural accumulator distribution. The MMAC computation itself remains full BM64. Generic x4 output conversion was slower and needed additional layout-transpose LDS. Neither consumer performs global metadata loads.

LDS100KiB: A2×64×256 =32KiB, B left/right each256×128 =64KiB, two row-metadata generations =2KiB, two SB generations per side =2KiB. P1 copies SB directly from its buffer to LDS before waiting for B EMPTY; consumers select slot `2*(b_epoch&1)+side`. No output mailbox. B uses native `matrix_load_64x32_b8` selected by `(128,32)` interleave2 panels and loader `(1,4)`; A uses native64x16. Requested WDRA quotas48/16/224/224, total512, no bypass.

A slots alternate `epoch&1`, READY1/2, EMPTY3/4 (eight consumer-wave credits), phase `(epoch>>1)&1`. Metadata READY12/0 has the same two-slot phase. B left/right use READY5/7, EMPTY6/8, phase `N_epoch&1`. Consumers issue A/B/SB/row-metadata reads, wait LDS completion, **then** return both A+metadata and B credits. Thus metadata cannot be overwritten while a consumer still reads it. A remains in registers for later N tiles. `protocol-audit.json` verifies actual trace register values for all eight consumer waves, both in two-output timing and five-M/two-N full traversal. This audits generation sequences, not every hardware-internal transaction.

## Perf identifies two different gaps

The following historical figures describe commit `973bc63` (12.168060 μs), CU0 SIMD0 and **issued-instruction intervals**, not isolated hardware instruction latencies. Whole-dispatch time remains the acceptance metric.

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
| Two-slot SB, direct buffer-to-LDS before B EMPTY (`a3db31f`) | 11.999036 | 9.490% |
| **Same producer protocol; row32 scale/convert/store groups (current)** | **11.954000** | **9.526%** |
| SB global read hoisted before B MLS | 12.267028 | 9.283% |
| SB LDS publication before B MLS | 12.337084 | 9.230% |
| P0/P1 each drain one consumer mailbox | 13.078788 | 8.706% |
| dual scatter plus SB prefetch | 13.020964 | 8.745% |
| P0 computes routed C row offsets and validity | 13.407384 | 8.493% |
| next-M A prefetch waits current A consumption | 12.591176 | 9.044% |
| B split into two K128 stages; native64x32 MLS | 12.710716 | 8.958% |
| P1 B send window uses priority1 | 12.252016 | 9.294% |

Only canonical source is retained in the repository. Rejected implementations remain under the raw temporary root for diagnosis. The original retained version improved BM64 initial baseline12.761312→12.168060μs (4.65% less time); the current two-slot SB version improves further to11.999036μs (5.97% less time versus the initial BM64 baseline). The remaining work is to overlap next-output DS reads with current scaling/store without extending register lifetimes into spills, and reduce cold B-right publication latency without serializing the two producers. These are future work, not claimed gains.

## Correctness and reproducibility

Bounded timing checks1,315,072 selected BF16 elements: zero mismatches at rtol0.02/atol0.002; unselected rows/columns remain NaN. Comparison uses the original strict scale/weight operation order. Independent full-N tests use ten M blocks, grid2 (five M per persistent TG), two N outputs per M: **197,632 elements, zero mismatches for both FP8 and INT8**, exercising slot wrap/reuse and N traversal. These full-N tests are correctness checks, not performance acceptance runs. They do not exhaustively validate every optional wide-address/FP16 mode.

[Historical 12.168 μs SQTT perf](canonical-repeat-sqtt/GEMM2_BM64_BN256_K256_GRID48_OUTPUT96_2M_PER_CU_GC1800_CU0.perf), [metrics](canonical/metrics.json), [phase split](canonical/phase-analysis.json), [repeat proof](repeat-proof.json), [toolchain provenance](provenance.json), and archived assembly/HSACO accompany the source. Raw IT0 traces and frozen tensor payloads are retained at `/home/zhenx/temp/moe-gemm2-bm64bn256-20261115`; perf SHA/path is recorded beside the exported perf.

Use container `zx-perf-node60`, Python `/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python`. Launch harness sets clean Triton source PYTHONPATH and clang wrapper with `-mllvm -hcu-control-only-ebarrier-sync`. Reproduce by copying retained operator to a case directory as `kernel.py`, setting `BENCH_BK=256`, and running `launch.py <case> 64 --kind fp8`; frozen input root is resolved relative to the harness. `--freeze` regenerates frozen inputs at seed3101 if needed. Set `BENCH_DFLAGS='["StatLog","IT0","TT0","Perf"]'` for SQTT. Analyze with `analyze.py`, `phase_analysis.py`, `audit_barriers.py`, `audit_protocol.py`; full-N checks use `launch_check.py <case> <source> fp8` or `int8`, with BENCH_BK256. Toolchain commits/diff states and actual clang/PMD binary hashes are recorded in provenance; LLVM's two untracked files are not compiler inputs. No compiler source changes were introduced for this operator.

## Follow-up after retained-code commit `973bc63`

| Perf-guided candidate | whole dispatch μs | result |
|---|---:|---|
| Consumer metadata/address preparation before B wait | 12.670684 | zero numeric errors; rejected (slower) |
| Remove scheduling barrier after MMAC | 12.372668 | zero numeric errors; rejected (slower) |
| Register/lane DPP transpose; buffer_store_dwordx2 | 13.232244 | zero numeric errors; rejected (slower) |
| Two DPP swaps without canonical register ordering; scalar stores | 15.787064 | zero numeric errors; rejected (slower) |
| Two DPP swaps with canonical registers; buffer_store_dwordx4 | 13.378472 | zero numeric errors; rejected (slower) |

Consumer address preparation reduced M0 store-issue spans C0/C1 from1644/1056 to1100/700 cycles, but last-store→next-MMAC intervals rose792/752→1140/1832 cycles. It added a separate metadata LDS completion wait and delayed the next B read. Thus a shorter store segment did not imply shorter dispatch. The DPP variants truly produced dwordx2/x4 in assembly after register ordering was canonicalized, but the extra cross-lane selection and data rearrangement cost more overall. Removing the post-MMAC scheduling barrier also did not beat the retained code. None of these candidate source changes were promoted; **these candidates did not exceed code commit973bc63**. Measurements use the identical frozen BM64 workload, GPU addresses and1.8GHz setting.

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


## Continuation: chunked stores, asynchronous reads and readiness probes

Retained code remains973bc63,12.168060 μs/9.358% whole-dispatch MMAC. The following source-only controls retain the same2P2C split-N roles,99KiB LDS,48 CU, two output tiles per persistent TG, frozen inputs/GPU addresses,1.8GHz and strict scale order. Every passing run checks1,315,072 elements with zero mismatches, canonical output hash and untouched tail. These are bounded two-output results; no new full-N correctness claim is made for these rejected controls. No Triton or LLVM source change was made.

| Control | dispatch μs | whole48-CU MMAC | static store instructions across both consumer bodies |
|---|---:|---:|---|
| retained |12.168060|9.358%|32 dword|
| split32-row stores, no prefetch |13.127160|8.674%|64 short|
| split stores, blocking read helper between halves |13.682604|8.322%|64 short|
| full-output async read helper before epilogue |13.240028|8.600%|32 dword|
| split stores, defer LDS completion until next MMAC |13.585304|8.382%|64 short|
| rebuild split-store addresses, no prefetch |12.206424|9.329%|32 dword|
| rebuilt addresses, async prefetch between halves |12.754640|8.928%|32 dword|
| nonblocking READY probe before optional prefetch |13.460204|8.460%|32 dword|
| prepare full epilogue before directly replacing current prefetch variables |13.240028|8.600%|32 dword|
| replace variables directly between rebuilt split stores |12.754640|8.928%|32 dword|

The first split implementation used reshape/permute/split on packed output, offsets and masks. It was numerically correct, but inspection found `buffer_store_short` instead of retained `buffer_store_dword`: static stores doubled32→64. Canonicalizing only the half-tile layout to a DistributedLinearLayout did not restore width; those controls retained exactly13.127160 and13.585304 μs. Rebuilding offsets after splitting the32-row route/expert vectors, with a new contiguous N arange in each half, restored dword stores without false alignment assumptions. Its no-prefetch control reduces13.127160→12.206424 μs but still loses to retained by0.038364 μs. Thus raw split-store timings must not be interpreted as a test of otherwise equivalent memory instruction width.

The asynchronous helper issues A/B/SB/row-metadata LDS reads but defers explicit `wait_lgkmcnt(0)` and A/B EMPTY returns until the next iteration before MMAC. This protects both matrix and metadata lifetimes: credits are not returned while pending reads are outstanding. It does not make READY acquisition nonblocking. Perf for full-output async prefetch still shows C0/C1 B READY wait-to-next-issued-instruction intervals764/316 cycles **before the current first store**. A separate control calls native nonblocking `abarrier_test_wait` for B, and for A/metadata at an M boundary; only when all are complete does it prefetch, otherwise it completes the old stores and performs the blocking read at next-iteration entry. It passes numerical checks and uses the real hardware probe API, but its extra state/control does not improve dispatch.

With rebuilt split-store addresses and async prefetch, C0's last-store→next-MMAC interval falls792→184 cycles, and its entire inter-output MMAC gap falls3488→3388 cycles. However, C1's gap grows2336→4116 cycles; first→last store issue intervals are2652/2708 cycles because the prefetch work is now between the stores. C0/C1 first MMAC times are8.336108/9.123404 μs; tails after the final MMAC are2820/1280 cycles. Consequently a shorter last-store→MMAC interval is not sufficient evidence of a faster dispatch.

Moving the prefetch helper until after current packed values/addresses are prepared and assigning A/B/metadata directly instead of through `na/nb/...` changed register demand but not measured elapsed time: full-store13.240028 μs and split-store12.754640 μs remain identical to their preceding controls. All recorded final native quotas remain legal and scratch size is0. No code control was promoted.

[Combined launch records, numerical results, metrics, phases, source/assembly hashes and instruction-width counts](chunked-store-controls/summary.json) accompany this report. Completed rejected IT traces were losslessly gzip-compressed and their decompressed SHA256 verified to recover approximately0.96GB; canonical and current key comparison traces remain available. [Compression manifest](chunked-store-controls/compressed-rejected-controls.json) records restored paths and hashes. Original≤9.602 μs gate remains unmet.


## Retained update: two SB generations and early direct LDS prefetch — 2026-11-16

**11.999036 μs / 9.489829% whole-dispatch MMAC**, down1.39% in time from12.168060μs on identical BM64 work. The ≤9.602μs gate is still unmet. This does not establish 60% equal-work throughput improvement over BM128.

P1 now issues native buffer-to-LDS SB copies before B EMPTY, commits the async group, waits B EMPTY, issues the same B MLS, waits for SB completion, and publishes B READY. SB has two generations per side: slot `2*(epoch&1)+side`. LDS grows99→100KiB; no extra barrier IDs or output mailbox are introduced. P0, A/MA/B ownership, strict scale order, native interleave2 B reads, LTS MMAC and 2P2C remain intact. WDRA requests48/16/224/224; actual registers19/4/156/157, no spills.

Safety: SB(e) reuses SB(e−2). Before issuing SB(e), P1 already issued B(e−1), which required B EMPTY(e−2). All four consumer waves return that credit only after their B and SB reads finish. The first two generations use distinct slots. The trace audit checks actual M0/LDS effective addresses (including encoded DS offsets), completion waits before credit returns, and generation reuse on timing plus full-N FP8/INT8 runs. The A/MA/B protocol audit also passes.

The frozen timing run checks1,315,072 elements with zero errors and untouched masked output; its repeat produces the exact same11.999036μs, inputs, addresses and machine text. Full-N FP8 and INT8 each check197,632 elements with zero errors (five M per persistent TG, two N per M). Final source comment cleanup compiles to the same6728-byte `.text`, SHA256 `f8274e6c9ad6bfe85eb8c6dab8fbe62706cffb64dfda94f73932651e415d3fc6`.

| CU0 SIMD0 issued interval | C0 | C1 |
|---|---:|---:|
| first MMAC absolute time, μs |8.562956|8.847628|
| M0→M1 MMAC gap, cycles |2840|3260|
| last MMAC→first store, cycles |1028|1236|
| first→last store issue, cycles |1096|1228|
| last store→next MMAC, cycles |716|796|
| final MMAC→wave end, cycles |2104|1196|

C1 startup advances0.572μs; C0 startup and C1's inter-output gap do not improve. This is an overall dispatch gain, not a claim that every gap shrank. The remaining bottleneck includes scaling/scatter and next-output LDS acquisition, with substantial startup/tail cost at short K.

[Current best perf](sb-two-slot-direct-early/GEMM2_BM64_BN256_K256_GRID48_OUTPUT96_2M_PER_CU_SB_TWO_SLOT_GC1800_CU0.perf), [metrics](sb-two-slot-direct-early/first/metrics.json), [repeat proof](sb-two-slot-direct-early/repeat-proof.json), [source](sb-two-slot-direct-early/kernel.py), [provenance](sb-two-slot-direct-early/provenance.json). Raw case roots are `sb-two-slot-direct-early`, `sb-two-slot-direct-early-repeat-sqtt`, `sb-two-slot-fulln-fp8`, and `sb-two-slot-fulln-int8` under the existing temporary root. Reproduction uses the harness and BENCH_BK=256 described above.

### Ordinary-layout controls (rejected)

Native LEGACY/LIT/LIT_LTS with TN2 fail the backend's MLS element-count check; it was not bypassed. Legal TN1 variants validate numerically but take28.725740/15.138768/13.614772μs respectively. LIT TN1 emits16 static buffer_store_dwordx2 instructions versus32 packed dword stores in retained LTS, but is slower. For equal37888 valid output bytes in CU0, per-issued-instruction64/128-byte address-region references are1184/1184 versus592/592; these are address coverage counts, **not measured DRAM transactions**. Wider stores alone do not establish better throughput.

Reading interleave2 B directly as ordinary LIT produces1,222,257 mismatches and is rejected; explicitly converting the native layout fails compilation. These are layout incompatibilities, not evidence of a PMD defect. A metadata-overlap variant of the retained two-slot SB version takes12.050744μs and is also rejected. Rejected source variants remain in temporary evidence, not in production code.


## Controls after code commit `a3db31f` — 2026-11-16

The retained operator remains11.999036μs. All passing controls below use identical frozen inputs, GPU addresses, launch scope and expected output hash:48 CU/1.8GHz/two independent logical BM64 N0 outputs per persistent TG. Each checks1,315,072 elements with zero errors and preserves masked output. None exceeds the retained code; no candidate source was promoted. [Launch/metrics/result/source-hash evidence](post-a3db31f-controls/summary.json).

| Control | dispatch μs | whole48-CU MMAC |
|---|---:|---:|
| Initialize/invalidate only used abarriers |12.329300|9.236%|
| SB wait before B MLS |12.698484|8.967%|
| Prefetch both SB halves before left B |12.141372|9.379%|
| Merge four row-metadata stores into one same-byte-layout store |12.074652|9.430%|
| Merged row-metadata loads before A |12.194748|9.338%|
| One P0 wave owns merged row-metadata publication |12.079100|9.427%|
| One-wave interleaved row metadata and combined consumer LDS read |12.287600|9.267%|
| WM4/WN1 native MMAC layout |12.392684|9.188%|
| P0=A/Bleft/row metadata, P1=Bright |12.893640|8.831%|
| Same producer redistribution plus merged row-metadata store |12.528904|9.088%|
| Redistributed producers, A before expert descriptor acquisition |12.692924|8.971%|
| Native MLS L1 enabled, TRITON_HCU_DISABLE_MLS_BPS=1 |12.035732|9.461%|
| Native leading K128 panels with constant two-step consumption |12.354320|9.217%|
| Native leading K128 panels, opaque trip-count diagnostic |12.865284|8.851%|
| Separate SB READY; wait SB only after both B MLS groups issued |12.123024|9.393%|
| Separate SB READY plus merged row metadata |12.282596|9.271%|
| Explicit C0→C1 MMAC token ping-pong |12.734624|8.942%|

The merged row-metadata store removes three partition ebarrier synchronizations and their LDS drains. Its C0/C1 first MMAC advances8.563/8.848→8.267/8.643μs, but the inter-output MMAC gaps grow2840/3260→3484/3544cycles. Its earlier startup therefore does not shorten total dispatch. Issued intervals include competition between waves and must not be equated to intrinsic instruction latency. Electing one P0 wave additionally removes duplicate cross-wave metadata requests/stores; READY0/12 counts change4→1 and only that wave publishes, without changing A/B reader credits. It still does not improve whole-dispatch time.

Redistributing Bleft to P0 makes the producer byte counts more balanced but moves row-metadata publication onto a longer A/Bleft path. C0/C1 first MMAC becomes8.921/9.206μs in the merged-store control, later than retained. This is evidence against that redistribution as implemented, not a claim that independent B sends can never help.

The K128 control allocates the K panels explicitly and has producers write the matching native layouts; it does not reinterpret full-K storage into an incompatible physical geometry. The opaque-trip-count case is a diagnostic, not a production optimization: LLVM emits192 static MMAC instructions through runtime unrolling/remainder code versus128 for the constant control. Both execute the expected1024 active MMACs per CU. Thus the experiment **did not implement a smaller math body** and does not disprove the fetch hypothesis. No opaque identity is retained in production.

WM1/WN4 is rejected by native WDRA SGPR-spill checking. Capturing a computed expert scalar from the preamble is rejected by HCU WASP lowering. These have no performance result; native validation was not bypassed. Shared-descriptor reshape with mismatched layout rank also fails compilation; the passing merged-store controls use an explicit physically matching shared layout. No new PMD defect is inferred from these compiler failures.


A final SB control adds READY9/10, issues both left/right B MLS groups before the shared scale-copy completion wait, then publishes both SB READYs. Consumers complete both B and SB LDS reads before B EMPTY, so the two-slot SB reuse proof still applies; no early EMPTY is used to hide an outstanding scale read. Its12.123024μs and merged-metadata12.282596μs do not beat retained. Explicit consumer compute tokens9/10 enforce C0→C1→C0 ordering with four arrivals per phase. Although they eliminate cross-role MMAC issue interleaving, C1 first MMAC slips8.848→9.464μs and whole dispatch rises12.734624μs. The extra four-wave rendezvous is counterproductive here. These are bounded timing/correctness controls, not full-N promotion validation.

All17 passing controls are slower than `a3db31f`. The best source/perf remains unchanged; these results narrow the next work toward epilogue and fetch scheduling rather than adding producer/consumer synchronization. The requested ≤9.602μs goal remains active and unachieved.


## Retained update: row32 epilogue groups — 2026-11-16

Primary acceptance input: **11.954000 μs / 9.525581% whole48-CU dispatch MMAC**, down0.3753% from `a3db31f` (11.999036μs). Repeat SQTT uses identical source/options/inputs/addresses and the same6720-byte machine `.text` (SHA256 `4aced79ff52ca8cfefce641feb25ca87c17ca87248d2d16d62a659ae6e894087`), with exactly equal timing and outputs. This is a small measured improvement; **the ≤9.602μs goal is not met**.

C0/C1 still compute full BM64×N128×K256. After MMAC, the accumulator and row metadata are split into two32-row views whose partition boundary is a register basis. Each group separately performs strict `((acc*SA)*SB)*weight`, narrowing and routed dword scatter; a scheduling fence keeps the first group's stores ahead of the second group's arithmetic. Unlike the older split-store controls, this also splits the scaling/conversion work, rather than preparing all scaled values first. No extra LDS allocation, data exchange between waves, output mailbox, consumer global load, barrier ID, or slot ownership change is introduced. A/B/SB producer protocol, WDRA requests48/16/224/224 and actual19/4/156/157 remain the same; scratch0, LDS100KiB.

Both FP8 and INT8 full-N tests check197,632 elements with zero errors (five M per TG, two N per M). Timing checks1,315,072 elements with zero errors and untouched masked output. A/MA/B phase audits and two-slot SB address/completion/reuse audits pass on timing and both full-N traces. The timing barrier audit sees16 live waves/four per SIMD,1024 active MMACs/CU0 and zero exec=0 MMACs.

An independent input/routing seed3209 compares the old and row32 versions on identical new inputs, GPU addresses, launch options and expected outputs:11.993476→11.812220μs,1.5113% less time,1,280,000 checked elements with zero errors. These are a paired generalization control, **not a replacement for the primary acceptance input**. It supports a benefit beyond one frozen routing sample but does not prove improvement for every shape or routing distribution.

| CU0 SIMD0 issued interval, cycles | old C0 | row32 C0 | old C1 | row32 C1 |
|---|---:|---:|---:|---:|
| MMAC end to first store |1028|1092|1236|368|
| First to last store |1096|1348|1228|1644|
| Last store to next MMAC |716|784|796|976|
| Complete inter-output MMAC gap |2840|3224|3260|2988|
| Final MMAC to wave end |2104|2160|1196|1068|

C1 starts writing earlier and its inter-output gap shortens, while C0's gap grows. Whole-dispatch gain is therefore the acceptance evidence; there is no claim that every local interval improves. Increasing the memory cluster budget or enabling MMAC clustering leaves timing exactly11.999036μs; maximum-ILP scheduling regresses12.513892μs. Packing before scaling12.050188μs, column64 groups12.146932μs,32×64 quarter groups12.207536μs and C1-only async lookahead12.888636μs are rejected. [Control evidence](row32-scale-convert-store/rejected-controls.json).

[Current best perf](row32-scale-convert-store/GEMM2_BM64_BN256_K256_GRID48_OUTPUT96_2M_PER_CU_ROW32_EPILOGUE_GC1800_CU0.perf), [metrics](row32-scale-convert-store/first/metrics.json), [repeat proof](row32-scale-convert-store/repeat-proof.json), [seed3209 paired proof](row32-scale-convert-store/seed3209-pair-proof.json), [toolchain provenance](row32-scale-convert-store/provenance.json), and [retained source](row32-scale-convert-store/kernel.py) accompany this update. Raw case directories are `row32-scale-convert-store`, `row32-scale-convert-store-repeat-sqtt`, `row32-fulln-fp8`, `row32-fulln-int8`, `row32-seed3209`, and `baseline-seed3209` under the existing temporary root. Native compiler checks remain enabled. Rejected sources stay temporary. Completed old negative-control IT traces were losslessly compressed with decompressed SHA verification; [compression manifest](post-a3db31f-controls/compression-manifest.json).

## Retained update: P0 route preparation and paired metadata — 2026-11-16

**11.794984 μs / whole-dispatch 48-CU MMAC 9.654002%**, 1.3302% less time than the previous row32 version (11.954000 μs). Same 1.8 GHz / 48 physical TG / two independent BM64 outputs per TG scope. P0 decodes routes and folds invalid experts once; consumers drop the expert LDS read and duplicate decoding. SA/weight share one LDS store operation. No slot, phase, MMAC layout, multiplication order or LDS budget change.

CU0 SIMD0 inter-output MMAC gaps improve C0 **3224→3060 cycles**, C1 **2988→2784 cycles**. First MMAC starts later by 148/460 cycles, so startup starvation remains; final C1 tail grows 80 cycles, C0 tail shrinks 304. Overall repeat is identical and both handoffs shrink. FP8/INT8 full-N correctness and A/metadata/B/SB reuse audits pass. Independent seed3209 improves 11.812220→11.741608 μs.

[Algorithm, phase breakdown, validation and rejected controls](route-pair-metadata/README.md), [current perf](route-pair-metadata/GEMM2_BM64_BN256_K256_GRID48_OUTPUT96_2M_PER_CU_ROUTE_PAIR_METADATA_GC1800_CU0.perf), [repeat proof](route-pair-metadata/repeat-proof.json), [toolchain](route-pair-metadata/provenance.json). The ≤9.602 μs target remains unmet.
