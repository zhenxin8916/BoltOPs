# GEMM2 BM64 / BN512 / BK64 tuning

2026-11-15. Dedicated operator: `boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn512xbk64.py`. No change to BM128 operators or framework selection. Full N traversal is supported by the direct entry; timing intentionally measures only the first BN512 panel.

**Current best: 14.873556 us / whole-dispatch MMAC15.311577%, consumer-m1n4. The 60% target remains open.**

## Acceptance and current results

User explicitly chose elapsed time for **two output tiles of each geometry**, at 1.8 GHz and 48 CUs, rather than equal-work throughput. Thus BM64 does half the padded MMAC work of BM128. Neither the speedup below nor this bounded diagnostic is an equal-work/full-MoE throughput claim. The target was subsequently raised to **60% lower elapsed time**, requiring **<=8.272168 us** against historical BM128 best 20.680420 us. **Not achieved yet.**

| Algorithm | Full 48-CU dispatch us | Whole 48-CU dispatch MMAC | LDS KiB | Actual role VGPRs |
|---|---:|---:|---:|---|
| Historical BM128/BN512 best (older frozen toolchain) |20.680420|22.024466%|117|40/9/203/203|
| BM64, K128 B publication / K64 consumption / two M32 output panels |**16.051164**|**14.188230%**|99|33/9/126/126|
| BM64, native B64x32 MLS |**15.641948**|**14.559414%**|99|33/9/126/126|
| BM64, K128 consumption |16.096756|14.148043%|99|33/9/126/126|

The initial baseline elapsed-time reduction was **22.384729%**; the native-B update below is now best. The stricter 60% target remains active. Historical comparator source, HSACO and toolchain are under `../gemm2-20261115/bn512-b-batch-k128-best-gc1800/`. Current compiler is LLVM fd18eddb plus clean Triton 5b24821; provenance records exact binary hashes. This is a cross-toolchain historical comparison, not an isolated single-factor experiment.

The BM64 best checks 2,630,144 valid elements with zero mismatches; unselected columns and rows remain NaN. The repeated run matches inputs, GPU addresses, machine `.text` bytes, output hash and elapsed time exactly. See `repeat-proof.json`. Raw traces and frozen tensor payloads remain in the raw root recorded by `provenance.json`; perf path/SHA are in `perf-location.json`. The promoted source matches repeat/kernel.py exactly.

## Pipeline and resource layout

2P2C, four waves per role, native WDRA, requested registers 48/16/224/224. P0 loads one full K256 A tile, two metadata generations, prefetches next M and drains/scatters output mailboxes using buffer_store_dwordx4. P1 sends BL01 -> BR01 -> BR23 -> BL23 for each N128 phase and supplies SB. C0/C1 each own 256 columns in two N128 phases. They read A once per M, perform K64 MMACs, scale strictly ((acc*SA)*SB)*weight, convert to BF16/FP16 and publish two M32 output panels per phase. INT8 keeps INT32 accumulation until all K contributions are complete.

LDS is A16 + B64 + C16 + double MA2 + SB1 = **99 KiB**. A is 64x256 bytes. Each consumer has two K128xN128 physical B slots (16 KiB each). Each C mailbox is M32xN128x2 bytes (8 KiB). Metadata is two generations x four 64-element FP32 vectors (2 KiB); SB is two 128-element FP32 vectors (1 KiB).

48 CUs each complete one physical TG processing two M outputs. Each CU issues exactly 2048 active FP8 MMACs; CU0 has 16 live waves (four per SIMD), no exec-zero MMAC, no adjacent same-ID waits and IDs remain 0..15. A/B READY generations match both M iterations. Consumer VMEM loads and permlane instructions are absent; scratch spills are absent. `model_metrics.py` derives expected work from actual BM/BN/K/tile counts and uses the full HYGPERF dispatch window, not the shorter CU0 wave lifetime.

## Correctness

`test/fused_moe/test_channel_gemm2_bm64_bn512_equivalence.py`: four opt-in tests passed. Each covers complete N1024/K256 over multiple sorted M64 blocks on one persistent TG, against both CPU and ordinary non-Gluon Triton. FP8 and INT8, BF16 with routed weights, FP16 without weights, forced independent 64-bit A/B/C addressing and non-unit scale strides are included. All 395,264 elements per case match the reference kernel bit for bit; CPU tolerance is rtol=.02/atol=.002. Forced wide paths use small physical allocations and do not validate physical allocations above 4 GiB. This test invokes the dedicated operator directly, not framework dispatch selection.

```bash
RUN_GLUON_GEMM2_BM64_BN512_EQUIVALENCE=1 python -m pytest -s -q test/fused_moe/test_channel_gemm2_bm64_bn512_equivalence.py
```

## Reproduce the bounded performance case

Run in zx-perf-node60 with the paired Python environment. Create a case directory with the production source copied as kernel.py, then use launch.py <case-directory> 64 --freeze once and launch.py <case-directory> 64 --kind fp8. Both launcher and harness are archived here; the launcher configures the toolchain, 1.8 GHz and separate per-case cache. Run model_metrics.py and the three audit scripts against the case directory afterward. There is no warmup dispatch.

All times in this report are PMD model measurements, not physical-device measurements.

## Native B MLS update: current best

Promoted source now uses B MlsSharedLayout(1,(128,32),alt=1), loader waves(1,4), selecting native matrix_load_64x32_b8. The former (64,32)/(2,2) path selected matrix_load_64x16_b8. Static MLS count falls 40 -> 24 (B32 -> 16; A8 unchanged). No backend changes, LDS remains 99 KiB, actual registers 33/9/126/126 with no spills.

New full-dispatch result: **15.641948 us / 14.559414211% whole 48-CU dispatch MMAC**, **24.363490%** lower elapsed time than historical BM128 two-output20.680420 us, 2.549448% below initial BM6416.051164 us. The current 60% target still requires <=8.272168 us and is not achieved.

Source, same-text/input/address/time repeat proof, four passing full-N1024 equivalence cases, metrics and exact perf path are in [native-b-mls64x32](native-b-mls64x32/). This supersedes the initial source snapshot in repeat/; historical rows above remain as recorded.

Rejected numeric-passing candidates: full-CN256 with one B slot16.841240 us; consumer direct scatter19.987088 us; converting to BF16 before direct-scatter layout conversion19.009084 us; full-M64 mailbox17.010820 us; SB prefetch/joint metadata READY16.065064 us (16.464272 with native B); priority1 B15.995564 us; native B K128-consumption15.705332 us; left-first tail15.785396 us; N static unroll15.985000 us; two K128 A READY stages16.256328 us. These implementations remain temporary, not promoted. Wide-A native geometry is rejected by current clean Triton lowering, and a noinline initialization prototype fails the WDRA execution-mode check; neither has a performance result or backend workaround.

## Full-K256 B slot update

Each consumer now owns one K256xN128 B slot instead of two K128xN128 slots. P1 publishes left then right once per N128 phase; consumers read all four K64 quarters before returning EMPTY6/8, then compute from registers. READY5/7 phases are [0,1,0,1] across the two M outputs. The former second-K READY10/2 and EMPTY14/4 are unused (initialization retained). Scale is read before returning B credit. A, metadata, output handoff and scale arithmetic order are unchanged. LDS remains 99 KiB; actual VGPRs rise from 33/9/126/126 to 33/9/146/146, without scratch spills. This supersedes the B-slot description above.

At 1.8 GHz, 48 persistent TGs process two outputs each: **15.612480 us / whole-dispatch MMAC14.586895%**. Independent recompilation repeats exactly the same machine .text, inputs, addresses, output and elapsed time. Improvement versus commit338a785 is **0.188391%** (0.029468 us); versus historical BM12820.680420 us it is **24.505982%**, with half the padded work as stated above. The 60% target is still not achieved.

2,630,144 bounded-output elements match, untouched output remains NaN. Four full-N1024 FP8/INT8 equivalence cases pass; real-trace barrier/phase, active-work and no-consumer-VMEM audits pass. Evidence, tested source, production documentation-only edit proof, HSACO and perf location are under [full-k256-b-slot](full-k256-b-slot/). All measurements use the same clean toolchain and PMD configuration as the prior best.

## Perf-driven metadata scheduling update

Current best **15.538532 us / 14.656314%** whole 48-CU dispatch MMAC, same two-output-per-TG/1.8GHz scope. Relative to commit3208d1d15.612480 us, elapsed time falls **0.473647%**. Native WDRA actual VGPRs are36/9/146/146, LDS99 KiB, no spills. Four full-N FP8/INT8 equivalence cases and all trace audits pass; independent compilation repeats text, frozen inputs, GPU addresses, output and elapsed time exactly. Evidence is in [ma-register-prefetch](ma-register-prefetch/).

Root-cause evidence from prior best CU0/SIMD0: C0's EMPTY11 issue gap9.477020->10.471148 us is1788 cycles. During it, P0 is writing next-M metadata to LDS and issuing its VMEM/LGKM waits; first output drain starts10.422220 us. The whole between-MMAC gap after ordinal63 is2488 cycles, including scaling, handoff, next B reads and waits. These issue intervals include scheduling/resource effects and are not pure instruction latency.

Solution: preserve next-A MLS prefetch, load next-M scale/weight/route/expert into registers early, drain every current output mailbox, then publish next-M metadata to LDS before advancing M. Consumers still use READY12/0 before reading metadata; metadata generations remain double-buffered and bound by completed current output drain. Source separates `_read_metadata` from `_publish_metadata`. Arithmetic order and scatter mapping are unchanged.

CU0/SIMD0 diagnostics: C0 gap after ordinal63 falls2488->1688 cycles; C1 cross-M gap after127 falls2320->1308 cycles. Other intervals grow: C0 cross-M960->1640 cycles; hence only0.074us total gain, not a wholesale removal of stalls. Full dispatch is the acceptance metric.

Negative controls: deferring next metadata loads until the first output pair drains15.658628 us, or until all output drains15.899376 us, both worse; they move stalls rather than hide memory latency. Draining C0's two panels before C1 gives15.550208 us, slower than the new best. A two-output-slot prototype sharing one READY missed a generation after two back-to-back phase flips and was stopped; it has no timing result. Distinct READY IDs are required for that design. No backend or model workaround is used.

## Output LDS mapping update

Current best: **15.423440 us / whole 48-CU dispatch MMAC14.765681%** at1.8GHz, two output tiles per persistent TG. This is **0.740688%** lower time than commit b38e42315.538532 us. The 60% target remains unmet.

Only output mailbox shared layout changes: SwizzledSharedLayout(vec8,perPhase4,maxPhase4) -> (vec8,perPhase1,maxPhase8). The two-role producer/consumer assignment, single M32 mailbox per consumer, barrier phases, metadata schedule, arithmetic and scatter mapping remain intact. LDS stays99KiB and actual VGPRs36/9/146/146; no spills. Four FP8/INT8 full-N equivalence cases pass, as do active-work, barrier/phase and consumer-VMEM audits; independent recompilation matches machine text, input/address hashes, output and exact elapsed time. Evidence: [output-lds-8-1-8](output-lds-8-1-8/).

Perf evidence: mean LocalMemPipeline.creditStall falls960.729->723.063 cycles/CU. ldsBankConflict remains1024/CU; this is not evidence of eliminated bank conflicts. Static DS write opcode counts are unchanged, so this is not an instruction-count reduction. CU0/SIMD0 C0 gap after MMAC63 falls1688->764 cycles, while cross-M gap127 grows1640->1800 and metadata gap143 grows1556->2312. The remaining metadata READY issue interval is2296 cycles (11.458604->12.735180 us). These are issue gaps, including scheduling effects, not pure memory/instruction latency. Next action is earlier publication of already-prefetched next-M metadata after the first N phase drains.

Negative controls: output layout(vec4,perPhase1,maxPhase8)15.593020 us; two M32 slots with distinct READY IDs15.561328 us; those slots plus metadata prefetch15.571336 us; separate B-data READY before scale publication16.539888 us. All bounded numerics pass, but none exceeds current best, so their implementations remain temporary.

## Publish next-M metadata after the first N phase drains

Current best **15.322804 us / whole-dispatch MMAC14.862658%**, same1.8GHz/48CU/two-output-per-TG scope. This improves commit aca19e5 by 0.652487%. Four complete-N1024 FP8/INT8 equivalence cases and all dynamic barrier/phase, active-work and consumer-VMEM audits pass; independent recompilation repeats the exact machine text, inputs, addresses, output and elapsed time. LDS99KiB, actual role VGPRs36/9/146/146, no spills. Evidence: [ma-after-first-n](ma-after-first-n/).

Perf hypothesis: output-layout optimization moved the bottleneck to next-M metadata publication. C0's metadata READY issue interval reached2296 cycles after the first K64 chunk of the second M. Metadata was prefetched into registers but published only after all current N phases drained. Publish it after n==0/panel==1 instead: this avoids blocking the first pair of current output mailboxes, while allowing later N phases to overlap metadata LDS publication. READY12/0 and its M-generation phase remain unchanged. Each metadata slot is still reused only after every output of the older M has been drained. The condition is independent of the diagnostic N limit and works for full N traversal.

After this change, ordinal143's former2312-cycle between-MMAC gap is no longer among C0's twelve largest gaps; C1 cross-M gap127 falls1988->1172 cycles. Other waits remain: C0 cross-M gap1816 cycles, C1 gap1912116 cycles. Startup A READY wait is2440 cycles on C0/SIMD0; first B-right READY wait1564 cycles on C1. The 60% reduction target is still unachieved; these are small pipeline improvements, not completion of the objective.

## Initialize only live pipeline barriers

Current best **15.218832 us / whole 48-CU dispatch MMAC14.964197%**, same1.8GHz/two-output-per-persistent-TG scope. Improvement from commit efb1279 is0.678544%; versus the start-of-turn3208d1d15.612480 us it is2.521368%. The historical BM12820.680420 us elapsed reduction is26.409464% with half the padded work, not equal-work throughput. The60% objective remains open.

Removing the former second-K128 B stages left AB2/4/10/14 unused. They no longer receive an initializer or invalidation, and obsolete initial EMPTY4/14 credits are removed. Actual static AB init count16->12; WASP initializer EB4 joins fall by four. Active IDs are0/1/3/5/6/7/8/9/11/12/13/15. No live synchronization or wait is removed. The dedicated real-trace audit asserts no wait accesses an obsolete ID; four complete-N1024 FP8/INT8 equivalence cases pass, covering later N phases and persistent transitions. Generic phase/work/consumer-VMEM audits also pass. Independent compilation repeats exact text, inputs, addresses, output and time. LDS99KiB and the same WDRA register quotas are retained, without spills. Evidence: [live-barriers](live-barriers/).

This reduces startup/teardown instructions left over from the earlier two-slot B protocol, not the fixed PMD launch startup. The remaining largest first-round A/B and output/mailbox waits still need optimization. No hardware barrier ID or phase restrictions are relaxed.


Production documentation was refreshed after testing to describe the full-K B protocol and twelve live barrier IDs. The tested source is retained unchanged; repeat-proof.json records the tested and production hashes and the documentation-only AST comparison.

## Complete both final K64 MMAC groups before the two M32 epilogues

Current best **15.145996 us / whole 48-CU dispatch MMAC15.036159%**, same1.8GHz/48TG/two outputs per TG scope. This is0.478591% lower elapsed time than commit bbac17c15.218832 us. Four full-N1024 FP8/INT8 equivalence cases and barrier/phase, active-work and consumer-VMEM audits pass; independent compilation repeats exact machine text, frozen inputs, GPU addresses, output and time. LDS remains99KiB, native WDRA actual VGPRs36/9/146/146; no spills. Evidence: [finish-mmac](finish-mmac/). The60% objective remains unachieved.

Previously, C finished K3 for the first M32, scaled and wrote its mailbox, waited for LDS completion, then computed K3 for the second M32. Now both independent K3 MMAC groups run before either M32 epilogue. Output mailbox dimensions, producer drain order, barrier IDs/phases and scaling order ((acc*SA)*SB)*weight are unchanged. This removes one explicit epilogue/wait boundary between the final MMAC groups, without increasing LDS or changing the amount of MMAC work.

CU0/SIMD0: first MMAC times are unchanged(C0 8.129276us,C1 8.789804us); last MMAC ends improve C0 13.871644->13.475772us and C1 14.532172->14.149644us. Full dispatch improves only0.072836us, because output draining remains on the critical path. Cross-M between-MMAC gaps grow C0 2112->2544 and C1 1660->2808 cycles. This is not a claim that all stalls are removed. A C1 ordinal57 gap600 cycles has no intervening instructions from that wave; per-wave issue gaps alone cannot distinguish scheduling competition from instruction/resource readiness. Next-A priority controls are being checked separately.

Numeric-passing negative controls in this round: double A slots15.561884us; prefetch both SB halves after BL15.419548us; wider CN256 phase with native B MLS and metadata prefetch17.195968us; P0 route/expert register carriers15.646952us(extra256B shared conversion scratch); direct output-layout route loading15.378404us(removes that scratch but still slower); independent double SB slots15.927732us. None replaces the best. Completed negative traces were losslessly gzip-compressed to release space; original source, binaries, stats, perf and result JSON remain in the temporary case directories.

## Output strategy counterexamples and next reuse direction

The latest output-strategy controls did not beat commit7820cab15.145996us. Exact results, trace evidence and the unimplemented runtime-gated adjacent-M B-reuse protocol are recorded in [output-strategy-controls](output-strategy-controls/README.md). Production source remains unchanged. Reorder-only18.044980us is a negative control, not evidence of a speedup. The60% target is still active and unachieved.

## BN256 conversation comparison

The retained BN256 12.168060 us approach was transplanted to the original BN512 two-output workload: **16.098424 us**, zero numerical errors, whole-dispatch MMAC14.146577%. It does not replace BN512 best15.145996 us. See [algorithm and trace comparison](bn256-algorithm-control/README.md).

## SB global read before B EMPTY wait

Current best **15.032016 us / 15.150170% whole 48-CU dispatch MMAC**, the unchanged
1.8GHz / two M64xN512 output tiles per persistent TG scope. Previous best7820cab
was15.145996us: **0.752542% less elapsed time** (0.113980us). The60% target is not met.

P1 now computes the channel-scale address and issues its register buffer load
before waiting for the B slot EMPTY. B MLS remains after EMPTY and READY seq;
SB LDS store, LDS completion wait and READY arrive remain after MLS. Register
prefetch cannot overwrite a shared SB generation; completed B/SB reads still
return the shared slot credit. No extra barrier, LDS slot or metadata/global load
is added. A, consumer MMAC, strict scale order, output and persistent protocols
are unchanged. Default native WDRA requests48/16/224/224; LDS remains99KiB.

Independent compilation repeats machine.text, frozen inputs, GPU addresses,
output hash and elapsed time exactly. Four full-N FP8/INT8 x BF16/FP16 equivalence
cases pass, including independent wide A/B/C modes; bounded timing checks2630144
selected outputs with zero mismatches and untouched tail. Actual CU0 A/B generation,
no repeated adjacent wait ID, live-wave/work and no-consumer-VMEM audits pass.
Toolchains remain Triton5b24821ee57eeb7560074cc7979ce7b59604079f and
LLVMfd18eddbaf4a662ae5ce86c516d28e5b16952513; no tracked local changes to either.
LLVM has two unrelated untracked patch/test files, not compilation inputs.

Perf diagnosis is deliberately bounded: CU0 SIMD0 C1 first MMAC advances
8.789804->8.609660us, while C0 moves8.129276->8.171532us. C1 last MMAC ends
14.149644->14.167436us, slightly later; therefore neither earlier C1 startup nor
an individual wait alone explains the whole-dispatch gain. The largest sampled
P1 VMEM wait-to-next-issued-instruction interval grows1578->1934cycles. The change
improves total scheduling/tail behavior without removing every wait. All such
intervals include scheduling effects, not pure memory latency.

Negative controls from the same round: full-M64 direct scatter plus next-B
lookahead17.759752us; batching all four SB panel loads15.418992us; next-M metadata
read before A-credit15.244964us; output LDS swizzle(8,2,8)15.373400us. Numerics pass;
none is promoted. The direct-scatter lookahead added68+ vector register moves in
some inter-MMAC gaps and places a future B readiness dependency before current
output stores. Its lower LDS pipeline count did not imply faster dispatch.

Evidence: [SB-prefetch proof and trace metrics](sb-before-b-empty/).


## Startup and B-order controls after bf26fa0

Retained code remains **bf26fa0,15.032016us**. Moving initial A immediately after
READY1 initialization (before the other independent barrier initializations)
passes bounded numerics and generic A/B wait audits but measures15.103184us.
Prefetching the first expert before initialization and explicitly passing it
through an existing metadata LDS word measures15.425108us, bounded numerics pass.
Directly capturing a new loaded scalar across WASP fails current clean Triton
shared allocation lowering; rank-zero memdescs are also rejected. These are
compile failures, not performance results, and no compiler workaround was used.

Alternating BL/BR phase order with a completed own-READY separator measures
15.899376us and passes generic wait-ID audit, so it is rejected. Without that
separator, actual P1 wait IDs are6,8,8,6,6,8,8,6: **invalid** repeated adjacent IDs.
That initial diagnostic is not an acceptable kernel even though numerics passed.

The runtime adjacent-M B-reuse candidate now has a completed-READY separator
before member1 output handoffs; bounded numerics and generic wait-ID/work audits
pass at15.988336us (14.243984% whole-dispatch MMAC). This supersedes the older
“not yet implemented” opportunity note. It saves B traffic on28 of48 same-expert
pairs in this frozen prefix, but its two retained A tensors and N-major/member
loop ordering increase register/VALU and frontend work, and it loses to bf26fa0.
No paired implementation is promoted. A possible next algorithm is to retain
both B phase tensors across adjacent M members, consume A/metadata one M at a
time, and traverse one BN512 N tile per pair; this would require full-N, odd-M,
different-expert and generation proofs before promotion. It is not a claimed win.

Completed negative raw IT traces were losslessly gzip-compressed (1.083GB freed)
without dropping sources, compiler outputs, perf, stats or results. Exact list:
`/home/zhenx/temp/moe-gemm2-bm64bn512-20261115/compressed-followup-traces.json`.

## Consumer warp geometry M1/N4 — 2026-11-16

Current best **14.873556 us / 15.311577% whole 48-CU dispatch MMAC**, unchanged
1.8GHz and two logical M64/N512 outputs per persistent TG. This is
**1.054150% less elapsed time** than bf26fa0 (15.032016us), a0.158460us
reduction. The60% target <=8.272168us remains unmet.

Only the MFMA consumer warp geometry changes from [2,2] to **[1,4]**;
instruction16x16x32 and tiles-per-warp[1,2] remain. There are still four waves per
role,2P2C, native WDRA requests48/16/224/224, the same producers, slots, A/B
publications, credit/metadata generations, scaling order and routed output.
Actual VGPRs36/9/162/162 (was36/9/146/146), no spills, LDS99KiB unchanged.

With fewer M warp groups, B reads have less replication across M warps; A reads
have more replication across N warps, but A is retained for both N128 phases.
Assembly matrix-DS read instructions fall80->64 across the kernel. Dynamic CU0
LDS instructions fall1312->1248; VALU3004->2948 and SCA2320->2288; VMEM352 and
active MMAC2048 stay unchanged. All48 CUs perform2048 active MMACs. This is a
reduction of actual instruction work, not a reduction of mathematical output.

The timing benefit is not uniform: CU0 SIMD0 C0 first MMAC moves
8.171532->8.431740us and C1 moves8.609660->8.629676us. C1 final MMAC ends
14.167436->14.045116us; the later phase/tail schedule wins overall. C1 between-MMAC
gaps after63 and191 shrink2224/2184->2028/1864cycles, while its cross-M gap after127
grows2760->3292cycles. Issued-instruction intervals include other-wave scheduling.
Bank conflicts stay1024/CU; average LocalMemPipeline.creditStall grows
1152.604->1311.083. Neither a bank-conflict fix nor uniformly shorter waits is claimed.

Independent compilation repeats .text, frozen inputs, GPU addresses, output and
elapsed time exactly. Bounded timing checks2630144 outputs with zero mismatches,
unchanged hash and untouched tail. Four full-N1024 FP8/INT8 x BF16/FP16 equivalence
cases pass, including independent forced-wide A/B/C and nonunit scale strides.
A/B phase, adjacent wait-ID, live-wave/work/8-cycle union and consumer-no-VMEM
load audits pass. Compiler sources and versions are unchanged: Triton5b24821,
LLVMfd18eddb, control-only-ebarrier clang wrapper. See [complete proof](consumer-m1n4/).

Other numeric-passing controls rejected this round: adjacent-M/two B register
caches with P0 scatter17.337192us; one B register cache plus retained B1 LDS/direct
scatter18.269048us. Their generic wait-ID/work audits pass, but first-batch cache
hit rates fall to about79% vs84.5% and frontend/epilogue costs increase. The
all-register direct-output variant exceeds the native per-SIMD VGPR budget and
has no performance result. No resource limit or compiler check was bypassed.

A16KiB native MLS Bright-K128 lookahead stage was also tested with the original
M assignment: early15.417880us; after current B publications15.193812us; just before
last current Bright15.423440us; uniform first-stage path15.105408us. All bounded
numerics pass. Generic wait/work audits pass for all tested variants. These
remain temporary controls; they do not replace the retained source.

