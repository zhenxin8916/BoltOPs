# GEMM2 BM64 / BN512 / BK64 tuning

2026-11-15. Dedicated operator: `boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn512xbk64.py`. No change to BM128 operators or framework selection. Full N traversal is supported by the direct entry; timing intentionally measures only the first BN512 panel.

**Current best: 15.423440 us / whole-dispatch MMAC14.765681%, output-lds-8-1-8. The 60% target remains open.**

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
