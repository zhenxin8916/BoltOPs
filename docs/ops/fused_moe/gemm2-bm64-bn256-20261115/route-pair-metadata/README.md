# P0 route preparation and paired metadata publication — 2026-11-16

Measured result: **11.794984 μs, whole-dispatch 48-CU MMAC 9.654002%** at 1.8 GHz, 48 physical persistent TGs, two independent BM64 output tiles each. Previous row32 version: 11.954000 μs / 9.525581%; latency decreases **1.3302%**. The ≤9.602 μs target remains unmet. BM64 does half the padded work of two BM128 output tiles; these times do not establish equal-work throughput gains over BM128.

## Retained algorithm

Keep 2P2C, four waves per partition, K256 MMAC, two A slots, one B slot per consumer, two SB generations per side, and 100 KiB LDS. P0 decodes CK token/rank to output route once and replaces routes for negative experts with -1. Consumers no longer read the expert metadata vector or repeat route decoding. The valid-route bound mask remains in the consumer. This preserves non-CK route semantics too, though timed and full-N acceptance cases use CK sorting.

P0 joins SA and weight into a [2,64] register panel and publishes both in one shared-memory store operation through a matching SharedLinearLayout view. Their physical addresses remain unchanged. Decoded routes occupy the original third metadata vector; the fourth reserved vector is unused. No new barrier IDs, speculative slot reuse, extra shared memory, changed scale multiplication order or MMAC hot-loop scheduling policy is introduced. Consumers still read A/B/SB and metadata, complete LDS reads, then return A/B EMPTY credits before computing. The source changes only comments beyond the exact measured archived kernel; executable AST equivalence was checked before promotion.

## Startup versus tail/head handoff

The screenshot's initial gap remains: the first A LDS reads precede B readiness and metadata readiness. Simply moving the metadata wait after MMAC starts calculation sooner but increases later costs. It is rejected. This update instead reduces duplicated metadata/scatter preparation and preserves the producer/consumer ordering.

CU0 SIMD0 issue intervals, in cycles; these include scheduling gaps and are not isolated instruction latencies:

| Interval | Previous C0 | New C0 | Previous C1 | New C1 |
|---|---:|---:|---:|---:|
| First MMAC, relative to CU0 first wave start |6883|7031|7395|7855|
| MMAC end → first store |1092|960|368|628|
| First → last store issue |1348|1360|1644|1452|
| Last store → next MMAC |784|740|976|704|
| Complete inter-output MMAC gap |3224|3060|2988|2784|
| Final MMAC end → wave end |2160|1856|1068|1148|

All eight CU0 consumer waves improve their inter-output gaps: C0 range 3200–3224 → 3060–3092 cycles, C1 range 2988–3280 → 2732–2868 cycles; see [all consumer handoffs](all-consumer-handoffs.json). SIMD0 gaps shrink by 164 and 204 cycles. Final C1 tail grows 80 cycles (~0.0445 μs), while C0 tail shrinks 304 cycles. The first MMAC starts **later**, by 148/460 cycles: this is not a claim that startup starvation was eliminated. The whole-dispatch repeat establishes the net benefit, with the sampled handoff preserved/improved. The initial P1 left/right publication serialization remains a follow-up bottleneck.

## Verification and reproducibility

- Same binary `.text`, source, frozen inputs, addresses and launch options repeated at exactly **11.794984 μs**; see [repeat proof](repeat-proof.json).
- Timed test checks 1,315,072 elements with zero errors and untouched masked tail.
- FP8 and INT8 full-N cases each check 197,632 elements with zero errors: five M iterations per physical TG and two N tiles per M.
- A/metadata/B phase audits and SB effective-address/completion/reuse audits pass on the timed and both full-N cases. Timing audit has 16 live waves/four per SIMD, 1024 active MMACs per CU and no exec=0 MMACs.
- Independent frozen routing/input seed3209: previous row32 **11.812220 μs → 11.741608 μs**; numerical check passes. This paired control is not substituted for the primary acceptance input.
- Actual WDRA VGPRs and compiler options are in [compiled metadata](compiled.json). LLVM/Triton/PMD revisions, local changes and flags are recorded in [provenance](provenance.json); no compiler changes were made for this update.

[Perf](GEMM2_BM64_BN256_K256_GRID48_OUTPUT96_2M_PER_CU_ROUTE_PAIR_METADATA_GC1800_CU0.perf), [exact measured source](kernel.py), [metrics](first/metrics.json), [phase intervals](first/phase-analysis.json), [FP8 result](full-n-fp8/result.json), [INT8 result](full-n-int8/result.json), [seed3209 result](seed3209/result.json).

## Rejected controls

| Control | Time μs | Reason |
|---|---:|---|
| Separate SB readiness; issue both B panels before SB wait |12.235892|Slower; C1 inter-output gap 3232 cycles |
| Whole four-vector metadata store |12.143596|Slower |
| Read metadata after MMAC |12.032952|C1 inter-output gap grows to 3412 cycles |
| Independent SB plus late metadata |12.503884|Slower |
| Whole metadata plus late read |12.268696|Slower |
| Late metadata without post-MMAC scheduling fence |12.311508|Slower |
| P0 route decode, separate SA/weight stores |11.832236|Improves baseline but slower than retained paired store |
| P0 route decode plus late metadata |11.930092|Slower than retained version |

All controls passed the primary numerical check. They remain scratch experiments and are not promoted. [Compact evidence](controls.json).

Archived compiler assembly/source are preserved byte-for-byte for hash verification; their original whitespace is retained.
