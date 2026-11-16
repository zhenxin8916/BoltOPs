# Startup, A-credit and K128 stage controls — 2026-11-16

Retained production code: `97cb4cc`, **14.776256 us /15.412402% full-dispatch
MMAC**, at1.8GHz,48CUs, two independent BM64xBN512 output tiles per persistent TG.
The compiler/model versions remain those in [provenance](../provenance.json).
None of these single-run controls improves the retained result. Production
source is unchanged. These controls passed2630144 selected output comparisons,
zero mismatches and untouched tail, plus the generic hardware wait-ID/live-wave/
MMAC-work audit. They were not independently repeated or full-N regression tested
because they are slower or equal. Compile-only failures have no timing claim.

| Algorithm | Full dispatch us | 48-CU dispatch MMAC |
|---|---:|---:|
| Batch first-use native AB initialization; retain distributed initial credits |15.327808|14.857806%|
| Batch first-use AB initialization and exact initial EMPTY credits |15.015892|15.166438%|
| Two K128 B stages, left-first second half |15.589684|14.608224%|
| Two K128 B stages, right-first second half |15.534640|14.659986%|
| Overlap second K128 B read with K1 MMAC |15.267204|14.916785%|
| P1 priority2 for first N phase only |14.793492|15.394445%|
| Initial A priority0, next A priority2 |14.776256|15.412402%|
| Issue initial A after READY1 initialization |14.903580|15.280731%|
| Return A credit at LG8 while B LDS reads remain outstanding |14.913588|15.270477%|
| Remove consumer scheduler barriers only |16.404224|13.882863%|
| Wait first B READY before starting A LDS reads |15.036464|15.145689%|
| Plain contiguous metadata LDS layout |15.547984|14.647404%|
| Compact metadata read layout then convert to MFMA rows |14.776256|15.412402%|

[Machine-readable evidence](summary.json) contains numerical results, source and
HSACO hashes, full HYGPERF spans, cache/stall counters, and hardware/work audits.
Traces for rejected cases were losslessly gzipped; stats and original perf remain.

Batching AB initialization reduces initializer joins, and first A issue becomes
earlier, but first B and downstream MMAC become later. Initial A issue alone is
not an adequate optimization criterion. The two K128 B stages retain the same
64KiB B capacity and115KiB total LDS. All16 AB IDs are initialized, with original
credit totals and legal alternating wait IDs; neither stage order beats full-K B.

Returning A at LG8 before B finishes passes the bounded numerical and hardware
checks, but full time rises14.776256 ->14.913588us. CU0 SIMD0 C0 cross-M MMAC gap
rises2092 ->2960cycles; C1 rises2292 ->2948cycles. The extra wait/credit schedule
changes downstream issue timing. A reduced early dependency does not imply a
smaller complete dispatch. These are issued-instruction gaps, not isolated
hardware barrier latency.

Removing consumer scheduler barriers makes time16.404224us. Plain row metadata
LDS takes15.547984us. Compact metadata reads followed by layout conversion produce
an equal14.776256us result, so are not promoted. No equal-time result is reported
as a gain.

The retained trace exposes another specific dependency: CU0 SIMD0 C0 issues its
first metadata READY wait at8.078124us and its next LDS read at8.667484us,
a0.589360us issue interval. P0 publishes SA/weight/route/expert as four shared
stores. Its generated code has LG0/four-wave EB6 joins between field stores,
before metadata READY at8.594092us. This is evidence to investigate batching
publication, not evidence that consumer reads alone are the cause. A combined
publication must preserve field order/addresses, generation ownership, final
store completion and READY; the next control follows that chain.

A buffer-to-LDS initial-A control using native MLS layout fails HCU lowering;
a swizzled-A variant fails LLVM translation on an unrealized conversion cast.
Neither executed the GEMM or has a performance result. The attempted metadata
rank/slice prototypes also fail descriptor rank requirements; they are excluded
from numeric/performance comparisons, and no compiler check was bypassed.
