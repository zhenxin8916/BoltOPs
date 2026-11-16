# Scale publication, batching and next-expert controls — 2026-11-16

Retained best remains code commit`98a1e1e`: **14.713428us /15.478215% whole48-CU
dispatch MMAC**, at1.8GHz,48 persistent TGs,two independent BM64xBN512 outputs
per TG. No production algorithm change from this round is promoted. Current
unrelated working-tree docstrings/comments are preserved; removing docstrings
from its AST matches the retained measured operator. Compiler/model versions
remain those in [provenance](../provenance.json).

All nine single-run controls pass2630144 selected outputs with zero mismatches
and untouched tail, expected98304 active MMAC instructions across48CUs, generic
AB wait-ID/live-wave/work audits, and nonempty SQTT capture verification. Extra
scale-READY generations are independently checked on the two-output trace. These
are rejected controls, not full-N or independently repeated operator validation;
neither is claimed because every control is slower than the retained result.

| Algorithm | Full dispatch us | Whole dispatch MMAC | LDS KiB |
|---|---:|---:|---:|
| Separate B/SB READY; read SB after MMAC |14.892460|15.292141%|115|
| Publish both B halves before separate SB publications |15.254972|14.928746%|115|
| Prefetch next expert before current N traversal |14.760688|15.428658%|115|
| Whole BN512 SB batch, separate READY/EMPTY |15.020340|15.161947%|116|
| Whole BN512 SB batch, both B EMPTYs protect reuse |15.085392|15.096565%|116|
| Independent N256 SB batch per consumer |14.812952|15.374221%|116|
| Independent SB batches; static P1 N traversal |14.722324|15.468862%|116|
| Separate READY with batched initialization of added counters |15.008664|15.173742%|115|
| Static P1 SB batches; next expert after last B-left |14.922484|15.261373%|116|

[Evidence](summary.json) preserves source hashes, original perf hashes, SQTT audits,
raw directories, numerical checks, hardware/work/phase audits and cache/stall
counters. Rejected source remains outside production. Full48CU dispatch time
includes model startup; CU0 local intervals are not the acceptance denominator.

The original CU0 SIMD0 P1 B-right sequence issues MLS at7.030620..7.177404us,
then vmcnt0 at7.178516; its next DS store issues7.978044 and B READY arrives at
8.102588us. The0.799528us issue interval therefore couples scale completion and
B publication in this implementation. It is not proven to be0.799528us of
recoverable dispatch latency: B DMA and other waves also progress in that time.

Separating READY retains private SB EMPTY ownership: each P1 store waits for
its matching consumer's completed SB read; the consumer reads scale only after
MMAC, waits for LDS completion, then returns that credit. The per-consumer
READY10/14 and EMPTY2/4 stay inside16 AB IDs. First CU0 SIMD0 C0/C1 MMAC moves
8.009180/8.471772 ->8.227132/8.700844us, and full time rises14.892460us. Sending
both B halves before scales takes15.254972us. Batching only the four additional
first-use AB initializations does not rescue the split:15.008664us. No native
barrier limit or first-wait/adjacent-wait restriction is bypassed.

The whole-BN512 SB batch reserves2KiB LDS for512 FP32 values over two N128
phases, with READY2 and EMPTY4(count8) returned by both consumers only after
second-phase reads complete. Both consumers use local register copies for scale
arithmetic. A co-ready variant instead waits both prior B EMPTYs before reusing
the whole tile, then publishes B-left only after the entire scale store completes.
Both are numerically correct in the bounded test, but slower. They add no
negative offsets or model-specific memory workarounds.

Independent per-consumer N256 batches avoid that cross-consumer reuse restriction:
P1 updates only that consumer's two128-scale rows after its B EMPTY, and the
consumer loads scale alongside B before returning EMPTY. Thus the next batch
cannot overwrite either N phase while scale remains in use. No extra AB IDs are
required. CU0 VMEM instructions fall352->336, LDS1248->1232, SCA2172->2164;
full time is14.812952us. Static P1 N traversal removes dynamic SB branch work
and reaches14.722324us, still0.008896us slower than the retained source. It is
not promoted or reported as a gain.

The whole-BN512 batch reduces CU0 VMEM to328 and LDS1224 yet takes15.020340us
with separate SB ownership, or15.085392us with both B EMPTYs protecting reuse.
Fewer loads/stores alone do not establish a win. Next-M expert prefetch before
current N traversal takes14.760688us. Prefetch after the final B-left of a static
SB-batched traversal takes14.922484us. These results reject the tested placements;
they do not establish that every expert prefetch or SB staging scheme is invalid.

The60% lower-time target <=8.272168us remains unmet. The next investigation
should follow actual remaining A/B and output handoff intervals rather than
counting one VM wait as a complete serialized cost.
