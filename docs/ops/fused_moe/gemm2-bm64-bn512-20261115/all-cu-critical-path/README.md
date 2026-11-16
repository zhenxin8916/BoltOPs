# All-CU critical path and real SIMD MMAC idle time — 2026-11-16

Retained measured algorithm: code commit `98a1e1e`, **14.713428 us**, whole-48-CU
dispatch MMAC **15.478215%**, **1.8 GHz**, 48 persistent TGs, two independent
BM64xBN512 output tiles per TG. This investigation changes no production algorithm.
Other concurrent working-tree changes are preserved and are outside these controls.

## All-CU recapture is the same experiment

`NIT` emits timestamped disassembly for every CU; `TT0` retains nonempty CU0 SQTT.
There are exactly 48 instruction traces, with global CU IDs read from WAVE_START
records rather than guessed from the SE-local filenames. Each CU has 16 waves,
four per SIMD, and each SIMD contains 512 MMAC instructions from its two consumers.
The union audit verifies all 192 physical SIMD timelines, 98,304 MMAC instructions,
no overlapping eight-cycle MMAC intervals, and exact busy-plus-idle accounting.

Baseline and output-EMPTY-late recaptures have identical `.text` hashes, input
hashes and GPU buffer addresses to their earlier IT0 captures. Full dispatch times
are unchanged: 14.713428 and 14.769584 us. Proof files are included. Optional NIT
HTML postprocessing is unavailable in this PMD installation; timestamped traces,
statistics and independently validated nonempty SQTT are present and usable.

| Version | Last CU | Last MMAC issue on that CU (us) | Last buffer-store issue on that CU (us) | Last endpgm / dispatch end (us) |
|---|---:|---:|---:|---:|
| Retained baseline |25|13.838840|14.393172|14.713428|
| Output EMPTY after conversion |21|14.027880|14.516048|14.769584|
| C1 MMAC priority 1 |24|13.762668|14.457112|14.824072|
| P1 expert via buffer load |38|14.049564|14.537732|14.720100|
| P1 B-right first for every N phase |4|14.306992|14.915812|15.173796|

The end of the baseline is P0 on CU25. Its last buffer-store issue precedes its
last endpgm by 0.320256 us. That interval includes outstanding stores and ending
control flow; it is not a measured isolated memory latency. CU0 is not the last CU.
Per-CU totalCycles alone also does not identify dispatch finish: CU39 has the
largest local duration in the EMPTY-late case, but CU21 actually ends last.

For CU21, moving output EMPTY later delays the final C1 MMAC from 13.878316 to
14.027880 us, and the last store from 14.366484 to 14.516048 us. CU0 improves at
the same time. This is why accepting a shorter CU0 interval would select a slower
48-CU dispatch.

## A single wave's gap need not be idle compute capacity

In the EMPTY-late capture, CU21 SIMD0 C1 MMAC ordinal180 issues at tick12495544;
ordinal181 issues at12789112. With period556 ticks/cycle and an eight-cycle MMAC,
its apparent gap is520 cycles. During the same gap, C0 on **the same SIMD** issues
64 MMAC instructions, occupying512 cycles. Almost all that interval is useful
MMAC work by the other consumer, not an empty arithmetic pipeline. There is no
intervening C1 instruction whose movement alone recovers520 idle cycles.

The retained CU25 SIMD0 does have a **real union gap of2052 cycles** between the
last MMAC of output M0 and the first MMAC of M1, tick10531752..11672664. C0/C1
are both idle for MMAC in that interval. That is a stronger optimization target
than the per-wave handoff gaps. `all-cu-mmac-union.json.gz` distinguishes union gaps
from per-wave gaps; its MMAC-span percentage is explicitly not the whole-dispatch
acceptance metric.

## Controls and validation scope

C1 priority changes only its MMAC interval and restores priority before epilogue.
P1 expert buffer loading uses a one-element distributed tensor and reduces it to
the uniform expert index; the scalar-offset call is rejected by the Gluon helper
and was not executed. B-right-first keeps fixed per-consumer slots and issues
EMPTY waits8,6 in alternating order, without new counters. All three bounded
controls are slower, so none is promoted or claimed as full-N validated.

The next-A control moves the next M's A LDS read ahead of the current last-phase
epilogue. A READY is still awaited, and EMPTY is returned at next-M first-phase
consumption, so P0 cannot overwrite an A generation still in use. The uncompressed
control passes all selected outputs and CU0 A/B/live-wave/active-work audits but
takes14.890236 us. Actual consumer registers rise155 ->224 without spilling.
A is replicated across all four N-direction consumer waves in the M1/N4 layout:
16KiB of A storage does not imply only16 additional VGPRs per lane in that layout.

The following next-A variants also lose; no consumer spills in any of them:

| Consumer next-A algorithm | Full dispatch us | Actual registers C0/C1 |
|---|---:|---:|
| Read next A before last-phase epilogue |14.890236|224/224|
| Carry next A packed into uint32 words |15.095956|220/220|
| Unroll short N traversal |15.105408|224/224|
| Reset obsolete next-A carrier at each M start |15.229952|168/168|

The reset preserves the current A value separately, then makes the default
next-A carrier zero until a real next-M prefetch is available. This removes
unnecessary retention on the no-prefetch branch and reduces actual registers
224 ->168. It is numerically correct and phase-audited but slower. Fewer live
registers alone does not prove lower dispatch time. The packed experiment's first
attempt had a distributed-layout join mismatch, and the static-N attempt initially
redeclared a constexpr inside static_range; both were compile-only failures,
corrected before these measured runs. They are not numerical/model failures.

All successful controls check2,630,144 selected outputs, zero mismatches and an
untouched tail. SGPR/VGPR spill counts, scratch size and private-segment bytes are
zero. The temporary execution harness now rejects those fields before a new
kernel execution. All-CU NIT omits operand values, so the new union audit does not
claim to replace the IT0 register/phase audit or independent full-N tests. Slower
controls do not receive those promotion-only tests. Existing baseline full-N
proofs remain linked from the main report.

Compiler branches/commits and local changes are in [provenance](provenance.json).
Per-case launch metadata, compiled metadata, numerical checks, nonempty SQTT,
all-CU timing, union intervals and raw artifact paths are recorded alongside
[summary.json](summary.json). The <=8.272168 us target remains unmet.

Large per-CU timing and union JSON snapshots are losslessly stored as `.json.gz`;
the analysis scripts emit the uncompressed originals in the raw directories.
