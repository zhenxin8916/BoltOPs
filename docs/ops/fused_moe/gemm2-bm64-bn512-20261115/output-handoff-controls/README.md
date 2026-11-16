# Output handoff and P1 loop controls — 2026-11-16

Retain the algorithm in code commit `98a1e1e`: **14.713428 us**, whole-48-CU
dispatch MMAC **15.478215%**, at **1.8 GHz**, 48 persistent TGs, two independent
BM64xBN512 output tiles per TG. Production runtime AST still matches that measured
source after removing docstrings; unrelated documentation edits are preserved.

| Control | Full dispatch us | Whole dispatch MMAC | SGPR spills / VGPR spills / scratch bytes |
|---|---:|---:|---:|
|Wait for output EMPTY immediately before first LDS output store|14.769584|15.419365%|0 / 0 / 0|
|Static P1 N traversal, original per-phase SB staging|14.961404|15.221673%|0 / 0 / 0|
|Pair SA/weight metadata LDS stores|15.286664|14.897796%|0 / 0 / 0|
|P0 priority 1 during output drain/scatter|14.842976|15.343123%|0 / 0 / 0|

All four controls pass 2,630,144 selected FP8 outputs, zero mismatches and untouched
tail. Audits verify 98,304 MMAC instructions over 48 CUs (2,048 per CU), 16 live
waves at CU0 (four per SIMD), A/B generations across both outputs, legal AB wait
IDs, zero consumer buffer/global loads and nonempty SQTT. Every compiled binary
has SGPR/VGPR spill counts zero and private segment/scratch bytes zero. Evidence
and temporary artifact locations are in [summary.json](summary.json). These slower
controls were not independently repeated or full-N validated and are not promoted.

The first control keeps output-slot ownership unchanged: register-only scale and
conversion run before waiting for output EMPTY, and the first LDS store still
waits for EMPTY. No store can overwrite the prior output before P0 returns the
credit. CU0 SIMD0 C0 cross-M MMAC gap shrinks 2780 -> 2340 cycles; C1 shrinks
2700 -> 2480 cycles. Last MMAC issues move 13.084348 -> 12.861948 us for CU0 C0
and 13.740428 -> 13.275612 us for C1. The last CU0 buffer store also moves earlier,
14.209692 -> 13.760444 us. Nevertheless full dispatch takes 14.769584 us.

That local improvement is not a global speedup. CU0 totalCycles falls 17958 ->
17159 while the largest per-CU totalCycles belongs to CU25 in the baseline and
CU39 in this control. These durations alone do not reveal those CUs' exact last
instructions or separate their start times. CU0 tracing cannot establish that P0
is the unique 48-CU critical path. The isolated P0 priority control also loses.
Next critical-path work must include the slow CUs and write completion, rather
than accepting a shorter CU0 MMAC interval as sufficient proof.

The pair-store control uses the existing physical SA/weight metadata addresses
with a rank-3 shared view; consumers and metadata generation ownership do not
change. The static-P1 control retains all B/SB EMPTY and READY ordering. Neither
reduces the measured dispatch time. Failed sources remain outside production.

The target <= 8.272168 us is still unmet.
