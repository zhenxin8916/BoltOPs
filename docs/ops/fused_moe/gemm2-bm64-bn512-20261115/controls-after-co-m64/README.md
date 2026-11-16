# Controls after full-M64 result publication

All measurements use the unchanged toolchain recorded in [provenance](../provenance.json),
1.8GHz,48CUs and two independent BM64/BN512 outputs per persistent TG.
The retained source remains `97cb4cc`: **14.776256us /15.412402% full-dispatch MMAC**.
Every row below passed2630144 bounded output comparisons with zero mismatches
and untouched tail. These are rejected single-run controls, not newly promoted
operators; full-N regression and independent repeat were not run because none
beats the retained result. [Machine-readable evidence](summary.json) includes
source/HSACO/perf hashes, complete dispatch spans, cache/stall counters and actual
audit results. No compiler source or production kernel changed in this round.

| Algorithm | Full dispatch us | Whole-dispatch MMAC |
|---|---:|---:|
| Two SB generations, direct buffer-to-LDS |14.829632|15.356929%|
| Direct SB with vec4 shared layout |14.829632|15.356929%|
| Separate B/SB READY; return B credit after SB read |15.092064|15.089891%|
| Prefetch both SB halves before BL; wait groups1/0 |15.013112|15.169247%|
| P1 scatters right output one phase behind B, with metadata credits |16.371976|13.910209%|
| Adjacent M pairs, no explicit B reuse |17.917100|12.710628%|
| Single M64 accumulator, consumer warps M1/N4 |15.130428|15.051630%|
| Single M64 accumulator, consumer warps M2/N2 |15.159896|15.022372%|
| Read one metadata expert word for P0 |15.428444|14.760892%|
| P1 expert via SMEM with explicit completion |15.042024|15.140090%|
| P0/P1 expert via SMEM with explicit completion |15.118196|15.063808%|
| Right LDS read before left store |14.950284|15.232995%|
| Batch both LDS result reads before completion wait |15.003660|15.178803%|

The direct SB path emits buffer_load_dwordx4 to LDS, but the lowered async
completion adds VM/LGKM waiting and a four-wave rendezvous. The intended
register-transfer reduction does not shorten the complete dispatch. Separating
B READY from SB READY is slower too; the consumer retains B credit until its
SB read completes, preserving two-generation overwrite safety. This does not
justify dropping that dependency or removing a wait without a memory proof.

Splitting output scatter leaves P0 lighter but adds metadata-read credits, right
scatter and frontend work to P1. Its legal variant takes16.371976us. It does not
prove P1 is universally the bottleneck; it rejects this specific schedule.

Adjacent-M assignment without explicit reuse expands the initial expert footprint:
L2 hit rate falls84.61% ->79.77%, and time rises17.917100us. Mere adjacency is
not enough; any future reuse schedule must account for the first-batch cache
working set as well as removing repeated transfers.

In CU0 SIMD0, baseline P1 first B MLS is6.343404us; C0 first MMAC8.009180us;
C1 first MMAC8.471772us; P0 first output store9.717212us. P1 expert SMEM moves
first B MLS to6.285580us but first P0 output store to9.941836us, with full time
15.042024us. Therefore earlier first B issue alone is not sufficient: the
consumer/epilogue/output chain must also improve. These are issued-instruction
timestamps, not isolated instruction latencies.

The attempted missing-wait scratch SMEM variants are excluded: the assembly
semicolon commented out the wait. Corrected newline-separated instructions with
early-clobber destination and explicit lgkmcnt0 pass numerical validation.
The base pointer includes the offset and the SMEM immediate is zero; PMD-004's
SOE register-plus-immediate form is not used. No new model issue is claimed.

The60% elapsed-time reduction target (<=8.272168us) remains unmet. Retained
full-N FP8/INT8 validation, exact repeat proof and current perf are in
[co-full-m64](../co-full-m64/).
