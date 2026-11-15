# Output strategy controls and next B-reuse design

All cases use the same frozen FP8 inputs, 1.8GHz, 48 physical TGs, two BM64/BN512 output tiles per TG, and full48CU dispatch timing. Production remains commit7820cab15.145996us. Results are negative controls, not promoted implementations; no full-N qualification is claimed for them.

| Algorithm | Dispatch us | LDS KiB |
|---|---:|---:|
| P0/P1 split output drains |17.239336|99|
| Combined left/right scatter |15.333368|99|
| Native packed consumer direct stores |15.693656|83|
| M32 row split / shared B |17.423928|115|
| Custom vec8 consumer direct stores |21.568908|115|

The custom vec8 conversion still introduces shared-memory scratch in this clean backend, despite unchanged warp bases; it cannot be treated as a cheap register-only shuffle. The split-writer result shows that simply moving C1 stores to P1 competes with B production. The row-split result trades lower A reads for replicated larger B reads and does not improve time. Exact metrics, numerical hashes and generic trace audits are archived by case. Temporary sources/binaries/perfs remain at raw_directory in results.json.

## Runtime-gated B reuse across adjacent M pairs (not implemented)

The existing stride48 mapping compares M[p] with M[p+48]. In the fixed96-M prefix there are0/48 same-expert pairs. Adjacent mapping (2p,2p+1) covers exactly the same96 M tiles and selected output rows and has28/48 same-expert pairs. A runtime same-expert gate could avoid28*256*512=3,670,016 bytes of B loads. This is a theoretical byte opportunity, not a performance result or a claim about arbitrary routing distributions. Reordering alone is insufficient; the current-toolchain reorder-only control is measured separately.

A general design keeps2P2C and BM64 computations: loop over adjacent M pairs, then N phases, then the two M members. P0 preloads two independent full-A slots and both metadata generations, then drains this pair's M32 outputs in N/member/panel order. P1 compares the two actual expert IDs at runtime: emit B/SB once per N phase when equal; emit twice when different. Each C retains both A tiles and metadata in registers and immediately reuses its N-phase B registers for member1 when equality holds. This avoids caching all N phases and therefore supports full N traversal. Odd final M counts must skip the second member without OOB loads or unmatched waits.

Resource/protocol sketch: A32+B64+C16+MA2+SB1=115KiB, plus a small LDS reuse flag. A slot0 uses READY1/EMPTY3; slot1 READY10/EMPTY14, EMPTY counts8, phase=pair_epoch&1. MA READY12/0 uses the same pair phase. B keeps READY5/7 and EMPTY6/8; its global epoch advances only on actual B publications, identically in P1 and C. Co retains READY9/13 and EMPTY11/15 with M32-panel generations. IDs2/4 stay unused. All init/first-wait/adjacent-ID and scope limits must still be audited.

The equality flag can be published with first B/SB so consumers need no new global load. C1 visibility must follow P1's completed first-half LDS stores before second-half publication. Keep each pair's MA until all its outputs drain; this first prototype does not prefetch the next pair's MA, avoiding an unproven cross-pair lifetime shortcut. Validate both equal/different expert branches, odd M tails, full N, INT8, output dtypes and wide addresses before considering promotion. Acceptance remains elapsed time for two logical BM64 outputs/TG; no kernel may skip selected rows or quietly change the comparison to per-tile throughput.

Current-best reorder-only control: **18.044980us**, L2 hit rate79.7157%, bounded numerics match exactly and generic wait/work audit passes. It is slower than15.145996us and is not promoted. Actual B reuse must overcome this loss as well; the theoretical saved bytes do not establish a net speedup.
