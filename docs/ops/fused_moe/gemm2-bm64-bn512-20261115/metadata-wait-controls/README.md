# Metadata-wait and publication controls

Retained best is98a1e1e:14.713428us /15.478215% whole48CU dispatch MMAC.
All cases below pass2630144 selected outputs, zero mismatches and untouched tail;
generic wait-ID/live-wave/MMAC-work audits pass. These rejected single runs
were not independently repeated or full-N regression tested. The promoted gain
has separate repeated/full-N/SQTT evidence under [metadata-after-k3](../metadata-after-k3/).
Same toolchain/provenance,1.8GHz,48CU,two independent BM64xBN512 outputs per TG.

| Control | Full dispatch us | Whole dispatch MMAC |
|---|---:|---:|
| Batch all four metadata fields in one LDS store |15.039800|15.142329%|
| Separate SA/weight READY before route/expert publication |15.485712|14.706305%|
| Join packed M32 halves into one C LDS store on97cb4cc |14.958068|15.225068%|
| Move metadata wait after K2 |14.787376|15.400812%|
| Join packed C store on98a1e1e |14.792380|15.395602%|

[Evidence](summary.json) preserves source hashes, raw directories, perf hashes,
output comparisons, hardware/work audits and counters. Combined metadata
publication shortens the first CU0 SIMD0 C0 intra-K metadata gap1096->140cycles,
but cross-M grows2092->2548 and first N handoff1584->2512; complete time rises.
This supports measuring the complete dispatch, not promoting earlier first issue
or one smaller gap alone. Register generation and final READY completion are
preserved in all numeric-passing publication controls.

Combining both C stores removes consumer-local EB8/9 joins, yet takes14.958068us
on97cb4cc and14.792380us on98a1e1e. Fewer joins alone do not prove a gain.
The combination is excluded from production. Metadata after K2 takes14.787376us;
after K3 is the independently verified improvement retained at98a1e1e.
