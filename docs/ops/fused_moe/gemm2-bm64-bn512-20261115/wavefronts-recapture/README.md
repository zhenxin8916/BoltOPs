# Corrected Wavefronts capture — 2026-11-16

The previous named perf omitted Wavefronts because launch.py enabled
`StatLog,IT0,Perf` without `TT0`. IT0 produced a text instruction trace for our
analysis, but it did not embed the SQTT stream used by the GUI. The earlier
8-KiB file had timing/metadata only. This was a capture configuration error.

The retained source remains `97cb4cc`. A fresh capture with
`StatLog,IT0,TT0,Perf` has been verified to contain a nonempty type11 SQTT
section:72050 raw bytes,10844 compressed bytes; full perf19167 bytes.
This records CU0 wavefronts. All48CUs remain enabled and counted for dispatch
elapsed time and MMAC utilization. No GUI interaction is claimed; the SQTT
section is validated directly from the written HYGPERF file.

[Corrected file location and hash](perf-location.json).
[Capture configuration](launch.py), [SQTT validator](audit_wavefronts.py),
[structural SQTT audit](wavefronts-audit.json), [repeat proof](repeat-proof.json).

Recompilation matches the old machine .text, frozen inputs, GPU addresses,
2630144 output comparisons, untouched tail, full48CU MMAC counts and elapsed
**14.776256us** exactly. Full-dispatch MMAC remains**15.412402%**. Kernel code
and compiler/toolchain are unchanged. Existing full-N equivalence validation
therefore applies to the identical machine code. The hardware wait-ID/live-wave
and instruction-work audit also passes on this new capture.

Future successful non-compile-only launch.py captures enable TT0 and require a
nonempty SQTT section. The previous stats-only perf remains in its original raw
measurement directory for provenance; use the new WAVEFRONTS filename for GUI
inspection. Correcting the capture does not constitute a kernel performance gain.
