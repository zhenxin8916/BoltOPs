# BM32 single-resident static VGPR allocation, 2026-11-17

Code parent: 243f9577. This commit changes BM32 default `wasp_wdra=True` to `False`. Preserve 2P2C/all16 waves, A2/B3, independent scales, early first-quarter store, A1-before-next-metadata, producer exclusion and consumer priorities. Two sequential outputs per TG remain; two resident hardware TGs are not required.

GEMM1 M128/N512/K2048, BM32/BN256/BK256, grid48/96 logical outputs, cold B, 1.8 GHz. B bandwidth uses48 MiB divided by global wave-active duration. These are PMD measurements.

| Algorithm | Dispatch us | B active TB/s | CU0 merged gap cycles | CU0 individual mean / max cycles |
|---|---:|---:|---:|---:|
| 243f9577 WDRA | 36.540320 | 1.566657 | 703 | 1368 / 1420 |
| Async bootstrap in A0 | 36.979004 | 1.545552 | 670 | 1327 / 1340 |
| Dedicated async bootstrap | 37.143580 | 1.537781 | 612 | 1186.5 / 1448 |
| Scalar count | 36.635952 | 1.562007 | 730 | 1561 / 2128 |
| Scalar count and first expert | 36.607040 | 1.563410 | 740 | 1332 / 1460 |
| Static allocation, this commit | 36.217284 | 1.582569 | 141 | 1269.5 / 1640 |
| Independent static repeat | 36.217284 | 1.582569 | 141 | 1269.5 / 1640 |
| Static with warmup | 36.330708 | 1.576945 | 913 | 1472 / 1716 |
| Static, store after B read | 36.472488 | 1.569971 | 717 | 1239.5 / 1592 |
| Static, late stores | 36.355172 | 1.575738 | 926 | 1475.5 / 1528 |
| Static P0 priority2 | 36.223400 | 1.582265 | 135 | 1306 / 1612 |
| Static P0 priority3 | 36.242304 | 1.581325 | 168 | 1325 / 1664 |
| Static max ILP | 36.247308 | 1.581077 | 323 | 1133.5 / 1332 |
| Static no regpressure | 36.217284 | 1.582569 | 141 | 1269.5 / 1640 |

Static allocation repeats identically: **36.217284 us total,31.803756 us wave-active,1.582569 TB/s**. Against243f9577:0.323036 us /0.8841% faster. Against original639b287 (36.580352 us):0.9925% faster. CU0 K-hot MMAC16.4481%, whole-active MMAC14.3214%.

**The1.6-TB/s goal remains unfinished.** Required active duration31.457280 us; another0.346476 us is needed. This improvement does not redefine the target.

## Resources and timing

Global allocation uses82 VGPR and56 SGPR in the fixed workload, zero scratch and zero SGPR/VGPR spills. Four waves per SIMD fit the modeled512-register VGPR file even after rounding. LDS remains66 KiB, already restricting residency to one TG per128-KiB CU.

Generated ISA has no `s_set_vgpr_size` or WDRA `ds_write_addtid`/`ds_read_addtid` entry save/restore. This uses the supported no-WDRA mode; it does not remove the WDRA hardware-bug workaround while resizing remains enabled. Normal ABI initialization stays intact. CU0 first B issue4291 ->3688 cycles after global first-wave start.

CU0 merged gap703 ->141; individual mean1368 ->1269.5, but max1420 ->1640. Thus not every wave improves. CU28 individual mean2364 ->1847.5 and max2536 ->2336. CU28 last MMAC end57116 ->56233, last store57344 ->56473; CU0 last MMAC56008 ->55745, last store56236 ->55993. Global end57782 ->57201 cycles. Raw absolute events and maxima accompany the means in JSON.

## Validation

- Two independent fixed-workload runs:101376 valid CPU-reference outputs correct, padding/untouched regions correct,512 valid FP8 MMAC instructions per CU0 consumer.
- CU0 all four P0 and eight consumer waves: B slot/phase/EMPTY order verified. All four waves of each producer: A128/B1024 loads, zero overlapping A/B issue windows, original issue positions.
- Independent-scale overwrite follows prior consumer read completion. Next-output A issue still precedes scale-reuse wait. No consumer ping-pong; consumer priority sequences equal the retained control.
- Perf embedded Wavefront matches raw TT for CU0 and CU28 (SE2/local CU4).
- Existing numerical API suite:12/12 passed, including FP8 E4/E5, INT8, mixed formats, FP16/BF16 output, wide flags, grouped-M and warmup, K1024. This is separate from performance validation. XML is adjacent.
- Thirteen compile-only configurations, including no-counter:81–82 VGPR,48–56 SGPR, zero scratch/spills and no WDRA entry instructions. These are not numerical or frequency measurements.
- Production-source ELF .text equals the measured repeat byte-for-byte; hash proof is adjacent.

## Artifacts and provenance

[Best perf, CU0 and SE2/CU4 Wavefront](/home/zhenx/temp/moe-gemm1-bm32-M128-20261116/bm32-static-vgpr-repeat-output96-gc1800/M128_N512_K2048_BM32_BN256_BK256_GRID48_OUTPUT96_GC1800_CU0.perf).

All exact case names, source hashes and metrics are in comparisons.json; snapshots, input parameters, ISA/HSACO, stats and IT/TT remain under `/home/zhenx/temp/moe-gemm1-bm32-M128-20261116/`.

No LLVM/Triton changes. Frozen clang SHA256 `b547297d968762bbcb017e1dd0a802e6b8b5e1afcef9b5072d8a5d0626b765c2`; libtriton SHA256 `49ae398b737f663ebe5dd369b66edc453f6de5a1a1a22156f4faf48e4941efe4`, both rechecked. Branch/commit/local-diff provenance: [frozen record](bm32-scale-bank-20261117/versions-1800.json), [additional record](bm32-scale-bank-20261117/versions.json). Compile-only context's default2.2-GHz message is not a benchmark.

Initial bootstrap attempts failed opening trace output before kernel execution; fresh-directory retries succeeded, with failed attempts excluded. No new PMD execution defect is claimed. Async bootstrap and scalar-load trials pass numerically but are slower; setup/rendezvous did not improve first B. Their code is not retained. Warmup, alternate store order and P0 priority2/3 also do not beat the static base and are not applied. Max-ILP uses84 VGPR but is slower (36.247308 us); disabling register-pressure scheduling produces byte-identical ELF .text and identical performance (proof JSON adjacent). These options are not applied.
