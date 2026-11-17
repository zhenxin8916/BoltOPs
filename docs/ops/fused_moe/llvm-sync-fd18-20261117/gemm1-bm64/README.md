# BM64 GEMM1: isolated LLVM sync comparison

Compared frozen clang build `fd18eddb` with new build `c49fb462`
(origin `ff89` plus only the retained `fd18` patch). Both use
`-hcu-control-only-ebarrier-sync`, the same clean Triton and 1.8 GHz model.

This comparison uses the existing `frozen_inputs.pt` fixture: M1024/E256/topk8,
actual N1024/K2048, BM64/BN512/BK256, grid48, 96 logical outputs (two per TG).
The launch options and input/routing hashes match exactly. Both complete with
zero mismatches over 1,580,032 valid elements, zero padding errors, zero selected
NaNs and no writes to unselected rows. The performance worker compares a fixed
CPU reference with rtol=.02/atol=.002 after BF16 rounding; it does not emit an
output-byte hash, so no direct old/new output-hash comparison is claimed.

| Compiler | Full dispatch | CU0 whole MMAC | K-loop MMAC | LDS |
|---|---:|---:|---:|---:|
| old fd18 | 69.403808 us | 28.445179% | 31.735910% | 128 KiB |
| new c49fb | 69.403808 us | 28.445179% | 31.735910% | 128 KiB |

Executable `.text` is identical: 12,960 bytes, SHA256
`d0a9244a7a36fef12caaaccb5002c77c13ae20d6d3b64b8a40998de7c7d5838b`.
Both have zero SGPR/VGPR spill and zero scratch/private bytes.
The actual kernel retains the same DS pairing: 64 matrix-trans reads,
16 read2st64-B64 reads, four read2-B64 reads, ten read2-B32 reads, four
addtid-B32 reads and three B32 reads. DS writes are likewise unchanged.
Thus the separate DS microprobe regression does not change this generated
program or its measured performance.

The older CSV's 69.284272 us differs slightly from this matched rerun;
the compiler comparison is the matched pair above, whose time difference is zero.
No CSV was changed by this check.

CU0 efficiency uses first IT WAVE_START through last IT s_endpgm plus one GC
period, with four-SIMD unions of active 8-cycle MMAC intervals. Whole efficiency
includes active BF16 zero initialization and FP8 math and excludes EXEC=0.
The all-CU wave span remains a separate field and is not the MMAC denominator.

Each side archives HSACO, CU0 perf, compressed ISA/LLIR/TTGIR, launch,
numerical and metric records. `comparison.json` records compiler hashes,
fixture hash, endpoints, static counts and exact timings. Raw model logs and
IT remain under `/home/zhenx/temp/llvm-sync-fd18-20261117/gemm1-bm64/`.
The archived worker/controller reproduce the original frozen-input invocation.
No kernel, dispatcher or compiler source was changed for this comparison.
