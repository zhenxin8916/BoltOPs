# BM32 GEMM1: isolated LLVM sync comparison

Compared frozen clang build `fd18eddb` with new build `c49fb462`
(origin `ff89` plus only the retained `fd18` patch). Both use
`-hcu-control-only-ebarrier-sync`, the same clean Triton and 1.8 GHz model.
Compiler executable hashes are in `compilers.json`.

The retained M128/E256/topk8, N512/K2048 fixture uses seed3101, grid48 and 96
logical outputs (two per TG). Input/routing hashes match between runs. Both
complete with zero errors over 101,376 valid elements; padded outputs are zero
and unselected rows remain untouched. The full selected output prefixes match
byte-for-byte (SHA256 `7321d54df6866da9d2154d10ef91bc4f5001090b8882205df96310bb1fedf408`).

| Compiler | Full dispatch | CU0 whole MMAC | K-loop MMAC | LDS |
|---|---:|---:|---:|---:|
| old fd18 | 36.580352 us | 14.601277% | 16.839703% | 64 KiB |
| new c49fb | 36.580352 us | 14.601277% | 16.839703% | 64 KiB |

Executable `.text` is identical: 7,828 bytes, SHA256
`0b9fe3907a747ac833538a26cd115143a52fca100ceb0fb8c89cd1af53fe08b3`.
The ISA serialization hashes in raw `result.json` differ because compiler/source
annotations differ; those fields are not executable `.text` hashes.

Static DS counts are unchanged: 32 matrix-trans reads, 16 B64 reads, four
addtid-B32 reads, four read2-B32 reads, two read2-B64 reads, and one each of
write-addtid-B32, write2-B32 and write-B32. Both have zero SGPR/VGPR spill and
zero scratch/private bytes. Thus the separately observed DS microprobe change
has no effect on this actual kernel's generated program or measured performance.
This conclusion is specific to BM32 GEMM1 and does not stand in for other kernels.

CU0 efficiency uses its own first IT WAVE_START through last IT s_endpgm plus
one GC period. Active 8-cycle MMAC interval unions include FP8 math and BF16
zero-initialization and exclude EXEC=0. The all-CU span is retained separately.

Each side archives the HSACO, CU0 perf, compressed ISA/LLIR/TTGIR, launch/input,
numerical and measurement records. `comparison.json` contains exact endpoints,
static counts, hashes and timings. Raw IT remains outside the repository under
`/home/zhenx/temp/llvm-sync-fd18-20261117/gemm1-bm32/`.
No kernel, dispatcher, CSV or compiler source was changed for this comparison.
