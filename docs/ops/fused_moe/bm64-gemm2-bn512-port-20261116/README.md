# BM64 GEMM2 BN512 integration verification

Source: `98a1e1e`, one kernel file
`gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn512xbk64.py`.
The current repository already contained the implementation. All JIT/layout
functions match the source commit after removing docstrings; no algorithm was
rewritten. The header now correctly says next A is loaded into LDS, while
next-M scales/weights/routes/expert are held in registers. The support query
also rejects `GROUP_SIZE_M != 1`, matching the existing entry assertion.

## Algorithm and compatibility

```text
P0: A(M0) + metadata(M0) -> wait A EMPTY -> A(M1) into LDS
    -> hold metadata(M1) in registers -> drain first output phase
    -> publish metadata(M1) -> drain remaining outputs -> scatter C
P1: B-left full K256 / SB -> C0; B-right full K256 / SB -> C1
C0/C1: retain A -> read B -> return A/B EMPTY after LDS completion
       -> four BK64 MMAC chunks -> scale -> output LDS buffer -> P0
```

A uses one 16-KiB slot. Each B half uses one 32-KiB full-K slot.
Two output buffers use 32 KiB; scales/metadata use 3 KiB. Total 115 KiB.
The 12 live barrier IDs are `0,1,3,5,6,7,8,9,11,12,13,15`.
A EMPTY gets four C0 plus four C1 arrivals; B/output EMPTY gets four.
The actual CU0 traces verify all IDs within 0..15, first waited ID nonzero,
no adjacent repeated wait ID, and the two A/four B generations per consumer.
Both consumer roles have no global/buffer loads or permlane instructions.

CK-sorted A and row scales, packed CK routes, optional sorted routed weights,
BF16/FP16 output and separate constexpr A/B/C wide-address paths are retained.
Full N traversal is the production default. `N_TILES_LIMIT=1` only bounds this
benchmark to two M64xN512 output tiles per persistent TG. Instruction warmup
is absent. There is no inline assembly in this kernel; exposed HCU operations
express the required MLS, LDS waits, priorities and barrier operations.

## Correctness

- Four E4M3FN/INT8 × BF16/FP16 cases pass against both ordinary non-Gluon
  Triton and an independent CPU result, with zero Gluon/Triton bit differences.
- Two additional E5M2 and mixed E4M3FN/E5M2 cases pass the same comparisons.
- Wide-address variants exercise all three constexpr paths and non-unit scale
  strides. These are forced-path checks, not allocations larger than 4 GiB.
- Three CPU support-query checks pass for E4M3FN/E5M2/INT8; CK sorting and
  the supported row grouping are required.
- All four bounded benchmark runs check 1,571,840 selected output elements,
  zero mismatches and an untouched output tail. FP8/INT8 final assembly has
  zero SGPR/VGPR spills, zero private segment and no scratch instructions.

Evidence: [equivalence.xml](equivalence.xml), [e5.xml](e5.xml),
[support-query.txt](support-query.txt), [source-audit.json](source-audit.json).
Each clock directory contains final assembly/LLVM IR, compilation metadata,
result, launch input hashes, barrier audits and metrics.

## PMD performance

Fixed inputs: **M1024, E256, top-k8, seed3101**, actual GEMM2 N2048/K256,
48 CUs/TGs, two independent M64xN512 outputs per TG (96 padded output tiles).
CPU inputs are frozen once and reused across frequencies. Effective checked
output is smaller than padded MMAC work because CK expert padding is present.
These inputs differ from historical M4096 measurements and the E16 BM64 BN256
benchmark; elapsed times are not a matched cross-geometry comparison.

| Type | GC | Time (us) | CU0 whole-kernel MMAC |
|---|---:|---:|---:|
| FP8 | 2.2 GHz | 17.228120 | 13.839708% |
| FP8 | 1.8 GHz | 18.651020 | 16.472291% |
| FP8 | 1.5 GHz | 20.230110 | 18.515505% |
| INT8 | 1.8 GHz | 19.046892 | 15.793329% |

The machine-readable [results.csv](results.csv) / [results.json](results.json)
are authoritative. Time uses the full HYGPERF dispatch window. MMAC uses CU0's
own first-wave start through its last `s_endpgm` plus one cycle, including
startup and tail; numerator is CU0 MMOP count × 8 / 4 SIMD. The raw CU0
`totalCycles` counter ends at `s_endpgm`, one cycle shorter, and is retained.
Every run has 16 active CU0 waves (four per SIMD), 2048 MMAC instructions and
zero EXEC=0 warmup MMAC instructions. This is PMD validation, not hardware
numerical certification.

Toolchain: clean Triton `5b24821ee57eeb7560074cc7979ce7b59604079f`, LLVM
`fd18eddbaf4a662ae5ce86c516d28e5b16952513`; control-only ebarrier-sync LLVM
option. PMD `b61a79c5`/CoreArch1769, known-issues index checked before testing.

Reproduce from Docker `zx-perf-node60` with `/home/zhenx` mounted as `/zhenxin`:
`python3 docs/ops/fused_moe/bm64-gemm2-bn512-port-20261116/launch.py 1800 fp8`.
The frozen inputs and full CU0 SQTT/IT/perf files remain under
`/home/zhenx/temp/bm64-gemm2-bn512-port-20261116/`.
