# BM32 channel GEMM1 integration

The retained `639b287` algorithm is unchanged: P0 streams four N64 B panels per
BK256; P1 fills two A slots; C0/C1 each retain A registers across their two B
panels. B uses native MLS, A uses buffer-to-LDS or typed load/store for 64-bit
offsets. Consumer ping-pong remains disabled. Instruction warmup defaults off.

## Supported interface

- gfx946; M128; BM32/BN256/BK256; N divisible by 256; K >= 1024 and divisible by 512.
- Contiguous A/B: E4M3FN, E5M2 (including mixed pairs), or signed INT8 pairs.
  FP8 accumulates in FP32; INT8 accumulates in INT32 and converts once for scaling.
- FP32 channel scales, including strided/broadcast scale views; BF16 or FP16 sorted output.
- CK-packed IDs: low 24 bits select the input token; rank bits do not select A.
  Padded rows produce zero. Unsorted output, sorted A, bias and routed-weight
  multiplication use the ordinary Triton fallback.
- Caller-selected positive CU/grid budget; optional device work count narrows
  `TOTAL_TILES`. Grouped-M traversal uses the ordinary Triton mapping.
- A/C independently select 64-bit offsets. B MLS always folds expert/panel/K into
  its 64-bit base; scale-index products are widened before applying their strides.

Fixed a C64 compile failure: the epilogue passes constant `block_m=0`, so its
conversion must use `gl.cast`, which accepts both Python constants and tensors.
The default 32-bit executable remains byte-identical to the retained version.
The kernel also asserts its supported 16-bit output types; the support query
rejects non-positive GROUP_SIZE_M, matching the compile-time contract.

## Numerical validation

`test/fused_moe/test_channel_gemm1_bm32_equivalence.py` covers **12 cases**.
Every case compares Gluon against both a direct ordinary Triton invocation and an
independent CPU reference, with zero tolerance. The frozen seed/routing includes
rank1 CK IDs, expert changes and padded rows. Coverage includes both mixed FP8
directions, INT8, BF16/FP16, independent A/B/C64 flags, strided scales,
GROUP_SIZE_M=4, grid2, capacity larger than the GPU work count, and instruction
warmup enabled. One case also executes the framework dispatcher; only the detected
CU budget is replaced with 1. No kernel or dispatcher result is mocked.

The first 11 cases passed together; the explicit warmup case passed separately.
See `numerical-validation.json`. All 12 cached base specializations have zero
SGPR/VGPR spill and zero private/scratch bytes (`no-spill.json`); the warmup test
also checks zero spill. These small tests force 64-bit code paths, not physical
allocations larger than 4 GiB.

## Fixed-work FP8 performance

M128 / E256 / topk8 / actual N512 / K2048; grid48; **two output tiles per TG**,
96 total. Seed3101 and identical input/routing hashes are used at all clocks.
This is a bounded diagnostic dispatch, not a complete MoE invocation. Each run
checks 101,376 valid output elements, zero-valued padding, and untouched rows.

| Requested GC | Dispatch time | CU0 whole MMAC | K-loop MMAC |
|---|---:|---:|---:|
| 2.2 GHz | 35.505924 us | 12.030075% | 13.879261% |
| 1.8 GHz | 36.580352 us | 14.601277% | 16.839703% |
| 1.5 GHz | 37.830240 us | 17.264743% | 20.351880% |

Full dispatch duration comes from HYGPERF section15. CU0 efficiency is the union
of active 8-cycle MMAC intervals per SIMD, summed over four SIMDs, divided by
four times CU0's own wave lifetime. The endpoint is instruction-derived:
**first CU0 WAVE_START to last CU0 s_endpgm plus one GC period**, matching the
existing integration tables. Native TT wave-end decoding is not claimed.
Whole efficiency includes the 32 active BF16 zero-initialization MMACs and 4096
FP8 MMACs; EXEC=0 visits are excluded. K-loop efficiency uses active FP8 MMACs
only, measured separately for each output/SIMD. The legacy all-CU wave span is
retained separately as `all_cu_wave_span_time_us`, never used as the MMAC denominator.

`gc*-record.json` supplies structured CSV inputs; `gc*/` retains CU0 perf,
HSACO, compressed ISA/LLIR/TTGIR and the numerical/launch/measurement records.
`final-text-proof.json` proves all three executable `.text` sections equal a
fresh final-source compile: SHA256
`0b9fe3907a747ac833538a26cd115143a52fca100ceb0fb8c89cd1af53fe08b3`.
The measured program uses 64 KiB LDS and no register spill or scratch.
Compiler executable hash is pinned in `toolchain.json`; source checkout and
compiler binary build identity are recorded separately when publishing the CSV.

## Reproduce

Use the recorded Triton environment and frozen compiler/control wrapper. From
this directory, with the model Python environment active:

```bash
python run-performance.py --output /path/to/new-results --clang /path/to/frozen-clang-control
python analyze.py /path/to/new-results/gc2200-final /path/to/new-results/gc1800-final /path/to/new-results/gc1500-final
RUN_GLUON_GEMM1_BM32_EQUIVALENCE=1 python -m pytest -q /path/to/BoltOPs/test/fused_moe/test_channel_gemm1_bm32_equivalence.py
```

`probe.py` regenerates inputs with the same fixed seed and records their hashes;
no large tensor fixture is checked in. Set `COMPILE_ONLY=1` to compile the same
case without launching the GEMM. Keep each model run in its own directory.
Raw failed harness runs remain outside the repository: an asynchronous trace
writer raced the old worker's `chdir`; the final worker synchronizes before
saving compiler artifacts. No production instruction workaround was introduced.
