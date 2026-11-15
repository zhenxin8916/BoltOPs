# BN512 channel GEMM1 production port

Single kernel file: `gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128.py`.
Source algorithm: commit 5f9c36d, four original files listed with hashes in provenance.json.
P0 publishes both B halves through a three-slot FIFO; P1 publishes A through two
slots. Consumers retain each K256 A pair across two N128 panels. Native fragment
stores preserve wave ownership. A2*16KiB + B3*32KiB = 128KiB LDS.

Production additions: CK packed routes; INT8 exact INT32 accumulation; BF16/FP16
output; non-unit channel-scale strides; independent 64-bit A/B/C constants;
caller CU budget, device work count and complete N traversal. The CPU query
selects BN512 for complete N512/K512 cycles; other shapes retain existing
BN128/256 or ordinary Triton selection. No compiler source change is needed.
Warmup defaults to false. No spill was reported by the compiler.

## Normalized three-clock measurements

M4096, E256, topk8, actual GEMM1 N1024/K2048, BM128/BN512/BK128.
48 CUs, one persistent TG each, two outputs per TG: 96 output tiles and
128*1024 padded elements per TG, matching BN256*4 and BN128*8 CSV work.
TOTAL_TILES=96 bounds this diagnostic; production traverses all output tiles.
Each type reuses one frozen seed3101 input/routing/reference at all clocks.

| Type | GC (GHz) | Full dispatch (us) | CU0 lifetime MMAC (%) | Hot MMAC (%) |
|---|---:|---:|---:|---:|
| FP8 | 2.2 | 59.522192 | 54.9531 | 66.4504 |
| FP8 | 1.8 | 61.429104 | 66.0239 | 80.3609 |
| FP8 | 1.5 | 65.405352 | 74.2626 | 88.9148 |
| INT8 | 2.2 | 59.708284 | 54.4043 | 66.3630 |
| INT8 | 1.8 | 62.410444 | 64.2006 | 79.1493 |
| INT8 | 1.5 | 65.386008 | 73.7844 | 89.6753 |

CU0 efficiency averages its four SIMD active MMAC interval unions over its own
first WAVE_START through last s_endpgm plus one cycle, including startup/tail.
It does not divide by the full 48-CU dispatch window. Full dispatch time comes
from the HYGPERF XCD time section. All 48 CUs complete one TG. Eight CU0 consumer
waves execute 4096 active FP8/INT8 MMACs each. BF16 accumulator clears and EXEC=0
warmup instructions are excluded. Hot efficiency averages each output/SIMD's
first through last active quantized MMAC span. Every wave's first abarrier wait
is nonzero; consecutive waits use different IDs; all IDs remain within 0..15.

All six runs match the CPU reference (rtol .02, atol .002): 4,314,112 valid
and 1,977,344 padded elements checked; unselected output rows remain NaN.
Per-type input/output hashes and HSACO are identical across clocks. Address
comparisons, compiler register quotas, exact toolchain and local LLVM patch
are archived. Default clock is 1.8GHz; all clock overrides are process-local.

Archived 5f9c36d reported warmup=true: 59.425280us / hot MMAC84.90% on its older
toolchain. Current warmup=true rerun at1.8GHz is 59.322976us /
hot MMAC84.2871%. It is retained as a separate
comparison, not substituted for the default warmup=false CSV measurements.

## Correctness and limits

Six independent ordinary-Triton/CPU comparisons cover E4M3/E5M2/mixed FP8,
INT8, GROUP_SIZE_M=2, BF16 normal
addresses, FP16 forced64-bit A/B/C, scale strides, padded/invalid experts and
six persistent outputs across M and N. Each compares393216 elements exactly;
zero differing elements. Combined CPU/framework regression:216 passed, six
opt-in model cases skipped there and separately run successfully. See
equivalence.json, validation.json and the opt-in pytest. Forced
64-bit branches were tested on small allocations; no physical >4GiB input
was allocated. This BN512 algorithm requires K>=512, K%512=0, N%512=0.

Narrow A preserves native buffer-to-LDS copies. Wide A uses two K64 packed
int32 global-load/LDS-store chunks per K128 slot, with a plain-row shared
layout. This limits producer temporaries while preserving the two A slots.
B uses MLS in both paths. Query rejection preserves ordinary Triton fallback.

## Reproduction

Run in the paired container environment specified by launch.py:

```bash
python run.py --freeze  # once; retain the frozen tensors for all comparisons
python launch.py fp8
python launch.py int8
python launch.py fp8 --warm
mkdir -p check
RUN_GLUON_GEMM1_BN512_EQUIVALENCE=1 python -m pytest /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm1_bn512_equivalence.py --junitxml=check/equivalence.xml
python -m pytest /zhenxin/BoltOPs/test/fused_moe/test_gluon_fused_experts.py /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm1_bn512_equivalence.py /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm2_dispatch.py /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm2_bn512_dispatch.py -q > check/combined-cpu-tests.log
python publish.py
```

Opt-in numerical comparison:
`RUN_GLUON_GEMM1_BN512_EQUIVALENCE=1 python -m pytest test/fused_moe/test_channel_gemm1_bn512_equivalence.py`.
Only compact evidence is tracked; raw CU0 traces and frozen tensor payloads
remain at raw_root in provenance.json. CSVs live beside the kernel: top_fp8.csv
and top_int8.csv with the full tuned_experiment_fmoe_gluon_w8a8channel_ prefix.
