# BM32/BN512 GEMM2 integration

The kernel is `boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm32xbn512xbk64.py`.
Its tuned algorithm comes from commit `d797ac0` in the BM32 worktree. Later worktree
commits only add evidence. The original source SHA256 is
`71da9cc7824dca219a9306a2d7f47884aef5832853dd939c60c6c798c65ea978`.

## Algorithm and supported inputs

```
P0: A + row metadata -------------> C0 / C1
    next-M A prefetch; hold metadata; drain first output; publish metadata
    C0 / C1 output mailboxes ------> global route-major C
P1: prefetch SB-left / SB-right; full-K B-left -> C0; B-right -> C1
C0 / C1: retain full A; read one K256/N64 B phase; release completed LDS reads
         four BK64 MMACs -> scale once -> M32/N64 output mailbox
```

Each BN512 tile has four N64 phases per consumer. A advances per M; B/output
advance per phase. Twelve abarriers stay within IDs 0..15; first waits are on
nonzero IDs, and P0 observes A READY1 between successive A EMPTY3 waits.
The default path still reserves 50,688 B LDS and role VGPRs 35/12/74/74.
The pipeline audit checks matching MLS seq/arrive tracks and completed LDS
reads/writes before EMPTY/output READY.

The port keeps CK-sorted A/routes, FP32 channel scales, optional routed weights,
and route-major BF16/FP16 scatter output before top-k reduction. FP8 E4M3FN,
E5M2 and mixed A/B formats are supported, as is signed INT8 with exact INT32
accumulation through K. K is fixed at 256; N is a multiple of 512. M, expert
metadata, constexpr top-k, grid and device padded-row limit remain caller-supplied
metadata/launch contracts.
Independent constexpr A/B/C wide-address flags and non-unit SA/SB strides are
preserved. The metadata support query accepts general valid sorted row counts,
not only the M128 tuning case. Unsupported queries fall back to ordinary Triton.

`INSN_WARMUP` defaults to false. True remains explicitly rejected, matching
`d797ac0`: copying another geometry's out-of-line MMAC warmup produced numerical
errors, so that attempted path was removed rather than silently enabled.
`N_TILES_LIMIT=0` covers full N; positive limits are for diagnostic equal-work tests.

## Correctness and resources

`test/fused_moe/test_channel_gemm2_bm32_bn512_equivalence.py` compares identical
fixed-seed inputs against ordinary Triton (Gluon fallback disabled explicitly)
and a dense CPU reference. Seven GPU cases passed: E4M3FN/E5M2/mixed FP8, INT8,
BF16/FP16 output, routed weighting on/off, forced wide A/B/C addressing and
non-unit scale strides. Each checks 395,264 outputs across multiple M blocks
and full N1024, top-k2, with zero Gluon/Triton differences. One case uses the real
framework dispatch bridge; only its discovered CU count is set to one.
The seventh case allocates 448 sorted rows, keeps TOTAL_TILES at that capacity,
but supplies a GPU WorkCount of 416 rows. Both kernels process only active M
blocks; all inactive route output rows retain the initial -11 sentinel.

All seven candidate HSACOs have zero SGPR spills, zero VGPR spills, zero private
segment bytes and no scratch instructions. Wide-address variants use up to
48/15/74/74 actual role VGPRs. Metadata query tests are CPU-only; GPU tests are
opt-in:

```bash
RUN_GLUON_GEMM2_BM32_BN512_EQUIVALENCE=1 pytest -q \
  test/fused_moe/test_channel_gemm2_bm32_bn512_equivalence.py
```

Forced wide flags validate the generated 64-bit paths on compact allocations;
these tests do not allocate tensors larger than 4 GiB.

## Fixed-work performance

M128/E256/N2048/K256/top-k8, fixed seed3101, 48 persistent TGs, each computing
two independent M outputs at the first BN512 N tile (96 actual tiles).
Inputs/routing reuse the original frozen files byte-for-byte. Each frequency
passed all 197,632 valid selected outputs; unselected output remained NaN.

| Nominal GC | Dispatch time (us) | CU0 total MMAC | CU0 compute-window MMAC |
|---|---:|---:|---:|
| 2.2 GHz | 14.275170 | 8.99034% | 13.80705% |
| 1.8 GHz | 15.466808 | 11.19432% | 19.07956% |
| 1.5 GHz | 16.699679 | 12.51375% | 21.30005% |

The 1.8 GHz time exactly reproduces the original best 15.466808 us. Final-source
`.text` SHA256 `85eeec2be054a4ab0c6e41c66dd66cadfdfcebf3a0cbc317d5224e5ad490b043`
is identical to all three measured artifacts and the original tuned compile.

Total MMAC uses CU0 IT min `WAVE_START` through max `s_endpgm` plus one GC
cycle, including startup and tail, with four SIMD spans as denominator. This
estimates terminal issue-to-end latency; it does not claim a decoded TT WAVE_END
endpoint or use the whole-dispatch window. Exact ticks and the clock period are
in `performance.json`. The previous original-source analyzer used global shader
wave endpoints; those rejected percentages are retained separately for audit. Compute-window MMAC uses each output's first-to-last active MMAC on each
SIMD; the gap between the two outputs is excluded. Busy intervals use eight GC
cycles per MMAC and are unioned per SIMD. Percentages are useful-work efficiency
for this small fixed-output benchmark, not a claim that the whole dispatch is
compute-bound. Frequencies are nominal; PMD uses integer clock periods.

The run used Triton `5b24821ee57eeb7560074cc7979ce7b59604079f`, clean release/3.6.x.
LLVM tree HEAD was `89bee503d8e9d0bdcfbb4d97e204d5e346f02550`; the actual executable
reports build revision `fd18eddb...` and SHA256
`0ec2b806f4f940fa67310006c3c2ef3ca602a0f6823538f988b0162405ea2add`, unchanged before
and after validation. Tree revision alone is not executable build provenance.
The existing control-only ebarrier-sync clang wrapper and PMD binary hashes are
recorded separately in `provenance.json`; no compiler/model rebuild or patch
was introduced by this port.

The three CU0 `.perf` files, measured HSACOs, compressed AMDGCN/LLVM IR/TTGIR,
launch/results and analyzers are archived here; `artifact-sha256.json` binds
them. Full raw logs, input tensors and IT/SQTT text remain under
`/home/zhenx/temp/bm32-gemm2-migration-20261117/`. Perf files embed the captured
CU0 wavefront trace, checked against the raw trace payload. No numerical failure
was attributed to a PMD defect.

The archived launcher defaults to this repository's production module, rather
than an absolute temporary source import. Reproduce from the matching source
revision inside the existing build container with:

```bash
bash docs/ops/fused_moe/gemm2-bm32-bn512-port-20261117/reproduce.sh \
  /zhenxin/temp/bm32-gemm2-reproduce
```

`TRITON_ROOT`, `PYTHON_BIN` and `PMD_PATH` may be overridden for a matching
installation. `clang-control-sync` records the exact compiler path/flag used.
Input generation runs once; subsequent frequencies reuse it. Regenerated
input hashes must match the archived frozen-input manifest. The archived
`analyze.py` uses the corrected CU0 IT endpoint, not the original global shader
endpoint. No large raw trace or input tensor is required to review the archived
perf; fresh analysis after reproduction uses that new run's raw IT.
