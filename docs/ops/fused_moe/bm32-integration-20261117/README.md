# BM32 Gluon MoE integration

The sorting configuration owns BM. With `BLOCK_SIZE_M=32`, the existing Gluon-first dispatcher now queries these candidates before falling back to ordinary Triton. BN/BK are selected by the kernel rather than overriding the sorting block. Caller CU counts determine the persistent grid.

| Stage | Single-file kernel | Retained algorithm | Support range |
|---|---|---|---|
| GEMM1 | `gemm_top_w8a8_channel_wasp_persistent_bm32xbn256xbk256.py` | `639b287` | M128, N divisible by 256, K>=1024 and divisible by 512 |
| GEMM2 | `gemm_bottom_w8a8_channel_wasp_persistent_bm32xbn512xbk64.py` | `d797ac0` | CK-sorted rows, K256, N divisible by 512 |

Both support signed INT8 and E4M3FN/E5M2/mixed FP8, FP32 channel scales, BF16/FP16 output, independent A/B/C wide-address constants, non-unit scale strides, CK padding and a GPU padded-row work counter. GEMM1 writes sorted C. GEMM2 reads sorted activations, optionally multiplies routed weights and scatters route-major C before top-k reduction. Unsupported metadata preserves ordinary Triton. The sorted-activation bridge queries the same GEMM2 candidate as dispatch.

GEMM1 fixes a C64 compile failure for constant block_m via `gl.cast`; its default executable remains unchanged. GEMM2 preserves the tuned full-K A register retention, N64 B phases, paired channel-scale prefetch and next-expert lookahead. Neither migration changes the tuned synchronization algorithm.

Instruction warmup remains disabled by default. GEMM1's enabled path passed numerical validation. GEMM2 explicitly rejects enabled warmup, matching its source version; a failed attempted implementation was removed. This is a documented limitation rather than a silently ignored option.

## Validation

- 146 framework/query/CSV CPU regressions passed, including BM128/BM64, 24 BM32 dispatch/address/fallback checks and 3 GEMM2 support-query cases. The 19 opt-in GPU cases were executed separately and are skipped in the CPU-only regression run.
- GEMM1: 12 GPU numerical cases, exact ordinary-Triton and independent CPU comparisons. Covers both FP8 mixed directions, INT8, output types, independent wide flags, strided scales, CK rank/padding, grouped traversal, clipped work count, warmup and real framework dispatch.
- GEMM2: 7 GPU numerical cases, exact ordinary-Triton comparison and CPU-reference tolerance checks. Covers FP8 formats, INT8, output types, weighting on/off, full N, wide flags, strided scales and actual dispatch. An allocated 448-row case caps GPU work to 416 rows and preserves the inactive output sentinel.
- All audited code objects have zero SGPR/VGPR spills and zero private/scratch storage. GEMM1's three-frequency waits and ordered 16-ID initialization pass the recorded abarrier constraints. GEMM2's pipeline audit checks first nonzero waits, distinct consecutive wait IDs, the 0..15 range and LDS completion before ownership transfer.

These are PMD checks; no real-hardware run is claimed. Forced wide-address tests use compact allocations rather than physical allocations above 4 GiB. See the operator evidence for exact test commands and coverage limits.

## Performance and CSV

M128, E256, top-k8, 48 CUs, fixed inputs/routing, two logical output tiles per persistent TG. Warmup is off. GEMM1 uses actual N512/K2048; GEMM2 uses N2048/K256 and measures only the first BN512 N tile (`N_TILES_LIMIT=1`). These are bounded diagnostic dispatches, not whole-MoE or full-N GEMM2 timings.

| GC | GEMM1 time (us) | CU0 whole MMAC | GEMM2 time (us) | CU0 whole MMAC |
|---|---:|---:|---:|---:|
| 2.2 GHz | 35.505924 | 12.030075% | 14.275170 | 8.990342% |
| 1.8 GHz | 36.580352 | 14.601277% | 15.466808 | 11.194315% |
| 1.5 GHz | 37.830240 | 17.264743% | 16.699679 | 12.513748% |

GEMM2 exactly reproduces d797ac0's 1.8-GHz best time. Final default-source executable sections match the measured artifacts for both stages. Dispatch time excludes host setup. MMAC efficiency unions active eight-cycle instruction intervals per SIMD and divides their summed busy ticks by four times CU0's own lifetime, including startup and tail. The common endpoint is IT min WAVE_START through max s_endpgm plus one GC period; terminal latency is estimated, not claimed to be decoded TT WAVE_END. Global shader/dispatch spans are not the MMAC denominator.

`published-records.json` contains the exact CSV inputs. `publish.py` rebuilds the two FP8 CSV rows from archived evidence, retaining all existing rows. No INT8 performance CSV is generated; INT8 performance estimates may refer to FP8 but must be labeled estimates. The tables record Triton/LLVM source revisions and explain the frozen compiler binary identity separately, including its stale build-version string.

Detailed records, CU0 perf, HSACOs, compressed ISA/IR and reproduction scripts:
- [GEMM1](../gemm1-bm32-bn256-integration-20261117/README.md)
- [GEMM2](../gemm2-bm32-bn512-port-20261117/README.md)

CSV contract: BM32 uses M128 and two tiles/TG; BM64 retains M1024 and two tiles/TG; BM128 retains the existing equal-padded-output normalization. BM32 BN256 and BN512 have different output-element counts per TG and should not be compared as equal arithmetic work.
