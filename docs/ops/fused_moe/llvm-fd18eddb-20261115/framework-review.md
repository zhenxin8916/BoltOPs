# Gluon channel MoE integration review

The four committed kernels are dispatched by `triton/moe_op.py`, before ordinary Triton-only tuning checks. `sol_type=triton` retains this internal selection. An unsupported candidate returns False and continues through ordinary Triton. The untracked BM64/BN512 GEMM2 experiment is not part of this production dispatch.

| Kernel | Production selection | Geometry |
|---|---|---|
| GEMM1 BM128/BN128,256/BK128 | CK-sorted C, unsorted A; chooses BN256 for N divisible by 128, otherwise BN128 | K >= 512, divisible by 256; supported MLS N boundaries |
| GEMM1 BM128/BN512/BK128 | N and K divisible by 512 | K >= 512; retains A across N panels |
| GEMM2 BM128/BN256/BK128 | Sorted A, route-indexed C; selected when N not divisible by 512 | K=256; N divisible by 256 |
| GEMM2 BM128/BN512/BK64 | Sorted A, route-indexed C; selected when N divisible by 512 | K=256 |

Both GEMM2 candidates choose BN/BK internally; only BM128 is a sorting constraint. Other shapes continue to ordinary Triton. GEMM2 keeps row-persistent traversal, so ordinary tuning GROUP_SIZE_M is not a requirement of its internal layout. GEMM1 passes GROUP_SIZE_M through to its tile traversal.

## Features checked against ordinary Triton

- FP8 E4M3FN/E5M2 (including mixed A/B formats), signed INT8, FP32 channel scales; BF16/FP16 output. INT8 accumulates in INT32 before FP32 dequantization.
- CK-packed token/route decoding, CK device-side padded work count, negative expert and padded-token behavior. GEMM1 writes sorted rows; activation/quantization preserves that layout; GEMM2 scatters back to token/top-k routes before the existing reduction.
- GEMM2 can multiply sorted route weights or omit them when weights were already folded by activation quantization. GEMM1 input router weighting and bias use ordinary Triton.
- Non-unit and broadcast channel-scale strides. Independent A/B/C wide-address flags. BN128/256 GEMM1 rebases expert B in 64 bits and independently handles large scale strides; BN512 and GEMM2 include scale/metadata spans when selecting wide addressing.
- Grid uses the device CU budget rather than a hardcoded 48-CU count. Metadata tests include 1/48/96/192 CU budgets. Instruction warmup remains disabled.
- Bias, zero points, non-channel quantization, unsupported output dtypes (including FP32), non-CK sorting, incompatible layouts/geometry and other architectures keep ordinary Triton. These are fallback coverage, not additional native Gluon features. Activations, expert-map setup, chunking and top-k reduction remain in the shared framework.

## Fixed in this review

The initial and final-chunk GEMM2 sorted-layout queries used `hidden_states.dtype`. Prequantized FP8/INT8 input with BF16/FP16 output could therefore miss Gluon selection; conversely BF16 input with FP32 output could request an unsupported Gluon bridge. Both queries now use the actual output dtype. No kernel code or benchmark geometry changed.

A CPU regression test executes the real fused-experts entry up to its layout query, then checks the actual candidate query. It covers FP8/INT8/BF16 inputs crossed with BF16/FP16/FP32 outputs. Existing dispatch tests cover framework selection, ordinary fallback, sorted bridging, independent wide-address flags, CU budgets and mixed FP8 formats. GPU numerical equivalence and performance evidence from the preceding LLVM verification remains applicable because generated kernel source is unchanged.

All four experiment CSVs use Triton `5b24821ee57eeb7560074cc7979ce7b59604079f` and LLVM `fd18eddbaf4a662ae5ce86c516d28e5b16952513`. Their LLVM descriptions no longer claim the old local load-sinking guard is present. All time/MMAC frequency columns are populated and linked to the archived run manifests.
