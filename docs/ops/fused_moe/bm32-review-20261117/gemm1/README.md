# BM32 GEMM1 migration review

This review uses the frozen pre-sync clang binary (SHA256
`0ec2b806f4f940fa67310006c3c2ef3ca602a0f6823538f988b0162405ea2add`,
build revision `fd18eddb`) and `-hcu-control-only-ebarrier-sync`.
It does not change or overwrite the prior twelve-case equivalence evidence or
three-frequency performance archive.

## Capacity and inactive TGs

Added `test/fused_moe/test_channel_gemm1_bm32_capacity.py`. **Both cases pass**
against a direct ordinary Triton call and an independent CPU reference.
M128/E2/N256/K1024, CK rank0/rank1, fixed seed3101 and BM32 are used.
Unlike the original small tests, backing storage covers the entire declared
capacity: six logical tiles, six expert entries, and 192 sorted/output rows.
Grid8 includes TGs beyond the capacity, as well as TGs within the capacity but
beyond the GPU work count.

| GPU active rows | Active TGs | Inactive TGs | Inactive output elements preserved |
|---:|---:|---:|---:|
| 64 | 2 | 6 | 32,768 |
| 0 | 0 | 8 | 49,152 |

Every inactive output element remains sentinel7. Valid active output elements
are bit-identical between Gluon, ordinary Triton and CPU. The eight active padded
rows deliberately expose a storage-contract difference: Gluon writes zero;
ordinary Triton masks padded-token stores and preserves their initial sentinel.
The test asserts both behaviors separately. It does not classify this expected
padding difference as a numerical failure or hide writes into inactive rows.
No production kernel or dispatcher is mocked.

The initial test compared padded sentinels against a common zero expectation and
failed only on ordinary Triton's 2,048 masked padded elements. The corrected test
compares valid rows and independently asserts both padding contracts. The final
run passed in 10.47 seconds; `capacity-validation.json` records exact coverage
and routing hashes. Raw model logs are retained in
`/home/zhenx/temp/bm32-gemm1-review-20261117/capacity-final/`.

## Source consistency

- Panel1 still checks the A READY barrier, but uses retained A registers; it
  performs no second A LDS read. Updated the comments to state both facts.
- P0's unused route expressions disappear during lowering. In the retained
  optimized TTGIR, P0's sorted-route pointer appears only in its signature;
  runtime P0 needs expert IDs, while P1 uses CK routes. Corrected the role comment.
- No algorithm, priority, barrier sequence or launch resource was changed.
  `source-review.json` proves the executable AST equals commit `598f0ec` after
  stripping docstrings; only comments/docstrings changed during this review.

## Address and scope review

Within the existing ordinary Triton logical-index contract, A's wide path uses
64-bit route/K address arithmetic, B folds the expert and panel base into a
64-bit pointer, C widens the sorted-row product when selected, and scale offsets
widen before applying strides. These cover allocations/base offsets and scale
strides exceeding 4 GiB without rejecting the shape solely by byte span. The
prior numerical suite independently forces A/B/C64 and exercises strided scales.
It does not claim to allocate or numerically test physical tensors exceeding
4 GiB on the model.

As in the existing non-Gluon contract, no new guarantee is introduced for logical
N/K dimensions themselves at or beyond 2**31. Some panel-column/K index products
remain 32-bit before base folding. This theoretical dimension-domain question
is recorded separately from the supported >4-GiB base/stride mechanism; no
arbitrary new shape rejection or untested production rewrite was added.

The earlier twelve-case suite remains the coverage for FP8 E4M3/E5M2/mixed,
INT8 INT32 accumulation, BF16/FP16, grouped-M, independent address flags,
strided scales, actual framework dispatch and optional instruction warmup.
This review adds the previously missing inactive/empty-work behavior.

## Reproduce

Use the frozen compiler environment recorded in `toolchain.json`:

```bash
RUN_GLUON_GEMM1_BM32_EQUIVALENCE=1 python -m pytest -q /path/to/BoltOPs/test/fused_moe/test_channel_gemm1_bm32_capacity.py
```

Each case invokes two kernels: the Gluon kernel on grid8 and the ordinary Triton
reference on its capacity grid. Keep this model run separate from the LLVM sync
and subsequent compiler regression measurements.
