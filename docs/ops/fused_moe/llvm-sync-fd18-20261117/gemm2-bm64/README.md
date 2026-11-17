# LLVM sync comparison: BM64 GEMM2, M1024

Two actual production kernels were compiled and rerun using independent frozen
old/new compiler and linker pairs, with the existing control-only ebarrier-sync
option. This is the standard M1024 workload, not the earlier M128 diagnostic.

Inputs are reused from the original benchmarks without regenerating top-k:

- BN256/BK256: `temp/bm64-bn256-integration-20261116/inputs-e256`.
- BN512/BK64: `temp/bm64-gemm2-bn512-port-20261116/inputs`.

Each run is FP8 E4M3FN/BF16, M1024/E256/N2048/K256/top-k8, seed3101, 1.8GHz,
48 persistent TGs with two independent M64 outputs per TG, clipping to the first
BN-sized N tile. All input hashes are checked before launch. Both BN settings
have96 output tiles; BN512 does twice the N work, so elapsed times across BN
are not matched-work performance comparisons.

| Kernel | Old time (us) | New time (us) | Checked values | CU0 whole MMAC | CU0 hot MMAC |
|---|---:|---:|---:|---:|---:|
| BM64/BN256/BK256 | 13.561396 | 13.561396 | 785,920 | 12.6443168% | 49.4447127% |
| BM64/BN512/BK64 | 18.651020 | 18.651020 | 1,571,840 | 16.4722915% | 31.8655671% |

Both compiler versions have zero tolerance failures against each fixed CPU
reference (rtol0.02/atol0.002), identical output hashes and untouched NaN output
regions. The times exactly reproduce the corresponding historical1.8GHz CSV
entries. Whole efficiency uses CU0 IT first WAVE_START through last s_endpgm
plus one GC cycle, including startup and tail, with four SIMD spans as the
denominator. Hot efficiency uses weighted first-to-last active MMAC compute
windows for the two outputs. Identical values apply to old and new compilers.

For each geometry, old/new native `.text` is byte-identical and all static
LDS/MLS opcode counts are unchanged:

- BN256/BK256 text SHA256:
  `52c3b17c0bc556ef7572680c67f3a4a58a585eb2a2691b30b60e841e3400ce5c`.
  Two `ds_read2_b64` instructions remain paired;12 `ds_read_b128` and48
  matrix-trans reads are unchanged. Actual role VGPRs17/4/148/149; LDS102,400B.
- BN512/BK64 text SHA256:
  `09acf1920299d90fa5a14898d801767660905ae28fa7ded4ce2fac9fa1ff5a51`.
  One `ds_read2st64_b32`, two `ds_read2_b32`,32 `ds_read_b128` and64
  matrix-trans reads are unchanged. Actual role VGPRs49/9/155/155; LDS117,760B.

All four measured HSACOs have zero SGPR/VGPR spills, zero private segment and
no scratch instructions. Full DS/MLS opcode inventories and allocation quotas
are in `comparison.json`. The separate nested/disjoint-OR probe issue does not
change these two production kernels; this does not claim that probe is fixed.

`provenance.json` records the frozen old fd18-built and new c49fb462-built
compiler/linker hashes and unchanged Triton/PMD. Each case archives the native
HSACO, compressed LLVM IR/TTGIR/AMDGCN, CU0 perf, launch/results/metrics and log.
Frozen-input manifests and source snapshots are included, with an artifact hash
manifest. No production kernel, dispatcher or CSV was modified here.

Raw traces remain in `/home/zhenx/temp/llvm-sync-fd18-20261117/gemm2-bm64/`.
The completed M128 diagnostic pair is isolated in its `diagnostic-m128/`
subdirectory and is not the result reported in this archive.

The recorded `controller.executed.py` contains exact execution paths. Replay
with explicit compiler binaries and the corresponding existing frozen input
fixture using the matching Triton Python environment:

```bash
python reproduce.py --bn 256 --old-clang /path/to/old-bin/clang-18 \
  --new-clang /path/to/new-bin/clang-18 --inputs /path/to/inputs-e256 \
  --output /path/to/replayed-comparison
```

Use `--bn512` as `--bn 512` with the BN512 fixture. Each compiler needs its
matching `ld.lld` beside it. The replay loads the archived source snapshot and
does not depend on a temporary source import. Input tensors are intentionally
not duplicated into the repository.
