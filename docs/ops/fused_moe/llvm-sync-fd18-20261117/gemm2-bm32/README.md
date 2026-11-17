# LLVM sync comparison: BM32 GEMM2

The unchanged production BM32/BN512/BK64 kernel was compiled and rerun with two
independent compiler/linker pairs. Both wrappers add only the existing
`-mllvm -hcu-control-only-ebarrier-sync` flag. No live old compiler wrapper was
used; old clang/linker were frozen before the sync, new clang/linker were copied
after the successful c49fb462 build. Before/after binary hashes are in
`provenance.json`; Triton and PMD remained unchanged.

Fixed workload: FP8 E4M3FN, BF16 output, M128/E256/N2048/K256/top-k8, seed3101,
48 persistent TGs, two independent M outputs per TG, first BN512 N tile only.
The exact original frozen A/B/scales/routes/weights were reused by both runs.

| Measurement | Old fd18-built compiler | New c49fb462-built compiler |
|---|---:|---:|
| Dispatch time | 15.466808 us | 15.466808 us |
| CU0 whole MMAC | 11.1943154% | 11.1943154% |
| CU0 compute-window MMAC | 19.0795603% | 19.0795603% |
| Actual P0/P1/C0/C1 registers | 35/12/74/74 | 35/12/74/74 |
| LDS bytes | 50,688 | 50,688 |
| SGPR/VGPR spill, private bytes | 0/0/0 | 0/0/0 |

Both runs passed all197,632 selected valid outputs against the dense CPU
reference (rtol0.02, atol0.002), preserved the untouched NaN output region and
produced identical output hashes. Native `.text` is byte-identical:
`85eeec2be054a4ab0c6e41c66dd66cadfdfcebf3a0cbc317d5224e5ad490b043`.
Static LDS/MLS instruction counts are identical, including one `ds_read2_b32`,
two `ds_read_b32`, eight `ds_read_b64`, twenty `ds_read_b128` and72 matrix-trans
reads. Complete opcode counts are in `comparison.json`. The separate synthetic
nested/disjoint-OR merge-probe regression did not change this kernel's native
instructions; this observation does not assert that probe is fixed.

Both native pipeline audits passed: first waits use nonzero IDs, consecutive
wait IDs differ, all IDs are within0..15, MLS is seq-tracked, and completed LDS
reads/writes precede EMPTY/output READY. No scratch instructions are present.
CU0 whole efficiency uses minIT WAVE_START to maxIT s_endpgm plus one GC cycle,
not the global dispatch interval. Perf files embed the captured CU0 wave trace.

Archive contents include old/new HSACOs, compressed AMDGCN/LLVM IR/TTGIR,
perf, launch/results/metrics and audits. Full raw IT and frozen tensors remain
under `/home/zhenx/temp/llvm-sync-fd18-20261117/gemm2-bm32` and the original
BM32 input directory. `controller.executed.py` records exact executed paths;
`reproduce.py` accepts explicit compiler and input paths and avoids a temporary
source import. For an existing matching Triton Python environment:

```bash
python reproduce.py --old-clang /path/to/old-bin/clang-18 \
  --new-clang /path/to/new-bin/clang-18 --inputs /path/to/frozen-inputs \
  --output /path/to/new-comparison
```

Each compiler must have its matching `ld.lld` beside it. An empty input directory
is generated once and checked against the archived frozen-input manifest.
No production kernel, dispatch or CSV was changed by this comparison.
