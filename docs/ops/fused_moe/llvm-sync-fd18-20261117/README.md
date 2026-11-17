# LLVM upstream sync with control-only ebarrier waits

The LLVM checkout tracks `origin/dtk-llvm-6.3.3`. After fetching, its upstream
HEAD remained `ff89db989ee3d0c990b77567090126a52361eedf`. The working branch now
contains only the cherry-pick of `fd18eddbaf4a662ae5ce86c516d28e5b16952513`:
`c49fb46267bc821ddae82940aed7cec8c8f7d374`. No matrix-store NOP protection or
local LDS-address recovery patch was retained. The LLVM working tree is clean.

The former checkout is protected by
`backup/dtk-llvm-before-sync-20261117-89bee503`. Its two untracked files and
frozen compiler/linker/tools are preserved outside the checkout under
`/home/zhenx/temp/llvm-sync-fd18-20261117/`. BoltOPs is also protected by
`backup/gluon-before-llvm-sync-20261117` after the BM32 review.

The previous source checkout was `89bee503`, but the actual old executable
reported build revision `fd18eddb` (SHA256 `0ec2b806...`). The new executable
reports `c49fb462` (SHA256 `11c1d097...`). Comparisons use explicit frozen old
and new compilers, matching `ld.lld`, distinct JIT caches, and
`-mllvm -hcu-control-only-ebarrier-sync` on both sides. See `provenance.json`
for full hashes; source HEAD is not substituted for executable identity.

## Compiler regression probes

Both DEFAULT and CONTROL checks of the retained ebarrier MIR regression pass.
The old compiler passes both IR and machine-code checks of both LDS-address
probes. The new compiler fails both IR checks and the `lds-common-base` GCN
check; the wrapped-constant probe still forms its expected read2 instructions.

Two functions, `loop_nested_disjoint_or` and `variable_disjoint_or`, each lose
one `ds_read2st64_b64`, replaced by two `ds_read_b64`. This is a pure compiler
comparison: no PMD execution, model workaround, source instruction substitution,
or extra synchronization is involved. Upstream therefore does not yet fully
replace local commit `8ed29ebc`'s address recovery. The missing normalization
leaves equivalent masked/OR LDS addresses as different expressions; the DS
optimizer cannot prove their common base and constant offset, so it retains
separate reads and more address arithmetic.

The fixtures, old/new optimized IR, assembly, FileCheck diagnostics and exact
validation statuses are retained in `ds-probes/`, `validation.json` and
`ds-pairing-comparison.json`. Failed expectations are recorded as regressions,
not removed, weakened, or counted as passing.

## Production-kernel comparisons

All comparisons use gfx946 PMD `b61a79c5`, GC 1.8 GHz, 48 CUs, fixed seed3101
routes and input manifests. BM32 uses M128, BM64 uses M1024, BM128 uses M4096.
GEMM1 BM128 normalizes per-TG work across BN: BN128/256/512 compute 8/4/2 tiles.
GEMM2 tests emit two independent M output tiles per TG at N0; they do not time
all N tiles. Work counts, selection masks and untouched output tails are
checked explicitly. These are matched old/new deterministic model comparisons,
not wall-clock timing of Python or data transfer.

The top-level summary and each subdirectory retain measured durations,
correctness results, native text/DS counts and spill checks. A source `.llir`
artifact precedes WASP's possible register-quota retry; only successful final
HSACOs establish a spill-free executable. Failed BN512 compilation is not
counted as a GPU numerical failure or assigned a fabricated time.

Historical three-frequency CSV rows are unchanged: this comparison reruns only
1.8 GHz. The independent report avoids relabelling unmeasured 2.2/1.5 GHz rows
with a new compiler identity.

## Matched old/new timings at 1.8 GHz

| Kernel | Old (us) | New (us) | Change | Native text / DS |
|---|---:|---:|---:|---|
| GEMM1 BM32 BN256 | 36.580352 | 36.580352 | +0.00% | identical |
| GEMM1 BM64 BN512 | 69.403808 | 69.403808 | +0.00% | identical |
| GEMM2 BM32 BN512 | 15.466808 | 15.466808 | +0.00% | identical |
| GEMM2 BM64 BN256 | 13.561396 | 13.561396 | +0.00% | identical |
| GEMM2 BM64 BN512 | 18.651020 | 18.651020 | +0.00% | identical |
| GEMM1 BM128 BN128 | 72.424560 | 73.076192 | +0.90% | changed |
| GEMM1 BM128 BN256 | 65.973848 | 67.652416 | +2.54% | changed |
| GEMM1 BM128 BN512 | 61.277316 | compile failed | — | not produced |
| GEMM2 BM128 BN256 | 23.618880 | 23.618880 | +0.00% | identical |
| GEMM2 BM128 BN512 | 20.994004 | 20.994004 | +0.00% | identical |

All nine successfully compiled new kernels passed their CPU-reference, padding
and untouched-output checks, and have zero final spills/private storage. BM128
GEMM1 BN128/BN256 also have byte-identical old/new outputs. BN512 failed during
compile before a GEMM dispatch. No performance value is assigned to that failure.

GEMM1 BM128 BN128 loses 12 static `ds_read2st64_b64` instructions, gains24
`ds_read_b64` and36 VALU. BN256 loses24 paired reads, gains48 single reads,
108 VALU and2 SALU. These are static whole-kernel counts; they are not dynamic
iteration totals. Old/new input LLVM IR is identical for these two paths.

A first-attempt BN512 compile reports role demands8/16/240/252, a4-aligned
sum516 exceeding the512 VGPR per-SIMD budget. The old executable succeeded
after adjusting final quotas to16/24/236/236; these quotas are not the old
actual live-register demands. The new Triton quota checker rejects
it before producing a launchable final kernel. The diagnostic assembly has
spills and is labelled **first attempt only**, not a usable code object.

The other seven geometries preserve native `.text`, DS/MLS instructions and
matched-run time exactly. This does not establish that the missing address
recovery is unnecessary: BM128 GEMM1 demonstrates both instruction and
compilation regressions. Only the requested control-only patch remains applied.
