# Persistent Gluon channel GEMM1

Implementation: `boltops/fused_moe/gluon/gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128.py`.
All supported geometries use the eight-slot shared A/B algorithm ported through
650d405bcdc41e55f6255d76c0afbbcd8b024189. Copy instructions and MMAC layouts
specialize at compile time for BN128/256, FP8/INT8, N tails and address widths.

## Supported calls

| Property | Support |
|---|---|
| Device | gfx946; caller-supplied positive CU budget, including multi-XCD |
| Inputs | Contiguous A[M,K], B[E,N,K]; E4M3FN/E5M2 including mixed FP8, or signed INT8 A/B |
| Scales | FP32 A[M,1], B[E,N] or B[E,N,1]; strided or broadcast |
| Tiles | BM128, BN128/256, BK128; K>=512 and K divisible by256 |
| Routing | CK sorting, c_sorted=True, a_sorted=False; topk1..256 |
| Output | Sorted/padded C in BF16 or FP16; invalid routes/experts produce zero |
| Traversal | GROUP_SIZE_M>=1, persistent tile_id+=grid, optional GPU CK work count |
| Addresses | Independent USE_ADDR_OFFSET_INT64_A/B/C; scale offsets widen from their strides |

The metadata-only support query runs before ordinary Triton dispatch. Unsupported
calls retain ordinary Triton, including GEMM2, bias, routed-weight multiplication,
other scale recipes and incompatible sorted layouts. Compile/runtime failures after
an accepted query are surfaced. The caller validates devices, routing buffers,
output allocation and downstream sorted activation/GEMM2 layouts.

CK IDs use low24 bits for the A token and high8 for its route; GEMM1 ignores
the route byte. Padding uses token M. Device work counts prevent idle TGs from
reading unused routing capacity. GROUP_SIZE_M matches ordinary Triton traversal.

## Roles and issue order

Each role has four waves. P0 owns both B halves; P1 owns A and channel scales.
C0 produces the left N half and C1 the right. All copies target the shared ring.

```text
                     startup              steady K issue order
P0 (B):          Bleft0 ------ Bright0 -------- Bleft1 -> Bright1 -------- ...
P1 (A):                  A0             A1                         A2 ...
                         exclusive A issue window, then resume B

C0: wait/read B -> wait/read A -> release B/A -> notify/wait handoff -> MMAC
C1: wait/read A -> wait C0 handoff -> wait/read B -> release A/B -> MMAC
```

Consumers process two BKs per loop, alternating ping/pong. C1 reads A before
its ping handoff. Per-SIMD count2 handoffs coordinate MMAC; an operand slot
becomes reusable only after every reader returns its EMPTY credits.
BN256 MLS has eight LDS reads per operand: lgkmcnt(8) releases the first-read
operand, then lgkmcnt(0) releases the second. Other layouts drain all reads
before either release. Signed INT8 accumulates in INT32 and converts once to
FP32 before channel scaling; FP8 accumulates in FP32. C.dtype selects BF16/FP16.
Instruction warmup is disabled by default (INSN_WARMUP=False). When explicitly
enabled it visits ping/pong instructions once per TG, with INT8 zeroing by MOV.
The paired-BK loop remains rolled with warmup on or off to bound operand lifetimes.

## Slot ownership and synchronization

Eight physical16-KiB slots use128KiB LDS and exactly16 abarriers for every
geometry. A occupies16KiB; BN128 B halves occupy8KiB within the same slot stride.
All operand and temporary views share this allocation.

Let k include all previously processed BKs. At an output's first BK,
Bleft/A/Bright take tickets3*k/3*k+1/3*k+2; later BKs use that order for
A/Bleft/Bright. slot=ticket&7 and phase=(ticket>>3)&1. Tickets remain continuous
across outputs, including K512/768; metadata and scale aliases rotate with them.

| Resource | IDs / count | Ownership |
|---|---|---|
| READY | (slot+1)&7 /4 | Four producer waves publish the copied operand |
| EMPTY |8+slot /8| Each A reader contributes1/wave; the sole B reader contributes2/wave |
| Producer metadata | EB1 /8 | P0/P1 capture routes/expert before A0 overwrites them |
| Exclusive A issue | EB3/4 /8 | P0 grants the window; P1 copies A and releases P0 |
| Scale READY | EB2 for C0, EB0 for C1 /8 | P1 publishes scales independently |
| Scale retirement | EB5 /16 | Next P0/P1 and both previous consumers coordinate reuse |
| MMAC handoffs | EB8..15 /2 | One directed handoff per SIMD/consumer |

Abarriers and ebarriers are independent namespaces. Initialization is batched;
EMPTY credits are seeded after a common rendezvous. Initial waits are nonzero;
adjacent wait IDs must differ. P0 prepares at priority1; P1 raises to2 only
inside the exclusive A-copy window.

```text
output i tail: P1 publish scales -> C0/C1 read to registers -> release -> store C
next output:   P0 metadata -> wait LDS stores -> publish EB1 -> Bleft0 ---- Bright0
               P1                         wait EB1 -> prepare A0 -> EB3 -> A0
                                                         scale retirement gates Bright0
```

Metadata occupies the next A0 slot. Scales occupy the next Bright0 slot, which
is distinct from metadata and Bleft0. Wait for each slot's last operand EMPTY
before repurposing it; the temporary stores do not advance operand READY phases.
The producer handshake prevents next Bright0 from overwriting scales until both
consumers have captured them. Initial metadata is loaded only after actual GPU
work is known. At K divisible by1024 the aliases specialize to slots1 and2.

## Copies, masks and addresses

A uses buffer-to-LDS with32-bit offsets for FP8 and INT8. The copy descriptor
uses an FP8 byte view of INT8 storage; consumers bitcast those bytes back to
INT8 before MMAC. There is no numeric conversion in this transport view.
Wide-A copies pack four unchanged input bytes into INT32 loads/stores,
then explicitly wait for LDS completion before READY. This reduces address
and byte-pack temporaries. This path uses64-bit global pointers and LDS stores.
B always uses interleave2 MLS. A constexpr N%(BN/2) condition omits
boundary_check entirely for divisible shapes. Supported tails check only N
against the remaining columns relative to the folded base; K is complete by
contract. MLS predicates suppress wholly absent halves. Shapes outside the
current native MLS boundary support fall back to ordinary Triton during the
CPU support query. There is no B copy-mode launch option.

A widens routed offsets when USE_ADDR_OFFSET_INT64_A is set. Each B copy folds
expert/N/K origins into a64-bit base before MLS. C widens its block index before
computing large bases. Scale strides independently select64-bit offsets, even
for small input tensors. USE_ADDR_OFFSET_INT64_B remains in the common launch
API; B always uses a64-bit folded base, so a large expert tensor does not require
a large buffer offset.

## Launch configuration

Use DEFAULT_LAUNCH_OPTIONS: num_warps=4, num_stages=2, async copy, WASP
partitions(4,4,4,4) and WDRA. Max-ILP uses the compiler default (disabled).
Both widths use quotas(32,48,216,216) with32-bit A offsets,
and(64,96,176,176) when USE_ADDR_OFFSET_INT64_A is set. There is no separate256-register budget for BN128. The kernel
and benchmark share gemm1_partition_regs; PARTITION_REGS overrides it explicitly.
A_STAGES is accepted for existing callers but does not change the eight-slot ring.

BN128 consumers use waves[4,1], BN256[2,2]. B always uses MLS.
Grid size comes from the caller's CU budget, with
hardware responsible for CU/XCD placement. No48-CU assumption is embedded.

Source timing uses clang's experimental -mllvm -hcu-control-only-ebarrier-sync
option. The library does not change global compiler settings. Numerical tests
also exercise ordinary clang; source timing requires its recorded option.
Explicit LDS waits protect operand release and scale publication.

## BN256 FP8 measurement reference (2026-11-14)

M4096/E256/N1024/K2048, BM128/BN256/BK128,48 CUs/48 TGs,
HBM1.8TB/s; only CU0 is traced. These are bounded diagnostics, not full MoE CSV
results. Both use CK-packed routes, quotas(32,48,216,216), max-ILP disabled,
and the control-only-ebarrier compiler option described above.

| GC | Outputs per TG | Local time(us) | MMAC window coverage | Source record time(us) |
|---|---:|---:|---:|---:|
|2.2GHz|1|16.486925|67.0898%|16.637530|
|1.5GHz|2|34.661989|84.1802%|34.105711|

All1,160,192 and2,336,768 valid CPU-reference values passed respectively,
with no padding errors or missing writes. The source record uses its own
input preparation; these differences are not an isolated algorithm comparison.
The two-output result is1.63% slower than the source record. Coverage measures
the union of8-cycle active MMAC intervals over each output's MMAC window,
summed across four SIMDs; it is not whole-kernel utilization.

Current clang SHA256: e3f14dd98d1188838e9f39c825cb546aa20fb25f5b4a79821d743c454cfa2941.
Artifacts: aligned-gc2200-1output/ and aligned-gc1500-2outputs/ under the port
artifact directory above. No global clock settings were changed.


## Compatibility review (2026-11-14)

The support query retains the previous public scope; unsupported inputs return
to ordinary Triton. All accepted geometries use the same eight-slot algorithm.
The current source passed 181 CPU checks, 20 CK/entry numerical checks and
12 large-address numerical checks. CK checks include topk129/256 signed packed
IDs, GPU work counts, grouped traversal and FP8/INT8 entry selection. Address
checks cover FP8/INT8, BN128/256, MLS/buffer B, A/B/C offsets above4GiB and
independent sparse scale offsets. Rebased pointers exercise actual wide
relative arithmetic while accessing small valid allocations.

Twelve BN128 compile profiles (FP8/INT8, MLS/buffer/wide A, warmup on/off)
are checked for zero spills/scratch; BN128 uses the same copy-dependent
quota policy as BN256. LDS remains128KiB per TG. The numerical checks
validate the CK metadata contract; they do not constitute a new end-to-end
benchmark of the CK sorting operator itself.


## Warmup comparison with persistent work

Before the early-metadata increment,48 persistent TGs each process8 output
tiles (384 diagnostic tiles, FP8/BN256, HBM1.8TB/s):

| GC | Warmup on(us) | Warmup off(us) | Off slowdown |
|---|---:|---:|---:|
|2.2GHz|124.346040|125.080865|0.590952%|
|1.5GHz|127.645791|130.620611|2.330527%|

All four runs check8,600,576 valid output elements with zero mismatches or
padding errors. These timing-only runs use identical inputs/options except
INSN_WARMUP and GC; each variant's native code matches its one-output diagnostic.
No hot-loop coverage was collected for these timing-only runs. The1.5GHz record
is warmup-comparison-gc1500-8outputs.json in the shared-ring-unify artifact directory.
These historical comparisons explicitly enable/disable warmup. The current
default is disabled; these timings do not measure the new default.

The MLS tail integration passed22 numerical checks on the original compiler,
including BN128/256, FP8/INT8 and BF16/FP16 at N248, plus181 CPU checks.
Divisible BN256 retains all1695 native instructions unchanged. Nonstandard
small-tail expansion and the isolated compiler experiment are not included.


## Early metadata publication

The port includes source commit650d405: after the next output's route/expert
stores finish, P0 arrives at EB1 before issuing Bleft0. P1 synchronizes on EB1
and prepares A0 concurrently; EB3 continues to order the actual operand copies.
The rotating metadata slot's previous owner must return its EMPTY credits before
those stores. EB5 still protects scale-slot reuse. K loops and barrier participant
counts are unchanged.

The source branch measured two outputs/TG at31.024630us (2.2GHz) and33.424037us
(1.5GHz), versus31.187520us and34.105711us before this change. These are source
experiment results, not a new timing measurement of this integrated port.

The integrated change passed181 CPU checks and47 numerical checks: both tile
widths, FP8/INT8, MLS/buffer paths, wide addresses, CK route bits, short K and
16 outputs/TG through K4096. Results are archived under
/home/zhenx/temp/boltops-shared-ring-unify-20261114/metadata-650d405-port/.


## Integrated-port timing after early metadata publication (2026-11-14)

Measured source:7f990ff; the following documentation amendment does not change
the kernel. M4096/E256/outputN1024/K2048, topk8, BM128/BN256/BK128, FP8 channel,
BF16 output, HBM1.8TB/s, warmup enabled. Grid48 runs on48 CUs with two output
tiles per persistent TG:96 diagnostic tiles out of1496, tracing only CU0.
Each frequency is one complete model execution, with the recorded control-only
EB sync compiler option and max-ILP disabled. No global clock settings changed.

| GC | Total time(us) | K-loop MMAC coverage | First output coverage | Second output coverage |
|---|---:|---:|---:|---:|
|2.2GHz|31.963750|60.03%|67.05%|54.34%|
|1.5GHz|33.804894|84.45%|92.50%|77.69%|

Total time is lastWaveEndTick minus firstWaveStartTick across the launch.
Coverage is the union of active8-cycle FP8 MMAC intervals over each output's
first-to-last MMAC window, weighted by window length across both consumers
and all four SIMDs. EXEC0 warmup is excluded; this is not whole-kernel utilization.
Both runs pass2,336,768 valid CPU-reference elements, with zero mismatches,
padding errors or missing writes. LDS is131072 bytes. Only XCD0/SE0/CU0 has
a nonempty instruction trace; all48 CUs still execute the kernel.

The previous local1.5GHz two-output measurement was34.661989us; this run is
33.804894us,2.47% lower. Input hashes, launch options, grid and diagnostic tile
count match. The source branch's31.024630/33.424037us records above use its own
input preparation and must not be treated as an isolated timing comparison.

Artifacts: metadata-7f990ff-gc2200-2outputs/ and
metadata-7f990ff-gc1500-2outputs/ under
/home/zhenx/temp/boltops-shared-ring-unify-20261114/.
Each directory includes launch/options, source and toolchain hashes, compiled
code, numerical results, stats, CU0 trace and a named .perf file. The combined
record is metadata-650d405-port/performance-summary.json in the same directory.


## Standalone AIDSL correctness cases

Instruction warmup now defaults toFalse; callers may explicitly opt in with
INSN_WARMUP=True. The timing tables above retain their recorded warmup settings.

boltops.tools.export_moe_gemm1 follows AIDSL's schema4 files workflow: build-side
Torch/Triton compiles the original library kernel and writes raw CPU inputs,
full CPU golden, HSACO/IR, the actual code-object ABI, and launch.json. Execution
uses the existing aidsl-bu-kernels/common/file_runner.cpp with HIP, without
Python, Torch, Triton or the BoltOPs package on the execution machine.

```bash
python -m boltops.tools.export_moe_gemm1 \
  --out /home/zhenx/aidsl-bu-kernels/cases/moe_gemm1
python -m boltops.tools.export_moe_gemm1 \
  --out /home/zhenx/aidsl-bu-kernels/cases/moe_gemm1 --output-dtype fp16
python -m boltops.tools.export_moe_gemm1 \
  --out /home/zhenx/aidsl-bu-kernels/cases/moe_gemm1 \
  --n 248 --k 768 --grid 2 --group-size-m 2
python -m boltops.tools.export_moe_gemm1 \
  --out /home/zhenx/aidsl-bu-kernels/cases/moe_gemm1 --grid 48 --bn 256
```

Build the existing file runner as in AIDSL's build_tilelang.sh, or compile only
that runner (set JSON_INCLUDE to the directory containing nlohmann/json.hpp):

```bash
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include -I"$JSON_INCLUDE" \
  /home/zhenx/aidsl-bu-kernels/common/file_runner.cpp \
  -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -o file_runner
./file_runner /home/zhenx/aidsl-bu-kernels/cases/moe_gemm1/fp8_bn128_bf16_m97_n256_k512_grid1
```

Use correctness mode. The generic runner's benchmark mode skips input-file
loading, including MoE routing/work counts, so it is not a valid MoE benchmark.
Its CLI --warmup controls repeated benchmark launches, independently of this
kernel's INSN_WARMUP option. PMD execution uses the usual GPU_CHIP=sb,
PMD_PATH=/opt/rocm/pmd and GPU_DFLAGS='["StatLog"]' environment settings.

Default cases use M97/E3/N256/K512, topk2, both BN128/256 and FP8/INT8. CK-packed
routes contain each token/route once, padding token M, and an invalid-expert
block. A GPU work count bounds all output tiles; grid1 runs4 or8 outputs/TG.
CPU golden computes dense expert GEMMs on decoded quantized operands before
independently gathering routed rows. Signed nonuniform inputs and channel scales
are deterministic (seed3197). All outputs, including padding and invalid experts,
are checked; the runner poisons output memory first to expose missing writes.
The exporter takes the disabled default without passing INSN_WARMUP explicitly
and rejects generated warmup labels, spills or scratch. It validates pointer/
scalar sizes against the compiled ABI, including the two null scratch arguments.

Generated cases live outside the BoltOPs checkout. Their .bin/HSACO/IR/JSON files
are build artifacts; the reusable exporter and CPU golden tests are in this commit.


The existing run_all.sh can select all exported MoE cases with a runner override:

```bash
RUNNER=/path/to/file_runner \
  bash /home/zhenx/aidsl-bu-kernels/run_all.sh moe_gemm1
```

Validation with the default disabled warmup:14 standalone C++/HIP cases pass
on gfx946 PMD, checking1,818,624 output elements with zero mismatches/nonfinite
values. Coverage includes both widths/input types/output types, N248/K768 tails,
group_size_m=2, continuous4/8-output persistence, invalid experts with a valid
token, padding and grid48 idle TGs. All14 compiled kernels have zero spills/
scratch and128KiB LDS. Another188 CPU checks pass, including an independent
per-row K-reduction check of exported goldens and bit-exact raw-file round trips.
Runner build log, per-case logs and source hashes are archived at
/home/zhenx/temp/moe-aidsl-validation-20261114/validation-summary.json.


## Frozen routing and Gluon/Triton equivalence

Random top-k assignments are generated once at export. Sorted.bin, Experts.bin
and WorkCount.bin are the complete CK routing inputs consumed by GEMM1. Subsequent
C++ runs and cross-backend comparisons load these files without RNG. All raw
inputs/goldens carry SHA256 digests. Exporting into an existing nonempty case now
raises FileExistsError, and raw blob writes are exclusive: rerunning the exporter
cannot silently replace the benchmark's routing. Use a fresh directory only when
intentionally creating a different dataset. load_frozen_case verifies hashes and
allows later kernel tests to reuse the original files unchanged.

```bash
AIDSL_MOE_CASE_ROOT=/home/zhenx/aidsl-bu-kernels/cases/moe_gemm1 \
  python -m pytest -q test/fused_moe/test_channel_gemm1_equivalence.py
```

Eight same-input comparisons pass for FP8/INT8, BN128/256 and BF16/FP16 output,
using M97/E3/N256/K512, topk2 and the saved CK metadata. Each compares131072
outputs:1,048,576 in total, zero differing elements and maximum absolute difference0.
Both backends also pass the independent CPU golden. Gluon uses its default
disabled warmup and grid1 persistence; ordinary Triton is explicitly selected and
its kernel launch is counted, preventing an accidental Gluon-versus-Gluon test.

Both output buffers are pre-zeroed according to ordinary Triton's contract:
that kernel leaves padding and invalid-expert rows untouched. Every output row is
then compared, including those zeros. The separate poisoned-output standalone
Gluon tests above still check its complete-write behavior. Input SHA256 digests
are checked before/after the comparison and recorded with each result.
Another190 CPU checks pass, including overwrite refusal and detection of a changed
routing byte. Comparison logs, per-case input hashes and results are archived at
/home/zhenx/temp/moe-gluon-triton-equivalence-20261114/comparison-summary.json.


## Normalized experiment CSVs (2026-11-15)

Default nominal GC is **1.8 GHz** (PMD period556ps, effective1.798561GHz),
with HBM1.8TB/s and48 CUs. Each CU runs one persistent TG. Keep padded
output work fixed at128*1024 elements per TG:

| BN | Output tiles per TG | Total tiles at48 CUs |
|---:|---:|---:|
|512|2|96|
|256|4|192|
|128|8|384|

The current GEMM1 kernel supports BN128/256; BN512 measurements belong to
GEMM2. GEMM1 uses M4096/E256/outputN1024/K2048/topk8 and processes the same
first6144 sorted output rows for both BNs. Frozen inputs are reused unchanged;
all measured outputs are checked against the CPU golden and the untouched tail
is checked separately. Warmup is disabled.

The performance tables live alongside the kernels in `boltops/fused_moe/gluon/`:

- `tuned_experiment_fmoe_gluon_w8a8channel_top_fp8.csv`
- `tuned_experiment_fmoe_gluon_w8a8channel_bottom.csv`

These tables contain FP8 results. INT8 results use separate `_top_int8.csv` /
`_bottom_int8.csv` files, created when validated results are available.

These are bounded kernel diagnostics, **not full-MoE or full-routing times**.
`time_us` is the complete HYGPERF dispatch, including startup;
the detailed JSON records preserve wave-span timing, frequency, work counts and
input/source hashes. The CSV keeps BM/BN/BK immediately before `time_us`,
followed only by the Triton/LLVM source commits and Chinese local-change notes.
These rows use nominal 1.8 GHz, 48 CUs and warmup disabled.
The bottom table records prior verified BN256/512 FP8 experiments; their
prefixes have equal padded work but different selected output subsets.
No `E=...` tuning JSON files are generated at this stage.

Update a validated result without duplicating its configuration row:

```bash
python -m boltops.tools.gluon_moe_experiment_csv --stage top --record record.json
```

The updater rejects wrong clocks, unequal work and failed validation, and
replaces matching rows atomically. `boltops.tools.gluon_moe_gemm1` also defaults
to1.8GHz; use an explicit `--gc-ghz` for a separate frequency experiment.

## Clean Triton release verification (2026-11-15)

The GEMM1 FP8 table uses Triton `5b24821ee57eeb7560074cc7979ce7b59604079f`
with no local source changes. `libtriton.so` was rebuilt from that checkout.
LLVM and its recorded local changes remain the same as the previous measurements.
At nominal 1.8 GHz, 48 CUs, M4096/E256/N1024/K2048, warmup disabled:

| BM | BN | BK | Outputs per TG | Full dispatch (us) | CPU-reference elements | Mismatches |
|---:|---:|---:|---:|---:|---:|---:|
|128|128|128|8|72.614712|6291456|0|
|128|256|128|4|65.418404|6291456|0|

Frozen input hashes, output hashes, and kernel `.text` bytes match the previous
runs. Both dispatch times are unchanged. BN128 had completed before the stop
request was handled; no additional run was started. GEMM2 and INT8 tables keep
their previous measurements and toolchain provenance.
Evidence: `gemm1-20261115/release-5b24821-gc1800/`.
