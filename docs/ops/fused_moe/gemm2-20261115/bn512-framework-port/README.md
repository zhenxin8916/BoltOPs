# BN512 GEMM2: two outputs per TG, three clocks

Current kernel: `gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn512xbk64.py`.
M4096, E256, topk8; actual GEMM2 N2048/K256; BM128/BN512/BK64.
48 CUs each run one persistent TG and two independent M output tiles: 96
BN512 tiles, or 128*1024 padded output elements per TG. This matches the
experiment CSV normalization. N_TILES_LIMIT=1 is an explicit diagnostic;
TOTAL_TILES=384 uses the kernel's full-N coordinate system. Production still
traverses all N. These are bounded GEMM2 experiments, not full-MoE timings.

| Type | GC (GHz) | Full dispatch (us) | CU0 whole-lifetime MMAC (%) |
|---|---:|---:|---:|
| FP8 | 2.2 | 19.059496 | 25.319116 |
| FP8 | 1.8 | 20.831652 | 28.235619 |
| FP8 | 1.5 | 24.097376 | 30.493207 |
| INT8 | 2.2 | 19.516770 | 23.660572 |
| INT8 | 1.8 | 21.605048 | 26.924341 |
| INT8 | 1.5 | 24.719688 | 28.912261 |

Each run checked 4,380,672 valid output elements with zero mismatches; unselected
rows/columns remained NaN. CPU reference tolerances: rtol .02, atol .002. Inputs,
CK routing and reference outputs were frozen once per type (seed3101). Across
clocks, each type has identical input/output hashes and HSACO. GPU addresses
are recorded per run and compared in provenance.json.

Only CU0 is traced. Efficiency is the four-SIMD average of active 8-cycle MMAC
interval unions, divided by CU0's first WAVE_START through its last s_endpgm
plus one cycle. This includes CU0 startup and tail; the denominator is not the
full 48-CU dispatch window. Dispatch time is read from the HYGPERF XCD time
section. All 48 CUs complete one TG and exactly 4096 MMAC instructions each;
CU0 has eight consumer waves with 512 active MMACs apiece. The interval union
equals instruction_count*8 cycles. FP8 and INT8 use their own opcode families.

CSV locations under boltops/fused_moe/gluon:

- tuned_experiment_fmoe_gluon_w8a8channel_bottom.csv (current FP8 BN512 row)
- tuned_experiment_fmoe_gluon_w8a8channel_bottom_int8.csv

The old BN512 row is archived in previous-bottom.csv; other geometries are
preserved. BN128/BN256 measurements were not changed by this task. Toolchain commit
IDs and Chinese change descriptions are included in each CSV. Triton 5b24821
has no local changes. Provenance records LLVM's local load-sinking patch and
clang's control-only ebarrier flag. No global frequency setting was modified.
Default measurement frequency remains 1.8 GHz; other clocks are per-process.

Reproduce inside the paired environment, keeping this directory writable:

```bash
python run.py --freeze
python launch.py fp8
python launch.py int8
python publish.py
```

launch.py specifies the exact Python/Triton/LLVM paths, 48-CU model and CU0-only
trace flags. Frozen tensor payloads and raw traces remain in the raw_root from
provenance.json; only scripts, manifests, validation and timing records are
tracked here. Publishing validates the numerical results, work normalization,
clock periods, one-TG-per-CU counts, input hashes and MMAC denominators first.

## Production integration and correctness

This ports the `a8ebfb1` BN512 B-batching algorithm (final archived record
`60c7865`) into one production kernel file. It keeps its K128 B publication,
M32 output mailboxes, 117-KiB LDS allocation and 16-barrier protocol. P0 owns
full-A/metadata and output scatter; P1 owns both B halves. Production traverses
all N; the CSV deliberately clips N to the first 512 columns for two independent
M outputs per TG. The CPU query selects BN512 for N divisible by 512 and K256;
other supported N geometries use BN256, and unsupported metadata uses ordinary
Triton. The sorted activation bridge is selected before its buffers are built.

FP8 (including E4M3/E5M2 metadata support) and INT8 accept BF16/FP16 output.
INT8 accumulates the complete K sum in INT32, then applies FP32 scales once.
Sorted weights are optional; scale strides are respected. A/B/C have independent
64-bit constexpr paths; large row/panel origins fold into 64-bit bases. Four
opt-in numerical cases cover FP8/INT8, both output types, complete N1024, four
M blocks, CK routing/padding, and forced wide paths with non-unit scale strides.
All 395,264 elements per case match ordinary Triton bit for bit, and each backend
passes the CPU reference (rtol .02, atol .002). Wide paths were forced on small
allocations; multi-GiB physical allocations were not tested. Metadata tests also
verify independent A/B/C wide flags using large meta tensors. There are 70
query/dispatch checks and 134 existing framework regressions, all passed.

Both 1.8-GHz traces pass the first-wait-nonzero, ID-range 0..15 and no-adjacent-
same-ID audits, with actual A/B generation checks. CU0 has 16 live waves and
four waves per SIMD. All eight consumer waves issue no VMEM loads or permlane.
Actual FP8 quotas are 40/9/203/203 registers; scratch and spills are zero. INT8
quotas and all compiler metadata are in correctness.json and compiled.json.
The production kernel uses Gluon APIs throughout, with no inline assembly.

Current FP8 GC1.8 elapsed time is 20.831652 us, 0.73% above the archived
20.680420-us result. The archive used an older compiler/runtime; the new result
includes the production metadata/full-N support on the clean 5b24821 Triton
baseline. This is the measured production port, not a claim that the older exact
timing was reproduced. The original archived best and records remain intact.

Run the production equivalence checks in the paired gfx946 environment:

```bash
RUN_GLUON_GEMM2_BN512_EQUIVALENCE=1 python -m pytest -s test/fused_moe/test_channel_gemm2_bn512_equivalence.py
python -m pytest -q test/fused_moe/test_channel_gemm2_bn512_dispatch.py test/fused_moe/test_channel_gemm2_dispatch.py
```
