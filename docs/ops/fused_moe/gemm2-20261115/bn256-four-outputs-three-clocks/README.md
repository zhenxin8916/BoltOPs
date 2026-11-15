# BN256 GEMM2: four outputs per TG, three clocks

Current kernel: `gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn256xbk128.py`.
M4096, E256, topk8; actual GEMM2 N2048/K256; BM128/BN256/BK128.
48 CUs each run one persistent TG and four independent M output tiles: 192
BN256 tiles, or 128*1024 padded output elements per TG. This matches the
experiment CSV normalization. N_TILES_LIMIT=1 is an explicit diagnostic;
TOTAL_TILES=1536 uses the kernel's full-N coordinate system. Production still
traverses all N. These are bounded GEMM2 experiments, not full-MoE timings.

| Type | GC (GHz) | Full dispatch (us) | CU0 whole-lifetime MMAC (%) |
|---|---:|---:|---:|
| FP8 | 2.2 | 21.406840 | 21.728290 |
| FP8 | 1.8 | 23.618880 | 24.892130 |
| FP8 | 1.5 | 26.148400 | 27.257603 |
| INT8 | 2.2 | 22.046116 | 20.840011 |
| INT8 | 1.8 | 24.018088 | 23.661939 |
| INT8 | 1.5 | 26.509248 | 26.215239 |

Each run checked 4,330,752 valid output elements with zero mismatches; unselected
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
- tuned_experiment_fmoe_gluon_w8a8channel_bottom.csv (current FP8 BN256 row)
- tuned_experiment_fmoe_gluon_w8a8channel_bottom_int8.csv

The old BN256 row is archived in previous-bottom.csv; other geometries are
preserved. BN512 measurements were not changed by this task. Toolchain commit
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
