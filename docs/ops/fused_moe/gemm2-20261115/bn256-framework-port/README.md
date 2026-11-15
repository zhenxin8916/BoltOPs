# BN256 GEMM2 framework port

Port of archived `88e09f4` (optimization `8836708`) into
`boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn256xbk128.py`.

The production entry traverses all N tiles. A/metadata generations advance per M,
B generations per N; consumers return A EMPTY once per M. A and the first B/SB
reads share one LDS completion wait, preserving the archived optimization.
P0 prefetches next-M A and scatters paired output panels; P1 loads both B halves.
16 abarriers; 119808 bytes LDS; default quota 48/16/224/224, no scratch spill.

Supports sorted channel FP8/INT8, INT32 accumulation for INT8, FP16/BF16 output,
optional routed-weight multiplication, scale strides, and independent compile-time
A/B/C 64-bit addressing. Wide paths fold A-row/B-panel origins into 64-bit bases.
The caller provides the CU budget. Unsupported shapes/layouts use ordinary Triton.
This specialization requires BM128/BN256/BK128, K256 and N divisible by 256.
Other GEMM2 implementations remain available for other geometries.

Framework selection chooses the sorted activation bridge before allocation and
tries this candidate before ordinary Triton. CK scatter uses ck_topk (the GEMM2
invocation itself has top_k=1) and existing SortedWeights. Device padded-row counts
are loaded inside each WASP role, as in GEMM1, before traversing work.

## Validation

167 metadata/dispatch/GEMM1 regression tests passed. Four numerical cases each
checked 197632 output elements: FP8/INT8 x BF16 with weights or FP16 without
weights. All match ordinary non-Gluon Triton bit for bit and CPU references within
rtol .02 / atol .002. Each case covers four M blocks, two N tiles, CK padding,
the framework candidate entry and device work count. FP16 cases force A/B/C
64-bit branches and non-unit scale strides; these are branch tests, not actual
allocations above 4 GiB. Input hashes and per-case results: equivalence.json.
The ordinary Triton comparison uses BM128/BN128/BK64 to avoid scratch allocation;
both backends compute the same complete outputs.

Run the opt-in numerical test on gfx946:

```bash
RUN_GLUON_GEMM2_EQUIVALENCE=1 python -m pytest test/fused_moe/test_channel_gemm2_equivalence.py -q
python -m pytest test/fused_moe/test_channel_gemm2_dispatch.py test/fused_moe/test_gluon_fused_experts.py -q
```

## Performance reproduction

1.8 GHz, 48 CU, M4096/E256/topk8, actual GEMM2 N2048/K256. This exactly follows
the archived diagnostic: every TG computes two independent M outputs, only the
first BN256 N panel, 96 output tiles total. N_TILES_LIMIT=1 is explicit in perf.py;
it is disabled in production. Each run checked 2190336 valid output elements.

| Type | Full dispatch time (us) |
|---|---:|
| Archived FP8 | 15.282772 |
| Port FP8 | 15.393416 |
| Port INT8 | 15.817644 |

The final FP8 difference is 0.724%. This is a toolchain/port retest, not a claim of a new
speedup over the archived kernel. Triton release/3.6.x is clean at 5b24821; the
archive used another Triton build. Toolchain revisions/LLVM changes are recorded
in provenance.json. No Triton source changes were made for this port.

The normalized bottom CSV uses four outputs per TG for BN256; these two-output
diagnostics do not replace that CSV row. Full-N correctness is tested separately.
The normalized four-output, three-clock CSV update is recorded in
[BN256 four-output results](../bn256-four-outputs-three-clocks/README.md).
Raw performance files remain under /home/zhenx/temp/moe-gemm2-bn256-port-20261115.

## GEMM1-style readability pass

English role/timeline comments, LDS sizes, READY/EMPTY/phase tables, register-view
panel diagrams and helper names now follow GEMM1. The old B-half barrier comment
was corrected to match the code: B-left uses READY5/EMPTY6 for both K panels;
B-right uses READY7/EMPTY8 for K0 and READY2/EMPTY4 for K1.

Only comments, formatting and four private helper names changed. Normalized ASTs
are equal; recompiling the FP8 performance specialization produced identical
9680-byte HSACO .text (SHA-256 in style-verification.json). All 167 regressions
passed again. No performance rerun was needed for this identical machine code.
