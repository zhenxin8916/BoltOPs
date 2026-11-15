# Rebuilt LLVM: FP8 Gluon verification

LLVM fd18eddbaf4a662ae5ce86c516d28e5b16952513; clean Triton 5b24821ee57eeb7560074cc7979ce7b59604079f. Compiler hashes and version are attached.

M4096, fixed CK routing, 48 CU / 48 persistent TGs; BN128/256/512 process 8/4/2 tiles per TG. Warmup is disabled. Time covers the dispatch; MMAC is the four-SIMD average over CU0's own first-wave-start to last-wave-end window. Frozen inputs are unchanged across clocks and compilers. INT8 CSVs have also been refreshed; see int8-summary.json.

All five FP8 configurations passed numerical checks with no spills. GEMM1 1.8 GHz machine code and time match the previous compiler exactly. GEMM2 BN256 changes 23.470428 -> 23.499340 us (+0.123%); BN512 changes 20.994004 -> 21.144124 us (+0.715%). BN256 2.2 GHz uses the canonical 1.8 GHz cache to avoid independent-compilation scheduling variation; the agent audit of the initial independent run is retained separately.

Core tests: 107 valid cases passed. The sole raw failure was the obsolete N129 test now removed. Both retained A-scale >4-GiB cases passed after removal. GEMM1 BN512 passed 18 cases; GEMM2 modules passed four each. Metadata tests passed 204 (one obsolete case was skipped in that earlier collection).

The standalone LLVM barrier-order test was removed at user request. Removing it does not change code generation or fix ds_read sinking. Actual kernel numerical/performance tests are retained. The separate matrix-store wait unit test fails on both old and rebuilt LLVM. Diagnostic outcomes are in llvm-test-comparison.json.

Raw compiler snapshots, frozen input references and traces are under /home/zhenx/temp/llvm-fd18eddbaf4a-retest-20261115. They are intentionally not copied into git. Per-run manifests and analysis scripts provide reproduction details.

## INT8 completion

All 15 INT8 configurations (GEMM1 BN128/256/512, GEMM2 BN256/512, each at 2.2/1.8/1.5 GHz) passed numerical checks with untouched output tails and valid abarrier wait rules. Previously completed measurements were audited and published after INT8 work resumed. GEMM2 BN256 2.2 GHz was rerun with the canonical 1.8 GHz cache; all three clocks now execute identical .text for each configuration. Sources and compiler hashes were checked against the run manifests; fixed routing/input hashes and output hashes match across clocks. Existing FP8 data is unchanged.
