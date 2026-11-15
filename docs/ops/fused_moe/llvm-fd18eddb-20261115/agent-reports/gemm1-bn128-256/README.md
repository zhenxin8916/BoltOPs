# GEMM1 BN128/BN256 FP8 new LLVM retest

M4096 N1024 K2048 E256 topk8, fixed CK routing, 48 CU/48 TG, BN128 8 outputs/TG, BN256 4 outputs/TG, instruction warmup off

|BN|GC GHz|New time us|CU0 full MMAC %|Hot MMAC %|
|---|---|---|---|---|
|128|1.8|72.424560|54.393493|66.579827|
|128|2.2|66.430000|48.160614|59.362214|
|128|1.5|82.678656|57.409662|68.745026|
|256|1.8|65.973848|60.989819|70.752847|
|256|2.2|65.986832|49.995041|58.819927|
|256|1.5|70.351160|69.762936|81.393291|

All 8 runs passed 6,291,456 checked elements with zero mismatch, untouched output tail, no SGPR/VGPR spills and 128 KiB LDS. All CU0 wave traces pass first-wait nonzero, ID range 0..15 and no adjacent same-ID waits.

Old/new paired 1.8 GHz runs have exactly equal timings, CU0 efficiency, inputs, outputs, addresses, amdgcn and LLVM IR. Three frequencies share executable machine code. Whole HSACO hashes differ only in debug path sections.

The broader existing GEMM1 core suite reports 107 passed / 1 failed, specifically sparse >4 GiB B-scale case `test_channel_scale_sparse_stride_above_4g[b-False]`. Root is investigating the failure independently; this report does not claim all correctness features pass.

Raw evidence paths are preserved in report.json; no CSV/Git changes were made by this agent.
