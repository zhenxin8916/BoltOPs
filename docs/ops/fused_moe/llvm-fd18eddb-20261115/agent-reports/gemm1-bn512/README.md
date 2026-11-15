# BN512 GEMM1 FP8 LLVM rebuild audit

| Build / clock / warmup | Full dispatch (us) | CU0 whole MMAC | Hot loop MMAC | Numerical |
|---|---:|---:|---:|---|
| new-top-bn512-fp8-gc1500 | 65.405352 | 74.2626% | 88.9148% | 0 / 6291456 |
| new-top-bn512-fp8-gc1800 | 61.429104 | 66.0239% | 80.3609% | 0 / 6291456 |
| new-top-bn512-fp8-gc1800-warm | 59.322976 | 68.7508% | 84.2871% | 0 / 6291456 |
| new-top-bn512-fp8-gc2200 | 59.522192 | 54.9531% | 66.4504% | 0 / 6291456 |
| previous-top-bn512-fp8-gc1800 | 61.429104 | 66.0239% | 80.3609% | 0 / 6291456 |
| previous-top-bn512-fp8-gc1800-warm | 59.322976 | 68.7508% | 84.2871% | 0 / 6291456 |

Same frozen routing, tensor addresses, input hashes, output hash and production source hash across all cases. New and previous LLVM produce byte-identical AMDGCN, LLVM IR and executable `.text` separately for warmup off/on. Whole HSACO hashes differ only in debug line sections; no machine-code difference.

48 CU, one TG per CU, two BN512 tiles per TG, BM128/BK128, M4096, 128 KiB LDS. No SGPR/VGPR spills and zero private segment in all compiled cases.

The existing correctness suite has 18 passed cases; this includes FP8 e4/e5/mixed, output BF16/FP16, forced 64-bit addressing and ordinary Triton comparison. No INT8 jobs were launched by this audit.

Metrics use the shared trace analyzer and its three wait-ID hardware checks. Timing rows require successful numerical verification and untouched tail. Warmup results remain separate from default-off performance.
