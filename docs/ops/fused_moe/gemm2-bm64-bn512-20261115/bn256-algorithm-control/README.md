# BN256 algorithm transplanted to the BN512 output scope

BN256 retained commit `973bc63` is 12.168060 us / whole-dispatch MMAC 9.358008%.
Its two M64/N256 outputs per TG perform half the MMAC work of BN512 two outputs;
that is a latency comparison, not equal-work throughput. Both use 48 CUs at 1.8 GHz.

This control copies the retained BN256 algorithm: two complete K256 A slots,
full M64/N128/K256 MFMA, consumer scaling and natural-layout packed dword scatter,
no output mailbox. Two N128 phases per consumer cover the original BN512 output
columns, retaining exactly the BN512 frozen inputs, addresses, selected outputs
and 2048 active MMAC instructions per CU. LDS is 101376 bytes; no compiler spills.

Result: **16.098424 us / 14.146577% whole-dispatch MMAC**. 2630144 selected elements
have zero mismatches; untouched tail and output hash match the retained BN512.
CU0 wait-ID range/first-ID/adjacent-ID audit passed, with 16 active waves, four per
SIMD, no exec-zero MMACs; consumers have no global/buffer loads. This is a bounded
FP8 diagnostic, not full-N/INT8 promotion validation.

CU0 SIMD0 first-to-second N-phase MMAC issue gaps are C0 3004 / C1 3540 cycles.
They contain 48 packed FP32 multiplies, 16 BF16 conversions, 16 buffer_store_dword,
route/mask/address work, then B/SB LDS reads. These are issued-instruction intervals,
not isolated wait latency. The larger M64 epilogue remains on the consumer's path
to its next MMAC. This confirms that consumer direct scatter plus full-M64 compute
is not automatically faster when transplanted; it does not isolate every cause.

Retained BN512 commit `7820cab` remains **15.145996 us / 15.036159%**, so this
control is 6.29% slower and its operator is not promoted. The screenshot's separate
B/SB READY experiment is in progress in the BN256 conversation; it is not the
source of that conversation's retained 12.168060 us result. Existing BN512 separate
B/SB readiness control was 15.927732 us and was also rejected.

Raw case and source: `/home/zhenx/temp/moe-gemm2-bm64bn512-20261115/bn256-algorithm-two-n-phases`.
Raw perf path/hash is preserved in model-metrics.json. No compiler sources changed.
