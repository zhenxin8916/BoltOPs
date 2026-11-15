# BN512/BK64 dedicated baseline

1.8GHz,48CU,48 physical persistent TGs, two independent M output tiles per TG, each128x512, K256. N2048 actual tensor; diagnostic truncates N traversal to first512 columns.96 wide tiles;4,380,672 valid elements checked, zero errors.22.131580us, full48CU dispatch MMAC20.580329%. Prototype and cleaned dedicated repository source independently reproduce the same time; repository bytes match the archived clean source. Wide work is twice the old96xBN256 diagnostic, so this is NOT a proven20% improvement over historical BN256. Full N traversal not covered.

2P2C,4waves per partition. P0: full A preload, SA/weight/route/expert LDS metadata, BF16 buffer_store_dwordx4 scatter. P1: BL0 BR0 BR1 BL1 BR2 BL2 BR3 BL3 per N subphase, SB at K0. C0 owns columns0..255 and C1 256..511, each computes two128-column subphases, retaining A once/M. K0..K2 update M64 accumulators; K3 computes/scales/writes fourM32 panels. FP32 order((acc*SA)*SB)*weight preserved.

85KiB LDS: A32KiB, Bleft2x8KiB, Bright2x8KiB, output2x8KiB, MA4KiB, SB1KiB. Actual VGPR40/9/187/187, no spill. BL READY5/10 EMPTY6/14; BR READY7/2 EMPTY8/4. K0..3 phase0,0,1,1; two complete toggles restore start phase between subphases. A READY1/EMPTY3 shared between consumers, returned once/M after actual LDS capture. AB10 is now second-Bleft READY, not a compute token.

Passed real IT audits:16 waves,4096 valid MMAC/CU,exec0=0,8-cycle busy union; barrier ID bounds, first wait nonzero, no adjacent same-ID waits; explicit A/B generation sequence; consumer0VMEM/0permlane; A-only priority2 and restoration0. Numerical scope is the96 wide outputs above.

Rejected full-width consumer prototype: demand40/9/256/256, exceeds512 total VGPR budget. Rejected first single-left-slot N128 prototype: numerical pass but adjacent READY5 waits violate hardware restrictions; PMD time is not a valid candidate metric.

Original frozen runtime retained; not the wide-A compiler diagnostic. libtriton SHA256 ddfa0f4ab5d93ffdb99f4fbd367d6a1f42a2ffec0f8f44545846b2983dc95dd4. Inherited toolchain provenance includes LLVM/Triton branch,commit,local changes,binary hashes. Perf path/hash in perf.json; reproducible source, launcher, binary and audits included. Container /zhenxin maps host /home/zhenx.
