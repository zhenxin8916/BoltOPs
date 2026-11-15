# Wide A MLS diagnostic — not promoted

1.8 GHz, 48 CU, 2 independent M output tiles per persistent TG; BM128 BN256 BK128 K256. All four executions: 2,190,336 checked elements, zero mismatches. Wider A shared tile64×128 emits four matrix_load_128x16_b8 per P0 wave instead of eight matrix_load_64x16_b8; same32KiB A allocation.

The frozen backend incorrectly expanded every M subrow in each unit-M16 consumer repetition (1024 values per thread instead of256). The isolated diagnostic patch chooses the logical M16 row within its physical MLS tile. Full numerical validation and phase/wait/MMAC/consumer-VMEM audits pass. Control retains identical native instructions and15.282772us; path/debug strings differ. Diagnostic wide A is slower: warps2×2 15.513512us, 4×1 15.673084us, 1×4 15.661408us. Do not promote source or compiler library. Generic packed-scale/other-dtype coverage has not been established; patch is a diagnostic, not a general compiler fix.

Base frozen libtriton SHA256 ddfa0f4ab5d93ffdb99f4fbd367d6a1f42a2ffec0f8f44545846b2983dc95dd4; isolated diagnostic1dea26387888ef7f12f4f0e85757e3e20851cf7d084868e5f1eba4835d6e5eb3. Original toolchain was not replaced. Build commands/source/object remain in /home/zhenx/temp/moe-gemm2-20261114/wide-a-unit-consumer-fix.
