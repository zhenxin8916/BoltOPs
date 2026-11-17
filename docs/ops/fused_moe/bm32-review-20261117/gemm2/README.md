# BM32 GEMM2 post-migration review

This review adds five GPU cases to the existing seven; it does not change the
production kernel, dispatcher, performance CSVs or original migration archive.
Each new case uses the same fixed seed3101 CK routes and CPU inputs as its two
backend comparisons. Every case checks all 395,264 output elements against a
dense CPU reference (rtol0.02, atol0.002) and ordinary Triton, with exact equality
between Gluon and ordinary Triton.

| Added case | A64/B64/C64 | Result |
|---|---|---|
| Only A wide, non-unit SA/SB strides | true/false/false | PASS |
| Only B wide, non-unit SA/SB strides | false/true/false | PASS |
| Only C wide, non-unit SA/SB strides | false/false/true | PASS |
| E5M2 A / E4M3FN B, FP16 output | true/true/true | PASS |
| Device WorkCount0, allocated448 sorted rows | false/false/false | PASS |

The zero-work case deliberately keeps TOTAL_TILES at allocation capacity.
All 386 route output rows (395,264 elements) remain at the initial -11 sentinel
for both Gluon and ordinary Triton. No work is silently inferred from capacity.
The independent-wide selectors are converted to three boolean constexprs;
strings identifying test modes never reach the kernel API. Compact allocations
exercise generated 64-bit paths, not physical tensor spans larger than 4GiB.

All five candidate HSACOs have zero SGPR spills, zero VGPR spills, zero private
segment bytes and no scratch instructions. The A-only wide specialization uses
55 P0 registers; the backend allocates a sufficient quota (68), rather than the
source's requested48. Requested, allocated and used quotas are recorded
separately in `no-spill.json` and the native AMDGCN confirms `s_set_vgpr_size`.
Consumer actual counts remain74/74; total LDS remains50,688B.

Manual review checked:

- One full-A slot is read by both consumers before the combined EMPTY credit;
  B and output EMPTY follow completed LDS reads, output READY follows writes.
- M and N epochs advance at their documented ownership granularity; next-M
  metadata is held in registers and published only after the first N phase.
- Full K256 contributions accumulate before FP32 scaling; INT8 uses INT32
  accumulation and converts only at the scaling boundary.
- A/B/C wide flags govern separate bases and offsets, including scale strides;
  output buffer element offsets match the BF16/FP16 typed store convention.
- Device padded-row count limits producers and consumers identically. General
  top-k/N support and CK route decoding agree with ordinary Triton; the module
  still explicitly rejects unsupported warmup=True.

No new production correctness defect was found. Tests close coverage gaps;
no algorithm rewrite, extra synchronization or model workaround was introduced.

The compiler and linker were copied before tests to isolate this review from
later LLVM syncing. The frozen clang SHA256 is
`0ec2b806f4f940fa67310006c3c2ef3ca602a0f6823538f988b0162405ea2add`
and reports build revisionfd18eddb. Before/after hashes, old Triton commit and
PMD binary hash are in `provenance.json`. An initial launcher setup omitted the
adjacent `ld.lld`, so those attempts failed at link time before GPU dispatch;
copying and hashing the matching linker fixed the setup. Raw setup-failure logs
remain separately under `temp/bm32-gemm2-review-20261117/setup-missing-lld` and
are not counted as numerical tests.

The five successful cases archive JUnit, compressed logs, native HSACO,
metadata and compressed LLVM IR/TTGIR/AMDGCN here. `correctness.json` retains input
hashes, actual device counts and wide flags. `review-test-snapshot.py` records
the extended suite. Only these five new cases were rerun on GPU; prior seven
results remain in the original migration archive. CPU support-query regression:
3 passed,12 GPU cases skipped when opt-in is disabled.
