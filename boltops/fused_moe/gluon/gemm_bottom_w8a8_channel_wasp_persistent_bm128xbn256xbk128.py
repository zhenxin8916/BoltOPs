# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent FP8/INT8 channel GEMM2 for gfx946, BM128/BN256/BK128.

Contiguous FP8 (E4M3FN/E5M2, including mixed formats) or signed INT8 A/B;
FP32 channel scales; BF16/FP16 scatter output, before the top-k reduction.
Requires CK-sorted A/routes, K256 and N divisible by 256. Four waves per role.

Roles and data flow:
    P0: A + SA/weight/routes -> C0/C1; output buffers from C0/C1 -> scatter C
    P1: B-left + SB-left -> C0; B-right + SB-right -> C1
    C0: left 128 columns; C1: right 128 columns. Both retain A across N.

Persistent traversal (arrows show issue order, not completion):
    P0 startup: load A(M0) + metadata(M0).
    P0 each M:  prefetch A(next M) + metadata -> drain all N output panels.
    P1 each N:  B-left K0 -> B-right K0 -> B-right K1 -> B-left K1.
    C0/C1, first N: wait/read A -> wait/read B(K0)/SB -> one LDS wait
                                  -> release A/B -> MMAC K0.
    C0/C1, later N: reuse A in registers; wait/read B(K0)/SB -> release B.
    C0/C1, every N: wait/read B(K1) -> release B -> MMAC/scale -> publish C.
    Each read completes before its EMPTY return. A needs both readers' credits.

One full-A LDS slot (32 KiB); B-left has one K128 slot, B-right has two
(48 KiB total). Each consumer publishes pairs of 32-row panels into its
64-row output buffer (32 KiB total). Metadata/scales take another 5 KiB.
Total: 117 KiB LDS, 16 initialized abarriers. A/metadata generations advance
per M; B generations advance per N. Grid is caller-supplied; M += grid.
INT8 accumulates in INT32 and converts once before FP32 scaling.
Instruction warmup is optional and defaults off. Production traverses all N; clipping is diagnostic only.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

# Launch options shared by the library dispatcher and standalone probes.
DEFAULT_LAUNCH_OPTIONS = dict(
    num_warps=4, num_stages=2, use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4), wasp_wdra=True,
)
MMAC_LAYOUT = gl.constexpr(3)  # LIT=0, LTS=1 (TRANSPOSE=3).
ROW_PERSISTENT_A = True
GIB = gl.constexpr(1024 * 1024 * 1024)  # Binary gigabyte, in bytes.


@gluon.constexpr_function
def _output_pack_layout(layout):
    """Pair adjacent output halfwords in registers before the LDS handoff."""
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:],
        layout.lane_bases, layout.warp_bases, layout.block_bases, layout.shape)


@gluon.jit
def _empty():
    """Keep the default WASP partition empty; four roles own all work."""
    pass


@gluon.jit
def _initialize(STAGES: gl.constexpr):
    """Initialize READY/EMPTY handoffs before splitting into P0/P1/C0/C1.

        Buffer             Producer -> Reader    READY  EMPTY  Phase
        A, full K256       P0 -> C0 + C1            1      3    M epoch & 1
        B-left, K0/K1      P1 -> C0                 5      6    K0=0, K1=1
        B-right, K0        P1 -> C1                 7      8    N epoch & 1
        B-right, K1        P1 -> C1                 2      4    N epoch & 1
        C-left, 64 rows    C0 -> P0                 9     11    panel 0/1
        C-right, 64 rows   C1 -> P0                13     15    panel 0/1
        M metadata        P0 -> C0 + C1          12/0     -    (M epoch >> 1) & 1

        producer                            reader(s)
          wait EMPTY -> load/store -> READY ---> wait READY -> read
               ^                                               |
               +------------------ EMPTY <---------------------+

    P0 publishes shared A and alternates two M-metadata slots. P1 publishes
    each B panel; its K0 READY also covers the corresponding SB stores.
    C0/C1 each return four A credits (8 total); B/output EMPTY needs four.
    READY counts four producer waves. Startup seeds EMPTY only, with two
    credits per wave for A and one for each B/output buffer. IDs 10/14 are
    reserved; metadata reuse is bounded by P0 draining this M's outputs.
    """
    hcu.abarrier_init(1, 4)
    hcu.abarrier_init(3, 8)
    for i in gl.static_range(2, 16):
        if i != 3:
            hcu.abarrier_init(i, 4)
    hcu.abarrier_init(0, 4)
    gl.thread_barrier()
    hcu.abarrier_arrive(3, 2)
    hcu.abarrier_arrive(6)
    hcu.abarrier_arrive(4)
    hcu.abarrier_arrive(8)
    hcu.abarrier_arrive(11)
    hcu.abarrier_arrive(15)
    hcu.abarrier_arrive(10)
    gl.thread_barrier()


@gluon.jit
def _load_a_and_metadata(A, SA, Weights, Sorted, Experts, am, ma, m, rows, epoch,
                         K: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr,
                         USE_A64: gl.constexpr, STRIDE_SA: gl.constexpr,
                         MUL_ROUTED_WEIGHT: gl.constexpr):
    """P0 loads one full-A slot and publishes this M's metadata.

        A slot: 0       READY=1, EMPTY=3       phase = epoch & 1
        MA slot: epoch & 1, four [128] vectors: SA, weight, route, expert
        MA READY: 12 for even epochs, 0 for odd; phase = (epoch >> 1) & 1

    Both consumers must finish reading A before P0 reuses slot 0. Metadata
    has two M slots, so P0 can prepare next M while this M is still computing.
    """
    ticket = epoch
    slot = 0
    phase = ticket & 1
    hcu.set_priority(2)
    hcu.abarrier_try_wait(3 + slot, phase)
    hcu.abarrier_seq(1 + slot)
    if USE_A64:
        # Fold the large row origin into the 64-bit descriptor base.
        a_base = A + m.to(gl.int64) * BM * K
        hcu.matrix_load(am.index(0), a_base, (BM, K), (K, 1), (0, 0), warps=(4, 1))
    else:
        hcu.matrix_load(am.index(0), A, (rows, K), (K, 1), (m * BM, 0), warps=(4, 1))
    hcu.abarrier_arrive(1 + slot)
    hcu.set_priority(0)
    hcu.sched_barrier()
    ml: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    mr = gl.arange(0, BM, ml)
    if USE_A64:
        row = m.to(gl.int64) * BM + mr.to(gl.int64)
        sa = gl.load(SA + row * STRIDE_SA)
        if MUL_ROUTED_WEIGHT:
            weight = gl.load(Weights + row)
        else:
            weight = gl.full((BM,), 1, gl.float32, ml)
        route = gl.load(Sorted + row)
        expert = gl.load(Experts + m.to(gl.int64) + 0 * mr)
    else:
        sa = buffer_load(SA, ((m * BM + mr) * STRIDE_SA).to(gl.uint32))
        if MUL_ROUTED_WEIGHT:
            weight = buffer_load(Weights, (m * BM + mr).to(gl.uint32))
        else:
            weight = gl.full((BM,), 1, gl.float32, ml)
        route = buffer_load(Sorted, (m * BM + mr).to(gl.uint32))
        expert = buffer_load(Experts, (m + 0 * mr).to(gl.uint32))
    ma.index((epoch & 1) * 4 + 0).store(sa)
    ma.index((epoch & 1) * 4 + 1).store(weight)
    ma.index((epoch & 1) * 4 + 2).store(route.to(gl.float32, bitcast=True))
    ma.index((epoch & 1) * 4 + 3).store(expert.to(gl.float32, bitcast=True))
    hcu.wait_lgkmcnt(0)
    hcu.abarrier_arrive(12 if (epoch & 1) == 0 else 0)


@gluon.jit
def _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch,
                KH: gl.constexpr, HALF: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                BN: gl.constexpr, BK: gl.constexpr, USE_B64: gl.constexpr,
                STRIDE_SBE: gl.constexpr, STRIDE_SBN: gl.constexpr):
    """P1 waits EMPTY, issues one [K128,N128] MLS panel, then publishes READY.

        HALF=0 (C0): K0/K1 reuse b0[0], READY5/EMPTY6, phases 0/1.
        HALF=1 (C1): K0 uses b1[0], READY7/EMPTY8;
                     K1 uses b1[1], READY2/EMPTY4; phase = epoch & 1.

    K0 also stores this N half's SB. Its READY covers both B and SB.
    """
    ready = (7 if KH == 0 else 2) if HALF == 1 else 5
    empty = (8 if KH == 0 else 4) if HALF == 1 else 6
    phase = (epoch & 1) if HALF == 1 else ((epoch * 2 + KH) & 1)
    physical = KH if HALF == 1 else 0
    dest = b1 if HALF == 1 else b0
    hcu.abarrier_try_wait(empty, phase)
    hcu.abarrier_seq(ready)
    if USE_B64:
        # Fold the N-panel origin into the 64-bit base, keeping MLS offsets local.
        panel_base = b_base + (n.to(gl.int64) * BN + HALF * (BN // 2)) * K + KH * BK
        hcu.matrix_load(dest.index(physical), panel_base, (BK, BN // 2),
                        (1, K), (0, 0), warps=(2, 2))
    else:
        hcu.matrix_load(dest.index(physical), b_base, (K, N), (1, K),
                        (KH * BK, n * BN + HALF * (BN // 2)), warps=(2, 2))
    if KH == 0:
        hcu.sched_barrier()
        ml: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
        mr = gl.arange(0, BN // 2, ml)
        col = n * BN + HALF * (BN // 2) + mr
        if USE_B64:
            sb = gl.load(SB + gl.maximum(expert, 0).to(gl.int64) * STRIDE_SBE + col.to(gl.int64) * STRIDE_SBN)
        else:
            sb = buffer_load(SB, (gl.maximum(expert, 0) * STRIDE_SBE + col * STRIDE_SBN).to(gl.uint32))
        mb.index(HALF).store(sb)
        hcu.wait_lgkmcnt(0)
    hcu.abarrier_arrive(ready)


@gluon.jit
def _four_row_vectors(x):
    """Register views: [128] -> 2 x [64] -> 4 x [32], matching output panels."""
    halves = gl.split(x.reshape((2, 64)).permute((1, 0)))
    low = gl.split(halves[0].reshape((2, 32)).permute((1, 0)))
    high = gl.split(halves[1].reshape((2, 32)).permute((1, 0)))
    return low[0], low[1], high[0], high[1]


@gluon.jit
def _producer(A, B, Experts, am, b0, b1, C, Sorted, valid_routes, co0, co1,
              SA, SB, Weights, ma, mb, pid, total, rows,
              N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr,
              BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr,
              USE_MLS_LOAD_A: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
              PARTITION: gl.constexpr, TOP_K: gl.constexpr, CK_SORTING: gl.constexpr,
              N_TILES: gl.constexpr, USE_A64: gl.constexpr,
              USE_B64: gl.constexpr, USE_C64: gl.constexpr,
              STRIDE_SA: gl.constexpr, STRIDE_SBE: gl.constexpr, STRIDE_SBN: gl.constexpr,
              MUL_ROUTED_WEIGHT: gl.constexpr, num_tokens_post_padded_ptr):
    """P0 prefetches next-M A/metadata and scatters C; P1 streams B across N.

        P0, before N loop: current routes -> next-M A/metadata prefetch
        P0, inside N loop: wait/read C0 panel -> EMPTY -> scatter left
                           wait/read C1 panel -> EMPTY -> scatter right
        P1, inside N loop: BL(K0) -> BR(K0) -> BR(K1) -> BL(K1)

    P0's epoch counts M preloads; P1's epoch counts N tiles across all M.
    """
    if num_tokens_post_padded_ptr is not None:
        total = gl.minimum(total, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * (N // BN))
    epoch = 0
    ml: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    mr = gl.arange(0, BN // 2, ml)
    output_store: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    output_rm = gl.arange(0, 64, gl.SliceLayout(1, output_store))
    output_rn = gl.arange(0, BN // 2, gl.SliceLayout(0, output_store))
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        if PARTITION == 0:
            if m == pid:
                _load_a_and_metadata(A, SA, Weights, Sorted, Experts, am, ma, m, rows, epoch, K, BM,
                    BK, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT)
                epoch += 1
            # Capture this M's route metadata before the next M overwrites ma.
            routes = ma.index(((epoch - 1) & 1) * 4 + 2).load(gl.SliceLayout(1, output_store)).to(gl.int32,
                bitcast=True)
            experts = ma.index(((epoch - 1) & 1) * 4 + 3).load(gl.SliceLayout(1, output_store)).to(gl.int32,
                bitcast=True)
            hcu.wait_lgkmcnt(0)
            route_parts = gl.split(routes.reshape((2, 64)).permute((1, 0)))
            expert_parts = gl.split(experts.reshape((2, 64)).permute((1, 0)))
            output_rows = ()
            output_valid = ()
            for panel in gl.static_range(2):
                route = gl.convert_layout(route_parts[panel], gl.SliceLayout(1, output_store))
                expert = gl.convert_layout(expert_parts[panel], gl.SliceLayout(1, output_store))
                if CK_SORTING:
                    route = (route & 0x00ffffff) * TOP_K + ((route >> 24) & 255)
                if USE_C64:
                    output_rows += (route.to(gl.int64) * N,)
                else:
                    output_rows += (route.to(gl.uint32) * N,)
                output_valid += ((route >= 0) & (route < valid_routes) & (expert >= 0),)
            next_m = m + gl.num_programs(0)
            if next_m < total // (N // BN):
                # This completed current-A READY separates consecutive A EMPTY3
                # waits. Only P0 can publish the next READY1 generation.
                hcu.abarrier_try_wait(1, (epoch - 1) & 1)
                # Both readers retained A in registers; reuse its LDS slot for next M.
                _load_a_and_metadata(A, SA, Weights, Sorted, Experts, am, ma, next_m, rows, epoch,
                    K, BM, BK, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT)
                epoch += 1
            for n in range(0, N_TILES):
                col = n * BN + output_rn
                for panel in gl.static_range(2):
                    hcu.abarrier_try_wait(9, panel)
                    x = co0.load(output_store)
                    hcu.wait_lgkmcnt(0)
                    hcu.abarrier_arrive(11)
                    offsets = output_rows[panel][:, None] + col[None, :].to(gl.uint32)
                    valid = output_valid[panel]
                    if USE_C64:
                        gl.store(C + offsets, x, valid[:, None])
                    else:
                        buffer_store(x, C, offsets, valid[:, None])
                    hcu.abarrier_try_wait(13, panel)
                    y = co1.load(output_store)
                    hcu.wait_lgkmcnt(0)
                    hcu.abarrier_arrive(15)
                    if USE_C64:
                        gl.store(C + offsets + BN // 2, y, valid[:, None])
                    else:
                        buffer_store(y, C, offsets + BN // 2, valid[:, None])
        else:
            expert = gl.load(Experts + m)
            b_base = B + gl.maximum(expert, 0).to(gl.int64) * N * K
            for n in range(0, N_TILES):
                _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch, 0, 0, N, K, BN, BK,
                    USE_B64, STRIDE_SBE, STRIDE_SBN)
                _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch, 0, 1, N, K, BN, BK,
                    USE_B64, STRIDE_SBE, STRIDE_SBN)
                _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch, 1, 1, N, K, BN, BK,
                    USE_B64, STRIDE_SBE, STRIDE_SBN)
                _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch, 1, 0, N, K, BN, BK,
                    USE_B64, STRIDE_SBE, STRIDE_SBN)
                # P1 owns READY5 publication. Its current K1 generation must
                # complete before the next EMPTY6 wait; avoids adjacent ID6.
                hcu.abarrier_try_wait(5, 1)
                epoch += 1


@gluon.constexpr_function
def _icache_label(partition, slot, kind):
    """Name one consumer's real MMAC block and its warmup return point."""
    return f"__gemm2_c{partition}_mmac{slot}_{kind}"


@gluon.jit
def _warm_mmac_instructions(PARTITION: gl.constexpr, BLOCKS: gl.constexpr):
    """Visit real MMAC blocks once with EXEC=0, before reading any tile.

    Four consumer waves divide the blocks by slot % 4. Only instructions
    are fetched; no arithmetic result, LDS slot or barrier credit changes.
    """
    wave = hcu.get_wave_id() & 3
    for slot in gl.static_range(BLOCKS):
        if wave == slot % 4:
            hcu.sched_barrier()
            # ### ASM Note: Gluon has no EXEC-mask API; save/clear it for instruction-only warmup.
            saved_exec = gl.inline_asm_elementwise(
                "s_mov_b64 $0, exec\n\ts_mov_b64 exec, 0", constraints="=&s,~{exec},~{memory}",
                args=[], dtype=gl.uint64, is_pure=False, pack=1)
            # ### ASM Note: Gluon cannot jump to the real MMAC block and name its return label.
            gl.inline_asm_elementwise(
                "s_branch " + _icache_label(PARTITION, slot, "entry") + "\n" +
                _icache_label(PARTITION, slot, "resume") + ":",
                constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
            # ### ASM Note: Gluon has no EXEC-mask API; restore the saved mask after the return.
            gl.inline_asm_elementwise(
                "s_mov_b64 exec, $1", constraints="=s,s,~{exec},~{memory}", args=[saved_exec],
                dtype=gl.int32, is_pure=False, pack=1)
            hcu.sched_barrier()


@gluon.jit
def _mmac(a, b, acc, PARTITION: gl.constexpr, SLOT: gl.constexpr,
          INSN_WARMUP: gl.constexpr):
    """Compute one block; optional EXEC=0 visits return before any output work."""
    if INSN_WARMUP:
        # ### ASM Note: Gluon cannot define a branch-target label around its generated MMACs.
        gl.inline_asm_elementwise(
            _icache_label(PARTITION, SLOT, "entry") + ":", constraints="=s,~{memory}",
            args=[], dtype=gl.int32, is_pure=False, pack=1)
    acc = gl.amd.cdna4.mfma(a, b, acc)
    if INSN_WARMUP:
        # ### ASM Note: Gluon cannot branch on EXEC to the matching warmup return label.
        gl.inline_asm_elementwise(
            "s_cbranch_execz " + _icache_label(PARTITION, SLOT, "resume"),
            constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
    return acc


@gluon.jit
def _consumer(C, SA, SB, Weights, Sorted, Experts, am, bm, co, ma, mb,
              pid, total, rows, valid_routes,
              N: gl.constexpr, K: gl.constexpr, TOP_K: gl.constexpr,
              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
              STAGES: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
              CK_SORTING: gl.constexpr, PARTITION: gl.constexpr, N_TILES: gl.constexpr,
              num_tokens_post_padded_ptr, INSN_WARMUP: gl.constexpr = False):
    """C0/C1 retain A across N and publish their scaled 128-column C half.

        Before N loop: wait A -> issue full-A LDS read -> split register views
        First N:       wait B(K0) -> read B/SB -> wait LDS -> EMPTY A + B
        Later N:       wait B(K0) -> read B/SB -> wait LDS -> EMPTY B
        Every N:       MMAC K0 -> wait/read B(K1) -> wait LDS -> EMPTY B
                       -> MMAC K1 + scale/narrow -> publish paired C panels

    Metadata is read once per M after K0 MMAC. One LDS wait completes the
    first A/B/SB reads together; A EMPTY is returned exactly once per M.
    """
    if num_tokens_post_padded_ptr is not None:
        total = gl.minimum(total, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * (N // BN))
    CN: gl.constexpr = BN // 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], MMAC_LAYOUT, [2, 2], tiles_per_warp=[1, 2])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    packing: gl.constexpr = _output_pack_layout(gl.to_linear_layout(result, (32, CN)))
    first_n = 0
    if INSN_WARMUP:
        _warm_mmac_instructions(PARTITION, 6)
        # ### ASM Note: An opaque zero prevents loop peeling/unrolling from cloning named ASM labels.
        first_n = gl.inline_asm_elementwise(
            "s_mov_b32 $0, 0", constraints="=s", args=[], dtype=gl.int32, is_pure=False, pack=1)
    a_epoch = 0
    b_epoch = 0
    ACC: gl.constexpr = gl.int32 if am.dtype == gl.int8 else gl.float32
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        ticket = a_epoch
        hcu.abarrier_try_wait(1, ticket & 1)
        # Register views only; A remains resident for all N tiles of this M:
        #     A [128,256] -> low/high [64,256] -> each K0/K1 [64,128]
        a = am.index(0).load(gl.DotOperandLayout(0, mma, 8))
        pair = gl.split(a.reshape((2, BM // 2, K)).permute((1, 2, 0)))
        low_full = gl.convert_layout(pair[0], gl.DotOperandLayout(0, mma, 8))
        high_full = gl.convert_layout(pair[1], gl.DotOperandLayout(0, mma, 8))
        low_halves = gl.split(low_full.reshape((BM // 2, 2, BK)).permute((0, 2, 1)))
        high_halves = gl.split(high_full.reshape((BM // 2, 2, BK)).permute((0, 2, 1)))
        a_low = (gl.convert_layout(low_halves[0], gl.DotOperandLayout(0, mma, 8)),
            gl.convert_layout(low_halves[1], gl.DotOperandLayout(0, mma, 8)))
        a_high = (gl.convert_layout(high_halves[0], gl.DotOperandLayout(0, mma, 8)),
            gl.convert_layout(high_halves[1], gl.DotOperandLayout(0, mma, 8)))
        a_epoch += 1
        sa = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, result))
        weight = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, result))
        for n in range(first_n, N_TILES):
            lo = gl.full((BM // 2, CN), 0, ACC, result)
            hi = gl.full((BM // 2, CN), 0, ACC, result)
            for k in gl.static_range(1):
                ticket = b_epoch * 2 + k
                ready = (7 if k == 0 else 2) if PARTITION == 1 else 5
                empty = (8 if k == 0 else 4) if PARTITION == 1 else 6
                physical = k if PARTITION == 1 else 0
                phase = (b_epoch & 1) if PARTITION == 1 else (ticket & 1)
                hcu.abarrier_try_wait(ready, phase)
                if k == 0:
                    current_sb = mb.index(PARTITION).load(gl.SliceLayout(0, result))
                b = bm.index(physical).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
                hcu.wait_lgkmcnt(0)
                if n == 0:
                    # A is loaded once per M; each reader returns one credit.
                    hcu.abarrier_arrive(3)
                hcu.abarrier_arrive(empty)

                hcu.sched_barrier()
                if INSN_WARMUP:
                    lo = _mmac(a_low[k], b, lo, PARTITION, 0, INSN_WARMUP)
                else:
                    lo = gl.amd.cdna4.mfma(a_low[k], b, lo)
                if k == 0:
                    hcu.sched_barrier()
                    hcu.sched_barrier()
                    if INSN_WARMUP:
                        hi = _mmac(a_high[k], b, hi, PARTITION, 1, INSN_WARMUP)
                    else:
                        hi = gl.amd.cdna4.mfma(a_high[k], b, hi)
                hcu.sched_barrier()

                if k == 0:
                    if n == 0:
                        # Metadata belongs to M, while b_epoch also advances in N.
                        meta_epoch = a_epoch - 1
                        hcu.abarrier_try_wait(12 if (meta_epoch & 1) == 0 else 0, (meta_epoch >> 1) & 1)
                        sa = ma.index((meta_epoch & 1) * 4).load(gl.SliceLayout(1, result))
                        weight = ma.index((meta_epoch & 1) * 4 + 1).load(gl.SliceLayout(1, result))
            ready = 2 if PARTITION == 1 else 5
            empty = 4 if PARTITION == 1 else 6
            phase = (b_epoch & 1) if PARTITION == 1 else 1
            hcu.abarrier_try_wait(ready, phase)
            b = bm.index(1 if PARTITION == 1 else 0).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1,
                mma, 8)))
            hcu.wait_lgkmcnt(0)
            hcu.abarrier_arrive(empty)
            # K1 works on 32-row panels to shorten accumulator lifetimes:
            #     lo/hi [64,128] -> four acc/A/SA/weight groups of 32 rows
            #     panels 0+1 -> first C [64,128]; panels 2+3 -> second C
            low_accs = gl.split(lo.reshape((2, 32, CN)).permute((1, 2, 0)))
            high_accs = gl.split(hi.reshape((2, 32, CN)).permute((1, 2, 0)))
            accumulators = (gl.convert_layout(low_accs[0], result),
                gl.convert_layout(low_accs[1], result), gl.convert_layout(high_accs[0], result),
                gl.convert_layout(high_accs[1], result))
            low_a = gl.split(a_low[1].reshape((2, 32, BK)).permute((1, 2, 0)))
            high_a = gl.split(a_high[1].reshape((2, 32, BK)).permute((1, 2, 0)))
            a_panels = (gl.convert_layout(low_a[0], gl.DotOperandLayout(0, mma, 8)),
                gl.convert_layout(low_a[1], gl.DotOperandLayout(0, mma, 8)),
                gl.convert_layout(high_a[0], gl.DotOperandLayout(0, mma, 8)),
                gl.convert_layout(high_a[1], gl.DotOperandLayout(0, mma, 8)))
            sa_parts = _four_row_vectors(sa)
            weight_parts = _four_row_vectors(weight)
            for panel in gl.static_range(4):
                hcu.sched_barrier()
                if INSN_WARMUP:
                    acc = _mmac(a_panels[panel], b, accumulators[panel], PARTITION, 2 + panel, INSN_WARMUP)
                else:
                    acc = gl.amd.cdna4.mfma(a_panels[panel], b, accumulators[panel])
                hcu.sched_barrier()

                panel_sa = gl.convert_layout(sa_parts[panel], gl.SliceLayout(1, result))
                panel_weight = gl.convert_layout(weight_parts[panel], gl.SliceLayout(1, result))
                # Compile-time dtype branch: INT8 scales its complete K sum once.
                if acc.dtype == gl.int32:
                    acc = acc.to(gl.float32)
                value = ((acc * panel_sa[:, None]) * current_sb[None, :]) * panel_weight[:, None]
                packed = gl.convert_layout(value, packing).to(C.dtype.element_ty)
                if panel % 2 == 0:
                    if panel == 2:
                        hcu.abarrier_try_wait(9 + 4 * PARTITION, 0)
                    hcu.abarrier_try_wait(11 + 4 * PARTITION, (panel // 2) & 1)
                co.slice((panel & 1) * 32, 32).store(packed)
                if panel % 2 == 1:
                    hcu.wait_lgkmcnt(0)
                    hcu.abarrier_arrive(9 + 4 * PARTITION)
            b_epoch += 1


@gluon.jit
def gemm_bottom_w8a8_channel_wasp_persistent(
    A, B, SA, SB, C, Weights, Sorted, Experts,
    num_sorted_rows, num_valid_routes, TOTAL_TILES,
    N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 128, BLOCK_SIZE_N: gl.constexpr = 256,
    BLOCK_SIZE_K: gl.constexpr = 128, GROUP_SIZE_M: gl.constexpr = 1,
    CK_SORTING: gl.constexpr = True, MUL_ROUTED_WEIGHT: gl.constexpr = True,
    PARTITION_REGS: gl.constexpr = None, INSN_WARMUP: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_A: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_B: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr = False,
    stride_asm: gl.constexpr = 1, stride_bse: gl.constexpr = None,
    stride_bsn: gl.constexpr = 1, N_TILES_LIMIT: gl.constexpr = 0,
    num_tokens_post_padded_ptr=None,
):
    """Allocate shared buffers and launch P0/P1/C0/C1 with caller-supplied grid.

    A is already CK-sorted. Weights holds SortedWeights when multiplication
    is enabled; otherwise it may be None. C is route-major, before top-k sum.
    TOTAL_TILES counts M blocks * (N/BN); the device padded-row count may cap
    it. N_TILES_LIMIT only clips N for diagnostics and defaults to full N.
    """
    gl.static_assert(BLOCK_SIZE_M == 128 and BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128)
    gl.static_assert(K == 256 and N % 256 == 0 and N > 0)
    STRIDE_SA: gl.constexpr = stride_asm
    STRIDE_SBE: gl.constexpr = N if stride_bse is None else stride_bse
    STRIDE_SBN: gl.constexpr = stride_bsn
    N_TILES: gl.constexpr = N // 256 if N_TILES_LIMIT == 0 else N_TILES_LIMIT
    gl.static_assert(N_TILES > 0 and N_TILES <= N // 256)
    USE_MLS_LOAD_A: gl.constexpr = True
    regs: gl.constexpr = (48, 16, 224, 224) if PARTITION_REGS is None else PARTITION_REGS
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    STAGES: gl.constexpr = 1
    gl.static_assert(GROUP_SIZE_M == 1, "A reuse uses row-persistent ownership")
    al: gl.constexpr = hcu.MlsSharedLayout(0, (16, 64), order=(1, 0))
    bl: gl.constexpr = hcu.MlsSharedLayout(1, (64, 32), alt=1, order=(0, 1))
    # A [1,128,256] 32 KiB; B-left [1,128,128] 16 KiB;
    # B-right [2,128,128] 32 KiB. All transports preserve input bytes.
    am = gl.allocate_shared_memory(A.dtype.element_ty, [STAGES, BM, K], al)
    b0 = gl.allocate_shared_memory(B.dtype.element_ty, [STAGES, BK, BN // 2], bl)
    b1 = gl.allocate_shared_memory(B.dtype.element_ty, [2, BK, BN // 2], bl)
    # C0/C1 each reuse one [64,128] 16-bit output buffer (16 KiB each).
    output_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 4, [1, 0])
    co0 = gl.allocate_shared_memory(C.dtype.element_ty, [64, BN // 2], output_shared)
    co1 = gl.allocate_shared_memory(C.dtype.element_ty, [64, BN // 2], output_shared)
    meta_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    ma_layout: gl.constexpr = gl.SharedLinearLayout([[4], [8], [1], [2], [16], [32], [64]], [], 16)
    # MA: two M slots x four [128] FP32 vectors = 4 KiB.
    # MB: left/right [128] FP32 channel scales = 1 KiB.
    ma = gl.allocate_shared_memory(gl.float32, [8, BM], ma_layout)
    mb = gl.allocate_shared_memory(gl.float32, [2, BN // 2], meta_shared)
    _initialize(STAGES)
    pid = gl.program_id(0)
    gl.warp_specialize([
        (_empty, ()),
        (_producer, (
            A, B, Experts, am, b0, b1, C, Sorted, num_valid_routes, co0, co1,
            SA, SB, Weights, ma, mb, pid, TOTAL_TILES, num_sorted_rows,
            N, K, BM, BN, BK, STAGES, USE_MLS_LOAD_A, GROUP_SIZE_M, 0, top_k,
            CK_SORTING, N_TILES, USE_ADDR_OFFSET_INT64_A, USE_ADDR_OFFSET_INT64_B,
            USE_ADDR_OFFSET_INT64_C, STRIDE_SA, STRIDE_SBE, STRIDE_SBN,
            MUL_ROUTED_WEIGHT, num_tokens_post_padded_ptr)),
        (_producer, (
            A, B, Experts, am, b0, b1, C, Sorted, num_valid_routes, co0, co1,
            SA, SB, Weights, ma, mb, pid, TOTAL_TILES, num_sorted_rows,
            N, K, BM, BN, BK, STAGES, USE_MLS_LOAD_A, GROUP_SIZE_M, 1, top_k,
            CK_SORTING, N_TILES, USE_ADDR_OFFSET_INT64_A, USE_ADDR_OFFSET_INT64_B,
            USE_ADDR_OFFSET_INT64_C, STRIDE_SA, STRIDE_SBE, STRIDE_SBN,
            MUL_ROUTED_WEIGHT, num_tokens_post_padded_ptr)),
        (_consumer, (
            C, SA, SB, Weights, Sorted, Experts, am, b0, co0, ma, mb, pid, TOTAL_TILES,
            num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
            GROUP_SIZE_M, CK_SORTING, 0, N_TILES, num_tokens_post_padded_ptr, INSN_WARMUP)),
        (_consumer, (
            C, SA, SB, Weights, Sorted, Experts, am, b1, co1, ma, mb, pid, TOTAL_TILES,
            num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
            GROUP_SIZE_M, CK_SORTING, 1, N_TILES, num_tokens_post_padded_ptr, INSN_WARMUP)),
    ], [4, 4, 4, 4], regs)
    for barrier in gl.static_range(16):
        hcu.abarrier_invalidate(barrier)


def is_persistent_channel_gemm2_supported(a, b, a_scale, b_scale, *, arch, cu_count,
                                        top_k, config, c_sorted=False, a_sorted=True,
                                        ck_sorting=True, output_dtype=None,
                                        scale_recipe="fp8_channel", has_bias=False,
                                        mul_routed_weight=True):
    """CPU metadata query for the BN256/K256 pipeline; otherwise use Triton.

    Only BM is part of the sorting contract. This candidate selects its own
    BN256/BK128, launch options and row-persistent traversal. The caller checks
    tensor devices, CK metadata and sorted routed weights before dispatch.
    """
    import torch
    if arch != "gfx946" or type(cu_count) is not int or cu_count <= 0:
        return False
    if c_sorted or not a_sorted or not ck_sorting or has_bias:
        return False
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    inputs = ((a.dtype in fp8 and b.dtype in fp8 and scale_recipe == "fp8_channel")
              or (a.dtype == b.dtype == torch.int8 and scale_recipe == "int8_channel"))
    if not inputs or a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        return False
    if output_dtype is not None and output_dtype not in (torch.bfloat16, torch.float16):
        return False
    if a.ndim != 2 or b.ndim != 3 or not a.is_contiguous() or not b.is_contiguous():
        return False
    rows, k = a.shape
    experts, n, bk = b.shape
    if min(rows, experts, n) <= 0 or k != 256 or bk != k or n % 256 or not 0 < top_k <= 256:
        return False
    if a_scale.shape not in ((rows,), (rows, 1)) or b_scale.shape not in ((experts, n), (experts, n, 1)):
        return False
    return config.get("BLOCK_SIZE_M") == 128
