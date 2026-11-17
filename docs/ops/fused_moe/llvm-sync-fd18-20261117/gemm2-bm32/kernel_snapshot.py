# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent FP8/INT8 channel GEMM2 for gfx946, BM32/BN512/BK64.

Contiguous FP8 (E4M3FN/E5M2, including mixed formats) or signed INT8 A/B;
FP32 channel scales; BF16/FP16 scatter output, before the top-k reduction.
Requires CK-sorted A/routes, K256 and N divisible by 512. Four waves per role.

    P0: A + SA/weight/routes -> C0/C1; C0/C1 output mailboxes -> scatter C
    P1: full-K B-left + SB-left -> C0; full-K B-right + SB-right -> C1
    C0: left 256 columns; C1: right 256 columns. Both retain A across N.

    P0 startup: load A(M0) + metadata(M0).
    P0 each M: wait A EMPTY -> preload next A; keep next metadata in registers
               -> drain first N64 phase -> publish next metadata -> drain rest.
    P1 each phase: prefetch both SB vectors -> B-left/SB-left -> B-right/SB-right.
    C0/C1: read A once -> each N64 phase read B/SB -> release A/B after LDS reads
           -> four BK64 MMACs -> scale once -> publish M32xN64 output mailbox.
    P0: wait output READY -> read mailbox -> EMPTY -> global scatter.

One full-A slot: 8 KiB; two full-K B slots: 32 KiB; two output mailboxes:
8 KiB; metadata/scales: 1.5 KiB. Total 49.5 KiB LDS. Twelve live abarriers
use IDs 0..15. M += caller grid; each BN512 tile has four N64 phases per
consumer. INT8 accumulates in INT32 through K and converts only for scaling.
Instruction warmup is disabled; True is rejected explicitly. N clipping is diagnostic only.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

DEFAULT_LAUNCH_OPTIONS = dict(
    num_warps=4,
    num_stages=2,
    use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4),
    wasp_wdra=True,
)
MMAC_LAYOUT = gl.constexpr(3)
ROW_PERSISTENT_A = True


@gluon.constexpr_function
def _output_pack_layout(layout):
    """Pair adjacent output halfwords before the LDS mailbox handoff."""
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:], layout.lane_bases, layout.warp_bases, layout.block_bases, layout.shape
    )


@gluon.jit
def _empty():
    """Keep the default WASP partition empty; P0/P1/C0/C1 own all work."""
    pass


@gluon.jit
def _initialize(STAGES: gl.constexpr):
    # A READY1/EMPTY3; full-K BL READY5/EMPTY6; BR READY7/EMPTY8.
    """Initialize READY/EMPTY generations before splitting the four roles.

        Buffer          Producer -> Reader     READY   EMPTY   Phase
        A full K256      P0 -> C0 + C1             1       3    M epoch & 1
        B-left full K     P1 -> C0                  5       6    N epoch & 1
        B-right full K    P1 -> C1                  7       8    N epoch & 1
        C-left/right     C0/C1 -> P0            9/13   11/15    N epoch & 1
        Metadata         P0 -> C0 + C1         12/0       -    (M epoch >> 1) & 1

        writer: wait EMPTY -> load/store -> READY ---> reader: wait READY -> read
                         ^                                               |
                         +---------------- EMPTY <-----------------------+

    A needs four credits from each consumer; B and C need four credits.
    READY counts four writer waves. Metadata has two M slots; draining this
    M's outputs bounds its reuse. Only live IDs are initialized; no ID exceeds 15.
    """
    hcu.abarrier_init(1, 4)
    hcu.abarrier_init(3, 8)
    for i in gl.static_range(2, 16):
        if i != 3 and i != 2 and i != 4 and i != 10 and i != 14:
            hcu.abarrier_init(i, 4)
    hcu.abarrier_init(0, 4)
    gl.thread_barrier()
    hcu.abarrier_arrive(3, 2)
    hcu.abarrier_arrive(6)
    hcu.abarrier_arrive(8)
    hcu.abarrier_arrive(11)
    hcu.abarrier_arrive(15)
    # IDs 2/4/10/14 are unused: each B half fits in one full-K slot.
    gl.thread_barrier()


@gluon.jit
def _load_a_and_metadata(
    A,
    SA,
    Weights,
    Sorted,
    Experts,
    am,
    ma,
    m,
    rows,
    epoch,
    K: gl.constexpr,
    BM: gl.constexpr,
    BK: gl.constexpr,
    USE_A64: gl.constexpr,
    STRIDE_SA: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    WITH_METADATA: gl.constexpr = True,
):
    """P0 fills the single full-K A slot; optionally publish M metadata too.

    A slot0: READY1 / EMPTY3, phase = epoch & 1.
    Metadata: two slots of four [BM] vectors; READY12/0 by epoch parity,
    phase = (epoch >> 1) & 1. Consumers both return A EMPTY after their reads.

    WITH_METADATA=False prefetches next A only. P0 separately holds next-M
    metadata in registers, publishing it after the current first N phase."""
    ticket = epoch
    slot = 0
    phase = ticket & 1
    hcu.set_priority(2)
    hcu.abarrier_try_wait(3 + slot, phase)
    hcu.abarrier_seq(1 + slot)
    if USE_A64:
        # Fold the large row origin into the 64-bit descriptor base.
        a_base = A + m.to(gl.int64) * BM * K
        hcu.matrix_load(am.index(0), a_base, (BM, K), (K, 1), (0, 0), warps=(2, 2))
    else:
        hcu.matrix_load(am.index(0), A, (rows, K), (K, 1), (m * BM, 0), warps=(2, 2))
    hcu.abarrier_arrive(1 + slot)
    hcu.set_priority(0)
    hcu.sched_barrier()
    if WITH_METADATA:
        _load_metadata_only(SA, Weights, Sorted, Experts, ma, m, epoch, BM, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT)


@gluon.jit
def _load_metadata_only(
    SA,
    Weights,
    Sorted,
    Experts,
    ma,
    m,
    epoch,
    BM: gl.constexpr,
    USE_A64: gl.constexpr,
    STRIDE_SA: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
):
    """Read and publish SA/weight/route/expert for one M; leave the A slot alone."""
    sa, weight, route, expert = _read_metadata(
        SA, Weights, Sorted, Experts, m, BM, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT
    )
    _publish_metadata(ma, epoch, sa, weight, route, expert)


@gluon.jit
def _read_metadata(
    SA,
    Weights,
    Sorted,
    Experts,
    m,
    BM: gl.constexpr,
    USE_A64: gl.constexpr,
    STRIDE_SA: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
):
    """Prefetch this M's four row vectors into P0 registers; do not signal READY.

    SA/weight are FP32; route/expert retain their integer bits. P0 can hold
    next-M values while current-M readers still own their metadata slot."""
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
    return sa, weight, route, expert


@gluon.jit
def _publish_metadata(ma, epoch, sa, weight, route, expert):
    """Publish four [BM] vectors into the alternating M-metadata slot.

    slot = epoch & 1; READY = 12 (even) or 0 (odd)
    phase = (epoch >> 1) & 1, tracked by consumers
    SA/weight/route/expert stores -> lgkmcnt(0) -> READY

    The caller delays next-M publication until the first N phase drains,
    so current-M metadata reads have finished before that storage is reused."""
    ma.index((epoch & 1) * 4 + 0).store(sa)
    ma.index((epoch & 1) * 4 + 1).store(weight)
    ma.index((epoch & 1) * 4 + 2).store(route.to(gl.float32, bitcast=True))
    ma.index((epoch & 1) * 4 + 3).store(expert.to(gl.float32, bitcast=True))
    hcu.wait_lgkmcnt(0)
    hcu.abarrier_arrive(12 if (epoch & 1) == 0 else 0)


@gluon.jit
def _load_b_panel(
    B,
    SB,
    b_base,
    expert,
    b0,
    b1,
    mb,
    n,
    epoch,
    KH: gl.constexpr,
    HALF: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    USE_B64: gl.constexpr,
    sb,
    STRIDE_SBE: gl.constexpr,
    STRIDE_SBN: gl.constexpr,
):
    """P1 publishes one full-K256/N64 B panel and its channel scales.

    KH is zero; HALF selects C0/C1. Four phases cover each BN512 output tile.
    Each B slot is reused only after its reader returns the matching EMPTY generation."""
    ready = (7 if KH % 2 == 0 else 2) if HALF == 1 else (5 if KH % 2 == 0 else 10)
    empty = (8 if KH % 2 == 0 else 4) if HALF == 1 else (6 if KH % 2 == 0 else 14)
    # Each full-K slot is used once per N64 phase.
    phase = epoch & 1
    physical = 0
    dest = b1 if HALF == 1 else b0
    col = (n // 4) * BN + (n % 4) * (BN // 8) + HALF * (BN // 2)
    hcu.abarrier_try_wait(empty, phase)
    hcu.abarrier_seq(ready)
    if USE_B64:
        panel_base = b_base + col.to(gl.int64) * K + KH * (2 * BK)
        hcu.matrix_load(dest.index(physical), panel_base, (K, BN // 8), (1, K), (0, 0), warps=(2, 2))
    else:
        hcu.matrix_load(dest.index(physical), b_base, (K, N), (1, K), (KH * (2 * BK), col), warps=(2, 2))
    if KH == 0:
        mb.index(HALF).store(sb)
        hcu.wait_lgkmcnt(0)
    hcu.abarrier_arrive(ready)


@gluon.jit
def _producer(
    A,
    B,
    Experts,
    am,
    b0,
    b1,
    C,
    Sorted,
    valid_routes,
    co0,
    co1,
    SA,
    SB,
    Weights,
    ma,
    mb,
    pid,
    total,
    rows,
    N: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    STAGES: gl.constexpr,
    USE_MLS_LOAD_A: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    PARTITION: gl.constexpr,
    TOP_K: gl.constexpr,
    CK_SORTING: gl.constexpr,
    N_PANELS: gl.constexpr,
    USE_A64: gl.constexpr,
    USE_B64: gl.constexpr,
    USE_C64: gl.constexpr,
    STRIDE_SA: gl.constexpr,
    STRIDE_SBE: gl.constexpr,
    STRIDE_SBN: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    num_tokens_post_padded_ptr,
):
    """P0 prefetches next-M A/metadata and scatters C; P1 streams both B halves.

    P0: current routes -> next-M A and metadata prefetch -> drain outputs;
    publish next-M metadata after the first N phase drains.
    P1: full-K BL -> full-K BR -> next N64 phase.
    P0 epochs count M preloads; P1 epochs count N64 phases across all M.
    """
    if num_tokens_post_padded_ptr is not None:
        total = gl.minimum(total, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * (N // BN))
    epoch = 0
    ml: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    mr = gl.arange(0, BN // 8, ml)
    output_store: gl.constexpr = gl.BlockedLayout([2, 8], [4, 16], [4, 1], [1, 0])
    output_rm = gl.arange(0, BM, gl.SliceLayout(1, output_store))
    output_rn = gl.arange(0, BN // 8, gl.SliceLayout(0, output_store))
    # Carry the next M expert across the loop; only the first M loads on demand.
    # Guard the lookahead by the same padded-M bound as the persistent traversal.
    prefetched_expert = pid
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        if PARTITION == 0:
            if m == pid:
                _load_a_and_metadata(
                    A,
                    SA,
                    Weights,
                    Sorted,
                    Experts,
                    am,
                    ma,
                    m,
                    rows,
                    epoch,
                    K,
                    BM,
                    BK,
                    USE_A64,
                    STRIDE_SA,
                    MUL_ROUTED_WEIGHT,
                )
                epoch += 1
            # Capture this M's route metadata before the next M overwrites ma.
            routes = (
                ma.index(((epoch - 1) & 1) * 4 + 2).load(gl.SliceLayout(1, output_store)).to(gl.int32, bitcast=True)
            )
            experts = (
                ma.index(((epoch - 1) & 1) * 4 + 3).load(gl.SliceLayout(1, output_store)).to(gl.int32, bitcast=True)
            )
            hcu.wait_lgkmcnt(0)
            route = routes
            if CK_SORTING:
                route = (route & 0x00FFFFFF) * TOP_K + ((route >> 24) & 255)
            if USE_C64:
                output_rows = route.to(gl.int64) * N
            else:
                output_rows = route.to(gl.uint32) * N
            output_valid = (route >= 0) & (route < valid_routes) & (experts >= 0)
            next_sa = gl.full((BM,), 0, gl.float32, ml)
            next_weight = gl.full((BM,), 0, gl.float32, ml)
            next_route = gl.full((BM,), 0, gl.int32, ml)
            next_expert = gl.full((BM,), 0, gl.int32, ml)
            next_m = m + gl.num_programs(0)
            if next_m < total // (N // BN):
                # This completed current-A READY separates consecutive A EMPTY3
                # waits. Only P0 can publish the next READY1 generation.
                hcu.abarrier_try_wait(1, (epoch - 1) & 1)
                # Independent full-A slot preserves the original next-M prefetch.
                _load_a_and_metadata(
                    A,
                    SA,
                    Weights,
                    Sorted,
                    Experts,
                    am,
                    ma,
                    next_m,
                    rows,
                    epoch,
                    K,
                    BM,
                    BK,
                    USE_A64,
                    STRIDE_SA,
                    MUL_ROUTED_WEIGHT,
                    False,
                )
                next_sa, next_weight, next_route, next_expert = _read_metadata(
                    SA, Weights, Sorted, Experts, next_m, BM, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT
                )
                epoch += 1
            for n in range(0, N_PANELS):
                col = (n // 4) * BN + (n % 4) * (BN // 8) + output_rn
                hcu.abarrier_try_wait(9, n & 1)
                x = co0.load(output_store)
                hcu.wait_lgkmcnt(0)
                hcu.abarrier_arrive(11)
                offsets = output_rows[:, None] + col[None, :].to(gl.uint32)
                valid = output_valid
                if USE_C64:
                    gl.store(C + offsets, x, valid[:, None])
                else:
                    buffer_store(x, C, offsets, valid[:, None])
                hcu.abarrier_try_wait(13, n & 1)
                y = co1.load(output_store)
                hcu.wait_lgkmcnt(0)
                hcu.abarrier_arrive(15)
                if USE_C64:
                    gl.store(C + offsets + BN // 2, y, valid[:, None])
                else:
                    buffer_store(y, C, offsets + BN // 2, valid[:, None])
                if n == 0:
                    if next_m < total // (N // BN):
                        _publish_metadata(ma, epoch - 1, next_sa, next_weight, next_route, next_expert)
        else:
            if m == pid:
                expert = gl.load(Experts + m)
            else:
                expert = prefetched_expert
            prefetched_expert = expert
            if m + gl.num_programs(0) < total // (N // BN):
                prefetched_expert = gl.load(Experts + m + gl.num_programs(0))
            b_base = B + gl.maximum(expert, 0).to(gl.int64) * N * K
            for n in range(0, N_PANELS):
                scale_col = (n // 4) * BN + (n % 4) * (BN // 8) + mr
                if USE_B64:
                    scale_base = SB + gl.maximum(expert, 0).to(gl.int64) * STRIDE_SBE
                    sb_left = gl.load(scale_base + scale_col.to(gl.int64) * STRIDE_SBN)
                    sb_right = gl.load(scale_base + (scale_col.to(gl.int64) + BN // 2) * STRIDE_SBN)
                else:
                    scale_origin = gl.maximum(expert, 0) * STRIDE_SBE
                    sb_left = buffer_load(SB, (scale_origin + scale_col * STRIDE_SBN).to(gl.uint32))
                    sb_right = buffer_load(SB, (scale_origin + (scale_col + BN // 2) * STRIDE_SBN).to(gl.uint32))
                hcu.sched_barrier()
                _load_b_panel(
                    B,
                    SB,
                    b_base,
                    expert,
                    b0,
                    b1,
                    mb,
                    n,
                    epoch,
                    0,
                    0,
                    N,
                    K,
                    BN,
                    BK,
                    USE_B64,
                    sb_left,
                    STRIDE_SBE,
                    STRIDE_SBN,
                )
                _load_b_panel(
                    B,
                    SB,
                    b_base,
                    expert,
                    b0,
                    b1,
                    mb,
                    n,
                    epoch,
                    0,
                    1,
                    N,
                    K,
                    BN,
                    BK,
                    USE_B64,
                    sb_right,
                    STRIDE_SBE,
                    STRIDE_SBN,
                )
                epoch += 1


@gluon.jit
def _consumer(
    C,
    SA,
    SB,
    Weights,
    Sorted,
    Experts,
    am,
    bm,
    co,
    ma,
    mb,
    pid,
    total,
    rows,
    valid_routes,
    N: gl.constexpr,
    K: gl.constexpr,
    TOP_K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    STAGES: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    CK_SORTING: gl.constexpr,
    PARTITION: gl.constexpr,
    N_PANELS: gl.constexpr,
    A_IS_INT8: gl.constexpr,
    num_tokens_post_padded_ptr,
    INSN_WARMUP: gl.constexpr = False,
):
    """Retain full M32/K256 A, consume full-K N64 B panels, and publish M32 output.

    Four phases cover each BN512 tile; A and metadata remain resident across N.
    All K contributions finish before strict ((acc*SA)*SB)*weight scaling.
    Only P0 performs global scatter; C0/C1 return fully read B slots before MMAC."""
    if num_tokens_post_padded_ptr is not None:
        total = gl.minimum(total, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * (N // BN))
    CN: gl.constexpr = BN // 8
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], MMAC_LAYOUT, [2, 2], tiles_per_warp=[1, 2])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    packing: gl.constexpr = _output_pack_layout(gl.to_linear_layout(result, (BM, CN)))
    gl.static_assert(not INSN_WARMUP, "BM32 instruction warmup is not supported")
    a_epoch, b_epoch = 0, 0
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        hcu.abarrier_try_wait(1, a_epoch & 1)
        a = am.index(0).load(gl.DotOperandLayout(0, mma, 8))
        ap = gl.split(a.reshape((BM, 2, K // 2)).permute((0, 2, 1)))
        aq = ()
        for half in gl.static_range(2):
            a2 = gl.split(ap[half].reshape((BM, 2, BK)).permute((0, 2, 1)))
            aq += (
                gl.convert_layout(a2[0], gl.DotOperandLayout(0, mma, 8)),
                gl.convert_layout(a2[1], gl.DotOperandLayout(0, mma, 8)),
            )
        a_epoch += 1
        sa = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, result))
        weight = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, result))
        for n in range(N_PANELS):
            ACC_TYPE: gl.constexpr = gl.int32 if A_IS_INT8 else gl.float32
            acc = gl.full((BM, CN), 0, ACC_TYPE, result)
            hcu.abarrier_try_wait(7 if PARTITION == 1 else 5, b_epoch & 1)
            current_sb = mb.index(PARTITION).load(gl.SliceLayout(0, result))
            b_all = bm.index(0).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
            hcu.wait_lgkmcnt(0)
            if n == 0:
                hcu.abarrier_arrive(3)
            hcu.abarrier_arrive(8 if PARTITION == 1 else 6)
            bp = gl.split(b_all.reshape((2, K // 2, CN)).permute((1, 2, 0)))
            bq = ()
            for half in gl.static_range(2):
                b2 = gl.split(bp[half].reshape((2, BK, CN)).permute((1, 2, 0)))
                bq += (
                    gl.convert_layout(b2[0], hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8))),
                    gl.convert_layout(b2[1], hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8))),
                )
            hcu.sched_barrier()
            for k_part in gl.static_range(K // BK):
                acc = gl.amd.cdna4.mfma(aq[k_part], bq[k_part], acc)
                hcu.sched_barrier()
            if n == 0:
                hcu.abarrier_try_wait(12 if ((a_epoch - 1) & 1) == 0 else 0, ((a_epoch - 1) >> 1) & 1)
                sa = ma.index(((a_epoch - 1) & 1) * 4).load(gl.SliceLayout(1, result))
                weight = ma.index(((a_epoch - 1) & 1) * 4 + 1).load(gl.SliceLayout(1, result))
            hcu.abarrier_try_wait(11 + 4 * PARTITION, b_epoch & 1)
            if acc.dtype == gl.int32:
                acc = acc.to(gl.float32)
            value = ((acc * sa[:, None]) * current_sb[None, :]) * weight[:, None]
            packed = gl.convert_layout(value, packing).to(C.dtype.element_ty)
            co.store(packed)
            hcu.wait_lgkmcnt(0)
            hcu.abarrier_arrive(9 + 4 * PARTITION)
            b_epoch += 1


@gluon.jit
def gemm_bottom_w8a8_channel_wasp_persistent(
    A,
    B,
    SA,
    SB,
    C,
    Weights,
    Sorted,
    Experts,
    num_sorted_rows,
    num_valid_routes,
    TOTAL_TILES,
    N: gl.constexpr,
    K: gl.constexpr,
    top_k: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 32,
    BLOCK_SIZE_N: gl.constexpr = 512,
    BLOCK_SIZE_K: gl.constexpr = 64,
    GROUP_SIZE_M: gl.constexpr = 1,
    CK_SORTING: gl.constexpr = True,
    MUL_ROUTED_WEIGHT: gl.constexpr = True,
    PARTITION_REGS: gl.constexpr = None,
    INSN_WARMUP: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_A: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_B: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr = False,
    stride_asm: gl.constexpr = 1,
    stride_bse: gl.constexpr = None,
    stride_bsn: gl.constexpr = 1,
    N_TILES_LIMIT: gl.constexpr = 0,
    num_tokens_post_padded_ptr=None,
):
    """Launch four roles with caller grid; TOTAL_TILES = M blocks * (N/BN).

    A/SA and optional Weights are sorted. C is route-major before top-k sum.
    Device padded-row count caps M work. N_TILES_LIMIT clips BN512 tiles
    for equal-work diagnostics only; zero traverses every output N tile.
    A/B/C each have an independent compile-time wide-address option.
    """
    gl.static_assert(K == 256 and BLOCK_SIZE_M == 32 and BLOCK_SIZE_N == 512 and BLOCK_SIZE_K == 64)
    USE_MLS_LOAD_A: gl.constexpr = True
    STRIDE_SA: gl.constexpr = stride_asm
    STRIDE_SBE: gl.constexpr = N if stride_bse is None else stride_bse
    STRIDE_SBN: gl.constexpr = stride_bsn
    N_PANELS: gl.constexpr = N // 128 if N_TILES_LIMIT == 0 else 4 * N_TILES_LIMIT
    gl.static_assert(N_PANELS > 0 and N_PANELS <= N // 128)
    regs: gl.constexpr = (48, 16, 224, 224) if PARTITION_REGS is None else PARTITION_REGS
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    STAGES: gl.constexpr = 1
    gl.static_assert(BM == 32 and BN == 512 and BK == 64)
    gl.static_assert(K == 256 and BK == 64)
    gl.static_assert(K > 0 and K % BK == 0 and N % BN == 0)
    # The full A preload precedes EMPTY returns; it must fit resident slots.
    gl.static_assert(STAGES == 1, "A halves are read and credited before overwrite")
    gl.static_assert(GROUP_SIZE_M == 1, "A reuse uses row-persistent ownership")
    al: gl.constexpr = hcu.MlsSharedLayout(0, (16, 64), order=(1, 0))
    bl: gl.constexpr = hcu.MlsSharedLayout(1, (128, 32), alt=1, order=(0, 1))
    am = gl.allocate_shared_memory(A.dtype.element_ty, [STAGES, BM, K], al)
    b0 = gl.allocate_shared_memory(B.dtype.element_ty, [1, K, BN // 8], bl)
    b1 = gl.allocate_shared_memory(B.dtype.element_ty, [1, K, BN // 8], bl)
    output_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    co0 = gl.allocate_shared_memory(C.dtype.element_ty, [BM, BN // 8], output_shared)
    co1 = gl.allocate_shared_memory(C.dtype.element_ty, [BM, BN // 8], output_shared)
    meta_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    ma_layout: gl.constexpr = gl.SharedLinearLayout([[4], [8], [1], [2], [16]], [], 16)
    ma = gl.allocate_shared_memory(gl.float32, [8, BM], ma_layout)
    mb = gl.allocate_shared_memory(gl.float32, [2, BN // 8], meta_shared)
    _initialize(STAGES)
    pid = gl.program_id(0)
    gl.warp_specialize(
        [
            (_empty, ()),
            (
                _producer,
                (
                    A,
                    B,
                    Experts,
                    am,
                    b0,
                    b1,
                    C,
                    Sorted,
                    num_valid_routes,
                    co0,
                    co1,
                    SA,
                    SB,
                    Weights,
                    ma,
                    mb,
                    pid,
                    TOTAL_TILES,
                    num_sorted_rows,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                    STAGES,
                    USE_MLS_LOAD_A,
                    GROUP_SIZE_M,
                    0,
                    top_k,
                    CK_SORTING,
                    N_PANELS,
                    USE_ADDR_OFFSET_INT64_A,
                    USE_ADDR_OFFSET_INT64_B,
                    USE_ADDR_OFFSET_INT64_C,
                    STRIDE_SA,
                    STRIDE_SBE,
                    STRIDE_SBN,
                    MUL_ROUTED_WEIGHT,
                    num_tokens_post_padded_ptr,
                ),
            ),
            (
                _producer,
                (
                    A,
                    B,
                    Experts,
                    am,
                    b0,
                    b1,
                    C,
                    Sorted,
                    num_valid_routes,
                    co0,
                    co1,
                    SA,
                    SB,
                    Weights,
                    ma,
                    mb,
                    pid,
                    TOTAL_TILES,
                    num_sorted_rows,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                    STAGES,
                    USE_MLS_LOAD_A,
                    GROUP_SIZE_M,
                    1,
                    top_k,
                    CK_SORTING,
                    N_PANELS,
                    USE_ADDR_OFFSET_INT64_A,
                    USE_ADDR_OFFSET_INT64_B,
                    USE_ADDR_OFFSET_INT64_C,
                    STRIDE_SA,
                    STRIDE_SBE,
                    STRIDE_SBN,
                    MUL_ROUTED_WEIGHT,
                    num_tokens_post_padded_ptr,
                ),
            ),
            (
                _consumer,
                (
                    C,
                    SA,
                    SB,
                    Weights,
                    Sorted,
                    Experts,
                    am,
                    b0,
                    co0,
                    ma,
                    mb,
                    pid,
                    TOTAL_TILES,
                    num_sorted_rows,
                    num_valid_routes,
                    N,
                    K,
                    top_k,
                    BM,
                    BN,
                    BK,
                    STAGES,
                    GROUP_SIZE_M,
                    CK_SORTING,
                    0,
                    N_PANELS,
                    A.dtype.element_ty == gl.int8,
                    num_tokens_post_padded_ptr,
                    INSN_WARMUP,
                ),
            ),
            (
                _consumer,
                (
                    C,
                    SA,
                    SB,
                    Weights,
                    Sorted,
                    Experts,
                    am,
                    b1,
                    co1,
                    ma,
                    mb,
                    pid,
                    TOTAL_TILES,
                    num_sorted_rows,
                    num_valid_routes,
                    N,
                    K,
                    top_k,
                    BM,
                    BN,
                    BK,
                    STAGES,
                    GROUP_SIZE_M,
                    CK_SORTING,
                    1,
                    N_PANELS,
                    A.dtype.element_ty == gl.int8,
                    num_tokens_post_padded_ptr,
                    INSN_WARMUP,
                ),
            ),
        ],
        [4, 4, 4, 4],
        regs,
    )
    for barrier in gl.static_range(16):
        if barrier != 2 and barrier != 4 and barrier != 10 and barrier != 14:
            hcu.abarrier_invalidate(barrier)


def is_persistent_channel_gemm2_supported(
    a,
    b,
    a_scale,
    b_scale,
    *,
    arch,
    cu_count,
    top_k,
    config,
    c_sorted=False,
    a_sorted=True,
    ck_sorting=True,
    output_dtype=None,
    scale_recipe="fp8_channel",
    has_bias=False,
    mul_routed_weight=True,
):
    """CPU metadata query for the BN512/K256 pipeline; otherwise use Triton.

    Only BM is part of the sorting contract. This candidate selects its own
    BN512/BK64, launch options and row-persistent traversal. The caller checks
    tensor devices, CK metadata and sorted routed weights before dispatch.
    """
    import torch

    if arch != "gfx946" or type(cu_count) is not int or cu_count <= 0:
        return False
    if c_sorted or not a_sorted or not ck_sorting or has_bias:
        return False
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    inputs = (a.dtype in fp8 and b.dtype in fp8 and scale_recipe == "fp8_channel") or (
        a.dtype == b.dtype == torch.int8 and scale_recipe == "int8_channel"
    )
    if not inputs or a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        return False
    if output_dtype is not None and output_dtype not in (torch.bfloat16, torch.float16):
        return False
    if a.ndim != 2 or b.ndim != 3 or not a.is_contiguous() or not b.is_contiguous():
        return False
    rows, k = a.shape
    experts, n, bk = b.shape
    if min(rows, experts, n) <= 0 or k != 256 or bk != k or n % 512 or not 0 < top_k <= 256:
        return False
    if a_scale.shape not in ((rows,), (rows, 1)) or b_scale.shape not in ((experts, n), (experts, n, 1)):
        return False
    return config.get("BLOCK_SIZE_M") == 32 and config.get("GROUP_SIZE_M", 1) == 1
