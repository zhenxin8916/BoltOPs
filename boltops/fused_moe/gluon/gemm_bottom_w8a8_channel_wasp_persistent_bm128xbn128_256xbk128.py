# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Short-K GEMM2: sorted A, channel scales, routed weight, unsorted C.

P0 copies A once per M block, P1 copies Bleft/Bright for each N block;
C0/C1 retain the full short-K A in registers across N. A and B credits
advance independently. A requires both consumers to return EMPTY.
BK128 uses two resident A/B slots; consumers scatter output directly.
The framework uses the dedicated BN256 production kernel; this module retains
the BN128/256 direct-scatter path for the standalone probe.
Output is [M * top_k, N]; top-k reduction remains a separate operation.
The compile-time short K is fully unrolled; persistent output traversal stays
dynamic so generated code does not grow with the number of routed tiles.
TOTAL_TILES must cover complete M blocks (all N tiles).
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from .gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import _buffer_lds_layout

DEFAULT_LAUNCH_OPTIONS = dict(
    num_warps=4, num_stages=2, use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4), wasp_wdra=True,
)
MMAC_LAYOUT = gl.constexpr(3)
ROW_PERSISTENT_A = True


@gluon.constexpr_function
def _output_pack_layout(layout):
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:],
        layout.lane_bases, layout.warp_bases, layout.block_bases, layout.shape)


@gluon.jit
def _scatter_output(C, value, n, output_rows, output_valid,
                    N: gl.constexpr, BN: gl.constexpr, PARTITION: gl.constexpr):
    CN: gl.constexpr = BN // 2
    store: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    # Two 64-row panels use 16 KiB scratch per consumer at BN256. Only the
    # result changes layout through LDS; route/mask never make that round trip.
    # Pair adjacent columns before narrowing, so the LDS writer receives
    # complete dwords instead of halfwords belonging to unrelated routes.
    packing: gl.constexpr = _output_pack_layout(gl.to_linear_layout(value.type.layout, value.shape))
    value = gl.convert_layout(value, packing).to(C.dtype.element_ty)
    halves = gl.split(value.reshape((2, 64, CN)).permute((1, 2, 0)))
    col = n * BN + PARTITION * CN + gl.arange(0, CN, gl.SliceLayout(0, store))
    for panel in gl.static_range(2):
        output = gl.convert_layout(halves[panel], store)
        offsets = output_rows[panel][:, None] + col[None, :].to(gl.uint32)
        valid = output_valid[panel]
        # N % BN == 0 is a constexpr assertion; only row validity is masked.
        buffer_store(output, C, offsets, valid[:, None])


@gluon.jit
def _empty():
    pass


@gluon.jit
def _initialize(STAGES: gl.constexpr):
    # A READY1/2, EMPTY3/4; Bleft READY5/6, EMPTY7/8;
    # Bright READY9/10, EMPTY11/12. First wait never uses abarrier0.
    for base in gl.static_range(1, 13, 4):
        for slot in gl.static_range(2):
            hcu.abarrier_init(base + slot, 4)
            hcu.abarrier_init(base + 2 + slot, 8 if base == 1 else 4)
    gl.thread_barrier()
    # Each of the two resident slots starts with one EMPTY generation.
    for slot in gl.static_range(STAGES):
        hcu.abarrier_arrive(3 + slot, 2)
        hcu.abarrier_arrive(7 + slot)
        hcu.abarrier_arrive(11 + slot)
    gl.thread_barrier()


@gluon.jit
def _producer(A, B, Experts, am, b0, b1, pid, total, rows,
              N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr,
              BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr,
              USE_MLS_LOAD_A: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
              PARTITION: gl.constexpr):
    epoch = 0
    if USE_MLS_LOAD_A:
        copy: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    else:
        copy: gl.constexpr = am.layout.producer_layout()
    rm = gl.arange(0, BM, gl.SliceLayout(1, copy))
    rk = gl.arange(0, BK, gl.SliceLayout(0, copy))
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        if PARTITION == 0:
            for k in gl.static_range(K // BK):
                ticket = epoch * (K // BK) + k
                slot = ticket & 1
                phase = (ticket >> 1) & 1
                physical = ticket & (STAGES - 1)
                hcu.abarrier_try_wait(3 + slot, phase)
                hcu.abarrier_seq(1 + slot)
                if USE_MLS_LOAD_A:
                    hcu.matrix_load(am.index(physical), A, (rows, K), (K, 1),
                                    (m * BM, k * BK), warps=(4, 1))
                else:
                    offsets = ((m * BM + rm[:, None]) * K + k * BK + rk[None, :]).to(gl.uint32)
                    async_copy.buffer_load_to_shared(am.index(physical), A, offsets)
                hcu.abarrier_arrive(1 + slot)
            epoch += 1
        else:
            expert = gl.load(Experts + m)
            b_base = B + gl.maximum(expert, 0).to(gl.int64) * N * K
            for n in range(0, N // BN):
                for k in gl.static_range(K // BK):
                    ticket = epoch * (K // BK) + k
                    slot = ticket & 1
                    phase = (ticket >> 1) & 1
                    physical = ticket & (STAGES - 1)
                    for half in gl.static_range(2):
                        dest = b0 if half == 0 else b1
                        ready = 5 + 4 * half + slot
                        hcu.abarrier_try_wait(ready + 2, phase)
                        hcu.abarrier_seq(ready)
                        hcu.matrix_load(dest.index(physical), b_base, (K, N), (1, K),
                                        (k * BK, n * BN + half * (BN // 2)), warps=(2, 2))
                        hcu.abarrier_arrive(ready)
                epoch += 1


@gluon.jit
def _consumer(C, SA, SB, Weights, Sorted, Experts, am, bm,
              pid, total, rows, valid_routes,
              N: gl.constexpr, K: gl.constexpr, TOP_K: gl.constexpr,
              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
              STAGES: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
              CK_SORTING: gl.constexpr, PARTITION: gl.constexpr):
    CN: gl.constexpr = BN // 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        4, [16, 16, 32], MMAC_LAYOUT, [2, 2], tiles_per_warp=[1, 2])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    rm = gl.arange(0, BM, gl.SliceLayout(1, result))
    rn = gl.arange(0, CN, gl.SliceLayout(0, result))
    a_epoch = 0
    b_epoch = 0
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        expert = gl.load(Experts + m)
        route = buffer_load(Sorted, (m * BM + rm).to(gl.uint32))
        if CK_SORTING:
            route = (route & 0x00ffffff) * TOP_K + ((route >> 24) & 255)
        valid = (route >= 0) & (route < valid_routes) & (expert >= 0)
        sa = buffer_load(SA, (m * BM + rm).to(gl.uint32))
        weight = buffer_load(Weights, route.to(gl.uint32), valid, other=0)
        output_store: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        output_rm = gl.arange(0, 64, gl.SliceLayout(1, output_store))
        output_rows = ()
        output_valid = ()
        for panel in gl.static_range(2):
            output_route = buffer_load(Sorted, (m * BM + panel * 64 + output_rm).to(gl.uint32))
            if CK_SORTING:
                output_route = (output_route & 0x00ffffff) * TOP_K + ((output_route >> 24) & 255)
            output_rows += (output_route.to(gl.uint32) * N,)
            output_valid += ((output_route >= 0) & (output_route < valid_routes) & (expert >= 0),)
        a_tiles = ()
        for k in gl.static_range(K // BK):
            ticket = a_epoch * (K // BK) + k
            slot = ticket & 1
            phase = (ticket >> 1) & 1
            physical = ticket & (STAGES - 1)
            hcu.abarrier_try_wait(1 + slot, phase)
            a = am.index(physical).load(gl.DotOperandLayout(0, mma, 8))
            a_tiles += (a,)
        hcu.wait_lgkmcnt(0)
        for k in gl.static_range(K // BK):
            ticket = a_epoch * (K // BK) + k
            credit = (ticket + STAGES) & 1
            hcu.abarrier_arrive(3 + credit)
        a_epoch += 1
        for n in range(0, N // BN):
            col = n * BN + PARTITION * CN + rn
            sb = buffer_load(SB, (gl.maximum(expert, 0) * N + col).to(gl.uint32))
            acc = gl.full((BM, CN), 0, gl.float32, result)
            b_tiles = ()
            for k in gl.static_range(K // BK):
                ticket = b_epoch * (K // BK) + k
                slot = ticket & 1
                phase = (ticket >> 1) & 1
                physical = ticket & (STAGES - 1)
                hcu.abarrier_try_wait(5 + 4 * PARTITION + slot, phase)
                b = bm.index(physical).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
                b_tiles += (b,)
            hcu.wait_lgkmcnt(0)
            for k in gl.static_range(K // BK):
                ticket = b_epoch * (K // BK) + k
                credit = (ticket + STAGES) & 1
                hcu.abarrier_arrive(7 + 4 * PARTITION + credit)
            for k in gl.static_range(K // BK):
                hcu.sched_barrier()
                acc = gl.amd.cdna4.mfma(a_tiles[k], b_tiles[k], acc)
                hcu.sched_barrier()
            b_epoch += 1
            # Preserve FP32 multiplication order from the reference kernel.
            value = acc * sa[:, None] * sb[None, :] * weight[:, None]
            _scatter_output(C, value, n, output_rows, output_valid,
                            N, BN, PARTITION)


@gluon.jit
def gemm_bottom_w8a8_channel_wasp_persistent(
    A, B, SA, SB, C, Weights, Sorted, Experts,
    num_sorted_rows, num_valid_routes, TOTAL_TILES,
    N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 128, BLOCK_SIZE_N: gl.constexpr = 256,
    BLOCK_SIZE_K: gl.constexpr = 128, USE_MLS_LOAD_A: gl.constexpr = True,
    GROUP_SIZE_M: gl.constexpr = 1, CK_SORTING: gl.constexpr = False,
    PARTITION_REGS: gl.constexpr = None,
):
    regs: gl.constexpr = (16, 16, 240, 240) if PARTITION_REGS is None else PARTITION_REGS
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    STAGES: gl.constexpr = 2
    gl.static_assert(BM == 128 and (BN == 128 or BN == 256) and BK == 128)
    gl.static_assert(K % BK == 0 and N % BN == 0)
    # The full A preload precedes EMPTY returns; it must fit resident slots.
    gl.static_assert(K <= STAGES * BK, "full short-K A must fit resident LDS slots")
    gl.static_assert(GROUP_SIZE_M == 1, "A reuse uses row-persistent ownership")
    if USE_MLS_LOAD_A:
        al: gl.constexpr = hcu.MlsSharedLayout(0, (16, 64), order=(1, 0))
    else:
        al: gl.constexpr = _buffer_lds_layout(BM, BK, 1, 4, 32, 16)
    bl: gl.constexpr = hcu.MlsSharedLayout(1, (64, 32), alt=1, order=(0, 1))
    am = gl.allocate_shared_memory(A.dtype.element_ty, [STAGES, BM, BK], al)
    b0 = gl.allocate_shared_memory(B.dtype.element_ty, [STAGES, BK, BN // 2], bl)
    b1 = gl.allocate_shared_memory(B.dtype.element_ty, [STAGES, BK, BN // 2], bl)
    _initialize(STAGES)
    pid = gl.program_id(0)
    gl.warp_specialize([
        (_empty, ()),
        (_producer, (A, B, Experts, am, b0, b1, pid, TOTAL_TILES, num_sorted_rows,
            N, K, BM, BN, BK, STAGES, USE_MLS_LOAD_A, GROUP_SIZE_M, 0)),
        (_producer, (A, B, Experts, am, b0, b1, pid, TOTAL_TILES, num_sorted_rows,
            N, K, BM, BN, BK, STAGES, USE_MLS_LOAD_A, GROUP_SIZE_M, 1)),
        (_consumer, (C, SA, SB, Weights, Sorted, Experts, am, b0, pid, TOTAL_TILES,
            num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
            GROUP_SIZE_M, CK_SORTING, 0)),
        (_consumer, (C, SA, SB, Weights, Sorted, Experts, am, b1, pid, TOTAL_TILES,
            num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
            GROUP_SIZE_M, CK_SORTING, 1)),
    ], [4, 4, 4, 4], regs)
    for barrier in gl.static_range(1, 13):
        hcu.abarrier_invalidate(barrier)
