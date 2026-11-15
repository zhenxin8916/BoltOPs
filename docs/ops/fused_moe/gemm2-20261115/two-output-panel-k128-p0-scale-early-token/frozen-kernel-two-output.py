# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Short-K GEMM2: sorted A, channel scales, routed weight, unsorted C.

P0 copies A once per M block, P1 copies Bleft/Bright for each N block;
C0/C1 retain the full short-K A in registers across N. A and B credits
advance independently. A requires both consumers to return EMPTY.
At K=BK=BN256 with MLS A, P0 also scatters C through independent consumer mailboxes,
overlapping output stores with subsequent MMAC and prefetching the next M A.
Other short-K shapes retain direct consumer scatter.
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
    amdgpu_enable_max_ilp_scheduling_strategy=False,
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
    # BK256 has one physical slot but two alternating barrier IDs. Release
    # credit to the NEXT physical owner, not to a second free buffer.
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
            for n in range(0, 2):
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
        for n in range(0, 2):
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
def _send_output_mailbox(C, value, co, out_epoch, BN: gl.constexpr, PARTITION: gl.constexpr):
    CN: gl.constexpr = BN // 2
    packing: gl.constexpr = _output_pack_layout(gl.to_linear_layout(value.type.layout, value.shape))
    value = gl.convert_layout(value, packing).to(C.dtype.element_ty)
    halves = gl.split(value.reshape((2, 64, CN)).permute((1, 2, 0)))
    for panel in gl.static_range(2):
        hcu.abarrier_try_wait((11 + 4 * PARTITION + panel) & 15, out_epoch & 1)
        co.store(halves[panel])
        hcu.wait_lgkmcnt(0)
        hcu.abarrier_arrive(9 + 4 * PARTITION + panel)


@gluon.jit
def _empty_mailbox():
    pass


@gluon.jit
def _initialize_mailbox(STAGES: gl.constexpr):
    # Single short-K B slot. Consumer C-mailbox waits separate successive
    # waits on the same B READY ID; P1 alternates Bleft/Bright EMPTY IDs.
    for i in gl.static_range(1, 3):
        hcu.abarrier_init(i, 4)
        hcu.abarrier_init(i + 2, 8)
    hcu.abarrier_init(5, 4)
    hcu.abarrier_init(6, 4)
    hcu.abarrier_init(7, 4)
    hcu.abarrier_init(8, 4)
    for i in gl.static_range(9, 16):
        hcu.abarrier_init(i, 4)
    hcu.abarrier_init(0, 4)
    gl.thread_barrier()
    hcu.abarrier_arrive(3, 2)
    hcu.abarrier_arrive(6)
    hcu.abarrier_arrive(8)
    hcu.abarrier_arrive(11)
    hcu.abarrier_arrive(15)
    # IDs10/14 are consumer half-MMAC tokens; C READY uses9/13.
    hcu.abarrier_arrive(10)
    gl.thread_barrier()


@gluon.jit
def _copy_a_mailbox(A, am, m, rows, epoch, rm, rk, K: gl.constexpr,
            BM: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr,
            USE_MLS_LOAD_A: gl.constexpr):
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


@gluon.jit
def _producer_mailbox(A, B, Experts, am, b0, b1, C, Sorted, valid_routes, co0, co1, SA, SB, Weights, pid, total, rows,
              N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr,
              BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr,
              USE_MLS_LOAD_A: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
              PARTITION: gl.constexpr, TOP_K: gl.constexpr, CK_SORTING: gl.constexpr):
    epoch = 0
    out_epoch = 0
    if USE_MLS_LOAD_A:
        copy: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    else:
        copy: gl.constexpr = am.layout.producer_layout()
    rm = gl.arange(0, BM, gl.SliceLayout(1, copy))
    rk = gl.arange(0, BK, gl.SliceLayout(0, copy))
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        if PARTITION == 0:
            # C1 FP32 output reuses A LDS only after both A readers finish.
            # Do not overwrite this storage with the next M before draining C1.
            _copy_a_mailbox(A, am, m, rows, epoch, rm, rk, K, BM, BK, STAGES, USE_MLS_LOAD_A)
            epoch += 1
            output_store: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
            output_rm = gl.arange(0, 64, gl.SliceLayout(1, output_store))
            output_rn = gl.arange(0, BN // 2, gl.SliceLayout(0, output_store))
            expert = gl.load(Experts + m)
            output_rows = ()
            output_valid = ()
            output_sa = ()
            output_weight = ()
            for panel in gl.static_range(2):
                route = buffer_load(Sorted, (m * BM + panel * 64 + output_rm).to(gl.uint32))
                output_sa += (buffer_load(SA, (m * BM + panel * 64 + output_rm).to(gl.uint32)),)
                output_weight += (buffer_load(Weights, (m * BM + panel * 64 + output_rm).to(gl.uint32)),)
                if CK_SORTING:
                    route = (route & 0x00ffffff) * TOP_K + ((route >> 24) & 255)
                output_rows += (route.to(gl.uint32) * N,)
                output_valid += ((route >= 0) & (route < valid_routes) & (expert >= 0),)
            for n in range(0, 2):
                col = n * BN + output_rn
                sb0 = buffer_load(SB, (gl.maximum(expert, 0) * N + col).to(gl.uint32))
                sb1 = buffer_load(SB, (gl.maximum(expert, 0) * N + col + BN // 2).to(gl.uint32))
                for panel in gl.static_range(2):
                    hcu.abarrier_try_wait(9, panel)
                    x = co0.load(output_store)
                    hcu.wait_lgkmcnt(0)
                    hcu.abarrier_arrive(12 if panel == 0 else 11)
                    offsets = output_rows[panel][:, None] + col[None, :].to(gl.uint32)
                    valid = output_valid[panel]
                    x = ((x * output_sa[panel][:, None]) * sb0[None, :]) * output_weight[panel][:, None]
                    buffer_store(x.to(C.dtype.element_ty), C, offsets, valid[:, None])
                    hcu.abarrier_try_wait(13, panel)
                    y = co1.load(output_store)
                    hcu.wait_lgkmcnt(0)
                    hcu.abarrier_arrive(0 if panel == 0 else 15)
                    y = ((y * output_sa[panel][:, None]) * sb1[None, :]) * output_weight[panel][:, None]
                    buffer_store(y.to(C.dtype.element_ty), C, offsets + BN // 2, valid[:, None])
                out_epoch += 1
        else:
            expert = gl.load(Experts + m)
            b_base = B + gl.maximum(expert, 0).to(gl.int64) * N * K
            for n in range(0, 2):
                for k in gl.static_range(K // BK):
                    ticket = epoch * (K // BK) + k
                    slot = ticket & 1
                    phase = ticket & 1
                    physical = ticket & (STAGES - 1)
                    for half in gl.static_range(2):
                        dest = b0 if half == 0 else b1
                        ready = 5 + 2 * half
                        hcu.abarrier_try_wait(ready + 1, phase)
                        hcu.abarrier_seq(ready)
                        hcu.matrix_load(dest.index(physical), b_base, (K, N), (1, K),
                                        (k * BK, n * BN + half * (BN // 2)), warps=(2, 2))
                        hcu.abarrier_arrive(ready)
                epoch += 1


@gluon.jit
def _consumer_mailbox(C, SA, SB, Weights, Sorted, Experts, am, bm, co,
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
        a_pair = gl.split(a_tiles[0].reshape((2, BM // 2, BK)).permute((1, 2, 0)))
        a_low = gl.convert_layout(a_pair[0], gl.DotOperandLayout(0, mma, 8))
        a_high = gl.convert_layout(a_pair[1], gl.DotOperandLayout(0, mma, 8))
        a_low_k = gl.split(a_low.reshape((BM // 2, 2, BK // 2)).permute((0, 2, 1)))
        a_high_k = gl.split(a_high.reshape((BM // 2, 2, BK // 2)).permute((0, 2, 1)))
        for n in range(0, 2):
            hcu.abarrier_try_wait(5 + 2 * PARTITION, b_epoch & 1)
            b_first = bm.index(0).slice(0, BK // 2, 0).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
            hcu.wait_lgkmcnt(0)
            hcu.abarrier_try_wait(10 if PARTITION == 0 else 14, b_epoch & 1)
            # READY covers the full logical K256 tile; read its second K128
            # without waiting here, allowing the first MMAC to overlap it.
            b_second = bm.index(0).slice(BK // 2, BK // 2, 0).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
            packing: gl.constexpr = _output_pack_layout(gl.to_linear_layout(result, (BM // 2, CN)))
            for panel in gl.static_range(2):
                a_panel = a_low if panel == 0 else a_high
                acc = gl.full((BM // 2, CN), 0, gl.float32, result)
                hcu.sched_barrier()
                panel_k = a_low_k if panel == 0 else a_high_k
                acc = gl.amd.cdna4.mfma(gl.convert_layout(panel_k[0], gl.DotOperandLayout(0, mma, 8)), b_first, acc)
                hcu.sched_barrier()
                if panel == 0:
                    # Both DS slices must be complete before the producer
                    # may overwrite this logical slot; phase remains N-only.
                    hcu.wait_lgkmcnt(0)
                    hcu.abarrier_arrive(6 + 2 * PARTITION)
                    hcu.abarrier_arrive(14 if PARTITION == 0 else 10)
                acc = gl.amd.cdna4.mfma(gl.convert_layout(panel_k[1], gl.DotOperandLayout(0, mma, 8)), b_second, acc)
                hcu.sched_barrier()
                hcu.abarrier_try_wait((11 + 4 * PARTITION + panel) & 15, b_epoch & 1)
                co.store(acc)
                hcu.wait_lgkmcnt(0)
                hcu.abarrier_arrive(9 + 4 * PARTITION)
            b_epoch += 1





@gluon.jit
def gemm_bottom_w8a8_channel_wasp_persistent(
    A, B, SA, SB, C, Weights, Sorted, Experts,
    num_sorted_rows, num_valid_routes, TOTAL_TILES,
    N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 128, BLOCK_SIZE_N: gl.constexpr = 256,
    BLOCK_SIZE_K: gl.constexpr = 256, USE_MLS_LOAD_A: gl.constexpr = True,
    GROUP_SIZE_M: gl.constexpr = 1, CK_SORTING: gl.constexpr = False,
    PARTITION_REGS: gl.constexpr = None,
):
    if K == 256 and BLOCK_SIZE_K == 256 and BLOCK_SIZE_N == 256 and USE_MLS_LOAD_A:
        regs: gl.constexpr = (112, 4, 196, 196) if PARTITION_REGS is None else PARTITION_REGS
        BM: gl.constexpr = BLOCK_SIZE_M
        BN: gl.constexpr = BLOCK_SIZE_N
        BK: gl.constexpr = BLOCK_SIZE_K
        STAGES: gl.constexpr = 2 if BK == 128 else 1
        gl.static_assert(BM == 128 and (BN == 128 or BN == 256) and (BK == 128 or BK == 256))
        gl.static_assert(K == 256 and BK == 256, "independent output mailboxes require one full-K B slot")
        gl.static_assert(K > 0 and K % BK == 0 and N % BN == 0)
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
        output_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 4, [1, 0])
        co0 = gl.allocate_shared_memory(gl.float32, [64, BN // 2], output_shared)
        co1 = am._reinterpret(gl.float32, [64, BN // 2], output_shared)
        _initialize_mailbox(STAGES)
        pid = gl.program_id(0)
        gl.warp_specialize([
            (_empty_mailbox, ()),
            (_producer_mailbox, (A, B, Experts, am, b0, b1, C, Sorted, num_valid_routes, co0, co1, SA, SB, Weights, pid, TOTAL_TILES, num_sorted_rows,
                N, K, BM, BN, BK, STAGES, USE_MLS_LOAD_A, GROUP_SIZE_M, 0, top_k, CK_SORTING)),
            (_producer_mailbox, (A, B, Experts, am, b0, b1, C, Sorted, num_valid_routes, co0, co1, SA, SB, Weights, pid, TOTAL_TILES, num_sorted_rows,
                N, K, BM, BN, BK, STAGES, USE_MLS_LOAD_A, GROUP_SIZE_M, 1, top_k, CK_SORTING)),
            (_consumer_mailbox, (C, SA, SB, Weights, Sorted, Experts, am, b0, co0, pid, TOTAL_TILES,
                num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
                GROUP_SIZE_M, CK_SORTING, 0)),
            (_consumer_mailbox, (C, SA, SB, Weights, Sorted, Experts, am, b1, co1, pid, TOTAL_TILES,
                num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
                GROUP_SIZE_M, CK_SORTING, 1)),
        ], [4, 4, 4, 4], regs)
        for barrier in gl.static_range(16):
            hcu.abarrier_invalidate(barrier)
    else:
        regs: gl.constexpr = (16, 16, 240, 240) if PARTITION_REGS is None else PARTITION_REGS
        BM: gl.constexpr = BLOCK_SIZE_M
        BN: gl.constexpr = BLOCK_SIZE_N
        BK: gl.constexpr = BLOCK_SIZE_K
        STAGES: gl.constexpr = 2 if BK == 128 else 1
        gl.static_assert(BM == 128 and (BN == 128 or BN == 256) and (BK == 128 or BK == 256))
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
