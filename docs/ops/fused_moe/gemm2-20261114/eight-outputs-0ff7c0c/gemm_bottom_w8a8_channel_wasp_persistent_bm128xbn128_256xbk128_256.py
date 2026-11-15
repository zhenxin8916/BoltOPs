# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Short-K GEMM2: sorted A, channel scales, routed weight, unsorted C.

P0 copies A, P1 copies Bleft/Bright; C0/C1 split N. Credits continue across
persistent output tiles. A requires both consumers to return EMPTY.
Output is [M * top_k, N]; top-k reduction remains a separate operation.
The compile-time short K is fully unrolled; persistent output traversal stays
dynamic so generated code does not grow with the number of routed tiles.
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


@gluon.constexpr_function
def _contiguous_n_store_layout(layout):
    # Interleave2 LTS has register bases M4, M8, N1. Pack N1 first so
    # converting to BF16 pairs adjacent columns, rather than distinct routes.
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:],
        layout.lane_bases, layout.warp_bases, layout.block_bases, layout.shape)


@gluon.jit
def _empty():
    pass


@gluon.jit
def _pid(tile, num_m, num_n, GROUP_SIZE_M: gl.constexpr):
    if GROUP_SIZE_M == 1:
        return tile // num_n, tile % num_n
    first_m = (tile // (GROUP_SIZE_M * num_n)) * GROUP_SIZE_M
    size = gl.minimum(num_m - first_m, GROUP_SIZE_M)
    return first_m + tile % size, (tile % (GROUP_SIZE_M * num_n)) // size


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
    for tile in range(pid, total, gl.num_programs(0)):
        m, n = _pid(tile, gl.cdiv(rows, BM), gl.cdiv(N, BN), GROUP_SIZE_M)
        expert = gl.load(Experts + m)
        b_base = B + gl.maximum(expert, 0).to(gl.int64) * N * K
        for k in gl.static_range(K // BK):
            slot = epoch & 1
            phase = (epoch >> 1) & 1
            physical = epoch % STAGES
            if PARTITION == 0:
                hcu.abarrier_try_wait(3 + slot, phase)
                hcu.abarrier_seq(1 + slot)
                if USE_MLS_LOAD_A:
                    hcu.matrix_load(am.index(physical), A, (rows, K), (K, 1),
                                    (m * BM, k * BK), warps=(4, 1))
                else:
                    offsets = ((m * BM + rm[:, None]) * K + k * BK + rk[None, :]).to(gl.uint32)
                    async_copy.buffer_load_to_shared(am.index(physical), A, offsets)
                hcu.abarrier_arrive(1 + slot)
            else:
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
    epoch = 0
    for tile in range(pid, total, gl.num_programs(0)):
        m, n = _pid(tile, gl.cdiv(rows, BM), gl.cdiv(N, BN), GROUP_SIZE_M)
        expert = gl.load(Experts + m)
        route = buffer_load(Sorted, (m * BM + rm).to(gl.uint32))
        if CK_SORTING:
            route = (route & 0x00ffffff) * TOP_K + ((route >> 24) & 255)
        valid = (route >= 0) & (route < valid_routes) & (expert >= 0)
        sa = buffer_load(SA, (m * BM + rm).to(gl.uint32))
        col = n * BN + PARTITION * CN + rn
        sb = buffer_load(SB, (gl.maximum(expert, 0) * N + col).to(gl.uint32))
        weight = buffer_load(Weights, route.to(gl.uint32), valid, other=0)
        acc = gl.full((BM, CN), 0, gl.float32, result)
        for k in gl.static_range(K // BK):
            slot = epoch & 1
            phase = (epoch >> 1) & 1
            physical = epoch % STAGES
            hcu.abarrier_try_wait(1 + slot, phase)
            a = am.index(physical).load(gl.DotOperandLayout(0, mma, 8))
            hcu.abarrier_try_wait(5 + 4 * PARTITION + slot, phase)
            b = bm.index(physical).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
            hcu.wait_lgkmcnt(0)
            credit = (epoch + STAGES) & 1
            hcu.abarrier_arrive(3 + credit)
            hcu.abarrier_arrive(7 + 4 * PARTITION + credit)
            hcu.sched_barrier()
            acc = gl.amd.cdna4.mfma(a, b, acc)
            hcu.sched_barrier()
            epoch += 1
        # Triton order: FP32 channel scales, routed weight, then BF16/FP16.
        value = acc * sa[:, None] * sb[None, :] * weight[:, None]
        store: gl.constexpr = _contiguous_n_store_layout(gl.to_linear_layout(result, (BM, CN)))
        offsets = route[:, None].to(gl.uint32) * N + col[None, :].to(gl.uint32)
        mask = valid[:, None] & (col[None, :] < N)
        value = gl.convert_layout(value, store).to(C.dtype.element_ty)
        buffer_store(value, C, gl.convert_layout(offsets, store), gl.convert_layout(mask, store))


@gluon.jit
def gemm_bottom_w8a8_channel_wasp_persistent(
    A, B, SA, SB, C, Weights, Sorted, Experts,
    num_sorted_rows, num_valid_routes, TOTAL_TILES,
    N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 128, BLOCK_SIZE_N: gl.constexpr = 256,
    BLOCK_SIZE_K: gl.constexpr = 128, USE_MLS_LOAD_A: gl.constexpr = True,
    GROUP_SIZE_M: gl.constexpr = 1, CK_SORTING: gl.constexpr = False,
    PARTITION_REGS: gl.constexpr = (48, 48, 208, 208),
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    STAGES: gl.constexpr = 2 if BK == 128 else 1
    gl.static_assert(BM == 128 and (BN == 128 or BN == 256) and (BK == 128 or BK == 256))
    gl.static_assert(K % BK == 0 and N % BN == 0)
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
    ], [4, 4, 4, 4], PARTITION_REGS)
    for barrier in gl.static_range(1, 13):
        hcu.abarrier_invalidate(barrier)
