# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent channel GEMM2 for gfx946, BM64/BN256/K256.

Four roles, four waves each (2P2C): P0 loads A and row metadata; P1 loads B
and column scales; C0/C1 compute the left/right 128 columns, apply the original
((acc * SA) * SB) * weight order, narrow, and scatter routed output directly.

Two full-A slots use 32 KiB. One full-K256 B slot per consumer uses 64 KiB.
Two M metadata slots use 2 KiB. Two SB generations per side use 2 KiB: 100 KiB total shared memory.
A slots alternate READY1/2 and EMPTY3/4, with phase (M_epoch >> 1) & 1.
Metadata alternates READY12/0 with the same phase. B uses READY5/7 and EMPTY6/8,
with phase N_epoch & 1. Initial A/B/metadata reads finish before returning A
and B credits, so metadata remains safe across arbitrarily many M iterations.

FP8 uses FP32 accumulation; INT8 uses INT32 through K, then converts once.
This is a standalone specialization, not selected by the framework dispatcher.
Full N traversal is the default. N_TILES_LIMIT is diagnostic only. The tested
performance scope is 48 CUs, two independent logical BM64 output tiles per TG
at 1.8 GHz. It has half the padded work of two BM128 output tiles."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

# Launch options shared by the library dispatcher and standalone probes.
DEFAULT_LAUNCH_OPTIONS = dict(
    num_warps=4, num_stages=2, use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4), wasp_wdra=True,
)
MMAC_LAYOUT = gl.constexpr(3)  # gl.amd.MmacLayout.LTS, explicit hardware mode.
ROW_PERSISTENT_A = True


@gluon.constexpr_function
def _output_pack_layout(layout):
    """Pair adjacent output halfwords in registers for routed scatter."""
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
    """Initialize all 16 hardware abarriers before splitting the four roles.

Used IDs: A READY1/2, EMPTY3/4 (8 reader waves); B READY5/7,
EMPTY6/8 (4 reader waves each); metadata READY12/0. Other IDs are unused.
A EMPTY starts with two credits per wave; B EMPTY starts with one.
"""
    hcu.abarrier_init(1, 4)
    hcu.abarrier_init(3, 8)
    for i in gl.static_range(2, 16):
        if i != 3:
            hcu.abarrier_init(i, 8 if i == 4 else 4)
    hcu.abarrier_init(0, 4)
    gl.thread_barrier()
    hcu.abarrier_arrive(3, 2)
    hcu.abarrier_arrive(6)
    hcu.abarrier_arrive(4, 2)
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
    """P0 publishes one M in its alternating A and metadata slots.

A slot = epoch & 1, READY = 1 + slot, EMPTY = 3 + slot;
A phase = (epoch >> 1) & 1. Metadata contains four [64] vectors:
SA, weight, route bits and expert bits. Its READY is 12/0 by slot.
Both consumers finish the row-metadata reads before returning A EMPTY.
"""
    ticket = epoch
    slot = ticket & 1
    phase = (ticket >> 1) & 1
    hcu.set_priority(2)
    hcu.abarrier_try_wait(3 + slot, phase)
    hcu.abarrier_seq(1 + slot)
    if USE_A64:
        # Fold the large row origin into the 64-bit descriptor base.
        a_base = A + m.to(gl.int64) * BM * K
        hcu.matrix_load(am.index(slot), a_base, (BM, K), (K, 1), (0, 0), warps=(4, 1))
    else:
        hcu.matrix_load(am.index(slot), A, (rows, K), (K, 1), (m * BM, 0), warps=(4, 1))
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
    """Publish a complete [K256,N128] B half and its channel scales.

HALF0 feeds C0 with READY5/EMPTY6; HALF1 feeds C1 with READY7/EMPTY8.
One physical B slot per side is reused only after its completed-read credit.
SB slot = 2 * (epoch & 1) + HALF. Prefetch SB directly to LDS before
waiting for B EMPTY. Generation e overwrites SB(e-2) only after B(e-1)
was issued; that issue required the B(e-2) EMPTY credit. Consumers return
that credit after both B and SB LDS reads finish, so the overwrite is safe.
The explicit async wait completes SB before publishing this B READY.
KH is retained in this private helper signature but callers always pass zero.
"""
    ready = 7 if HALF == 1 else 5
    empty = 8 if HALF == 1 else 6
    phase = epoch & 1
    physical = 0
    dest = b1 if HALF == 1 else b0
    if KH == 0:
        ml: gl.constexpr = gl.BlockedLayout([4],[64],[4],[0])
        mr = gl.arange(0,BN//2,ml)
        col = n*BN+HALF*(BN//2)+mr
        async_copy.buffer_load_to_shared(mb.index((epoch&1)*2+HALF),SB,(gl.maximum(expert,0)*STRIDE_SBE+col*STRIDE_SBN).to(gl.uint32))
        async_copy.commit_group()
    hcu.abarrier_try_wait(empty, phase)
    hcu.abarrier_seq(ready)
    if USE_B64:
        # Fold the N-panel origin into the 64-bit base, keeping MLS offsets local.
        panel_base = b_base + (n.to(gl.int64) * BN + HALF * (BN // 2)) * K + 0
        hcu.matrix_load(dest.index(physical), panel_base, (K, BN // 2),
                        (1, K), (0, 0), warps=(1, 4))
    else:
        hcu.matrix_load(dest.index(physical), b_base, (K, N), (1, K),
                        (0, n * BN + HALF * (BN // 2)), warps=(1, 4))
    async_copy.wait_group(0)
    hcu.abarrier_arrive(ready)




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
    """P0 streams A/metadata across two slots; P1 streams both B halves.

P0 can issue the next M without waiting for the current output stores.
P1's epochs advance per N tile, while P0's epochs advance per M.
"""
    if num_tokens_post_padded_ptr is not None:
        total = gl.minimum(total, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * (N // BN))
    epoch = 0
    output_epoch = 0
    ml: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    mr = gl.arange(0, BN // 2, ml)
    output_store: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    output_rm = gl.arange(0, 64, gl.SliceLayout(1, output_store))
    output_rn = gl.arange(0, BN // 2, gl.SliceLayout(0, output_store))
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        if PARTITION == 0:
            _load_a_and_metadata(A, SA, Weights, Sorted, Experts, am, ma, m, rows, epoch,
                K, BM, BK, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT)
            epoch += 1
        else:
            expert = gl.load(Experts + m)
            b_base = B + gl.maximum(expert, 0).to(gl.int64) * N * K
            for n in range(0, N_TILES):
                _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch, 0, 0, N, K, BN, BK,
                    USE_B64, STRIDE_SBE, STRIDE_SBN)
                _load_b_panel(B, SB, b_base, expert, b0, b1, mb, n, epoch, 0, 1, N, K, BN, BK,
                    USE_B64, STRIDE_SBE, STRIDE_SBN)
                epoch += 1


@gluon.jit
def _consumer(C, SA, SB, Weights, Sorted, Experts, am, bm, co, ma, mb,
              pid, total, rows, valid_routes,
              N: gl.constexpr, K: gl.constexpr, TOP_K: gl.constexpr,
              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
              STAGES: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
              CK_SORTING: gl.constexpr, PARTITION: gl.constexpr, N_TILES: gl.constexpr,
              num_tokens_post_padded_ptr):
    """Retain A across N, compute full K256, and scatter a 64x128 half.

First N: issue A/B/SB and metadata reads; wait once before returning
A/metadata and B credits. Later N: retain A/metadata in registers.
After MMAC, preserve scale order and pair BF16/FP16 values in registers.
The natural accumulator distribution writes packed dwords; no C mailbox
or global scale/route load is needed in either consumer.
"""
    if num_tokens_post_padded_ptr is not None:
        total = gl.minimum(total, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * (N // BN))
    CN: gl.constexpr = BN // 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], MMAC_LAYOUT, [2, 2], tiles_per_warp=[1, 2])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    packing: gl.constexpr = _output_pack_layout(gl.to_linear_layout(result, (BM, CN)))
    output_store: gl.constexpr = packing
    rn = gl.arange(0, CN, gl.SliceLayout(0, output_store))
    a_epoch = 0
    b_epoch = 0
    ACC: gl.constexpr = gl.int32 if am.dtype == gl.int8 else gl.float32
    for m in range(pid, total // (N // BN), gl.num_programs(0)):
        hcu.abarrier_try_wait(1 + (a_epoch & 1), (a_epoch >> 1) & 1)
        a = am.index(a_epoch & 1).load(gl.DotOperandLayout(0, mma, 8))
        sa = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, result))
        weight = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, result))
        route = gl.full((BM,), 0, gl.int32, gl.SliceLayout(1, result))
        expert = gl.full((BM,), 0, gl.int32, gl.SliceLayout(1, result))
        for n in range(0, N_TILES):
            hcu.abarrier_try_wait(7 if PARTITION == 1 else 5, b_epoch & 1)
            current_sb = mb.index((b_epoch&1)*2+PARTITION).load(gl.SliceLayout(0, result))
            b = bm.index(0).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
            if n == 0:
                hcu.abarrier_try_wait(12 if (a_epoch & 1) == 0 else 0, (a_epoch >> 1) & 1)
                sa = ma.index((a_epoch & 1) * 4).load(gl.SliceLayout(1, result))
                weight = ma.index((a_epoch & 1) * 4 + 1).load(gl.SliceLayout(1, result))
                route = ma.index((a_epoch & 1) * 4 + 2).load(gl.SliceLayout(1, result)).to(gl.int32, bitcast=True)
                expert = ma.index((a_epoch & 1) * 4 + 3).load(gl.SliceLayout(1, result)).to(gl.int32, bitcast=True)
            hcu.wait_lgkmcnt(0)
            if n == 0:
                hcu.abarrier_arrive(3 + (a_epoch & 1))
            hcu.abarrier_arrive(8 if PARTITION == 1 else 6)
            acc = gl.full((BM, CN), 0, ACC, result)
            hcu.sched_barrier()
            acc = gl.amd.cdna4.mfma(a, b, acc)
            hcu.sched_barrier()
            if acc.dtype == gl.int32:
                acc = acc.to(gl.float32)
            value = ((acc * sa[:, None]) * current_sb[None, :]) * weight[:, None]
            packed = gl.convert_layout(value, packing).to(C.dtype.element_ty)
            out_route = gl.convert_layout(route, gl.SliceLayout(1, output_store))
            out_expert = gl.convert_layout(expert, gl.SliceLayout(1, output_store))
            if CK_SORTING:
                out_route = (out_route & 0x00ffffff) * TOP_K + ((out_route >> 24) & 255)
            offsets = out_route.to(gl.uint32)[:, None] * N + n * BN + PARTITION * CN + rn[None, :]
            valid = (out_route >= 0) & (out_route < valid_routes) & (out_expert >= 0)
            buffer_store(packed, C, offsets, valid[:, None])
            b_epoch += 1
        a_epoch += 1



@gluon.jit
def gemm_bottom_w8a8_channel_wasp_persistent(
    A, B, SA, SB, C, Weights, Sorted, Experts,
    num_sorted_rows, num_valid_routes, TOTAL_TILES,
    N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 64, BLOCK_SIZE_N: gl.constexpr = 256,
    BLOCK_SIZE_K: gl.constexpr = 256, GROUP_SIZE_M: gl.constexpr = 1,
    CK_SORTING: gl.constexpr = True, MUL_ROUTED_WEIGHT: gl.constexpr = True,
    PARTITION_REGS: gl.constexpr = None,
    USE_ADDR_OFFSET_INT64_A: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_B: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr = False,
    stride_asm: gl.constexpr = 1, stride_bse: gl.constexpr = None,
    stride_bsn: gl.constexpr = 1, N_TILES_LIMIT: gl.constexpr = 0,
    num_tokens_post_padded_ptr=None,
):
    """Launch the standalone BM64/BN256/BK256 2P2C kernel.

TOTAL_TILES = M blocks * (N/256). Each physical persistent TG traverses
M += grid; a two-output diagnostic uses 96 M blocks and grid 48.
C is route-major output before top-k reduction. SortedWeights is supplied
when routed-weight multiplication is enabled. C64 is not implemented here.
"""
    gl.static_assert(BLOCK_SIZE_M == 64 and BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 256)
    gl.static_assert(K == 256 and N % 256 == 0 and N > 0)
    gl.static_assert(not USE_ADDR_OFFSET_INT64_C, "C64 scatter is not implemented in this standalone kernel")
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
    bl: gl.constexpr = hcu.MlsSharedLayout(1, (128, 32), alt=1, order=(0, 1))
    # Two A [64,256] slots total 32 KiB; each B [256,128] slot is 32 KiB.
    am = gl.allocate_shared_memory(A.dtype.element_ty, [2, BM, K], al)
    b0 = gl.allocate_shared_memory(B.dtype.element_ty, [STAGES, K, BN // 2], bl)
    b1 = gl.allocate_shared_memory(B.dtype.element_ty, [STAGES, K, BN // 2], bl)
    # Consumers scatter directly; no output mailbox allocation.
    output_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 4, [1, 0])
    meta_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    ma_layout: gl.constexpr = gl.SharedLinearLayout([[4], [8], [1], [2], [16], [32]], [], 16)
    # MA: two M slots x four [64] FP32 vectors = 2 KiB.
    # MB: two generations x left/right [128] FP32 scales = 2 KiB.
    ma = gl.allocate_shared_memory(gl.float32, [8, BM], ma_layout)
    mb = gl.allocate_shared_memory(gl.float32, [4, BN // 2], meta_shared)
    co0 = ma
    co1 = ma
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
            GROUP_SIZE_M, CK_SORTING, 0, N_TILES, num_tokens_post_padded_ptr)),
        (_consumer, (
            C, SA, SB, Weights, Sorted, Experts, am, b1, co1, ma, mb, pid, TOTAL_TILES,
            num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
            GROUP_SIZE_M, CK_SORTING, 1, N_TILES, num_tokens_post_padded_ptr)),
    ], [4, 4, 4, 4], regs)
    for barrier in gl.static_range(16):
        hcu.abarrier_invalidate(barrier)


