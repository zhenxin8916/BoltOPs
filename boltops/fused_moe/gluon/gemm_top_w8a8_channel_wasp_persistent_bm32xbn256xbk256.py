# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent channel GEMM1 for gfx946, BM32/BN256/BK256.

Contiguous FP8 (E4M3FN/E5M2, including mixed formats) or signed INT8 A/B;
FP32 channel scales; BF16/FP16 sorted C. CK token[23:0] selects A.
CPU selection is limited to M128; K>=1024, K%512=0, N%256=0. Four waves per role; grid is caller-supplied.

    P0: B-left panel0 -> B-right panel0 -> B-left panel1 -> B-right panel1
    P1:               A0/A1, with EB3/EB4 granting exclusive issue turns
    C0: left N128 = two N64 panels; C1: right N128 = two N64 panels

VGPR role budgets: P0/P1/C0/C1 = 16/16/80/80; no spill in the initial probe.
The 64-KiB LDS footprint permits two TGs per CU when the launch supplies enough work.

One K256 step, per consumer (arrows describe issue/dependency order):
    panel0: wait B/A -> read A/B -> complete LDS reads -> MMAC
    panel1: reuse A registers -> read B -> complete reads -> MMAC
    Normal K: return A after panel1. Final K: return A after panel0 read.

LDS and abarrier ownership (all 16 IDs initialized):
    A: 2 x [32,256] = 16 KiB; READY1,2 / EMPTY4,5
    B: 3 x [256,64] = 48 KiB FIFO; READY7..9 / EMPTY10..12
    A EMPTY: 4 C0 + 4 C1 waves. B EMPTY: four owning-consumer waves.
    Metadata aliases retired A0; scales alias retired A1. EB5 protects scales
    across outputs; FIFO slot/phase tickets continue when tile_id += grid.

B always uses MLS; A uses buffer-to-LDS (or typed load/store for wide A).
INT8 accumulates in INT32, FP8 in FP32; scales apply after K. Instruction
warmup defaults off; FP8 retains the original zero-MMAC initialization.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna4 import async_copy

MMAC_LAYOUT = gl.constexpr(3)  # LIT=0, LTS=1 (TRANSPOSE=3).
INST_M = gl.constexpr(16)
INST_N = gl.constexpr(16)
INST_K = gl.constexpr(32)
INIT_INST_K = gl.constexpr(16)  # BF16 zero initialization for FP8 GEMM.
INIT_K_WIDTH = gl.constexpr(4)
DOT_K_WIDTH = gl.constexpr(8)
WAVE_SIZE = gl.constexpr(64)
WAVES = gl.constexpr(4)
PRODUCER_WAVES = gl.constexpr(4)
CONSUMERS = gl.constexpr(2)


# Launch options shared by the library dispatcher and standalone probes.
DEFAULT_LAUNCH_OPTIONS = dict(
    num_warps=4,
    num_stages=2,
    use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4),
    wasp_wdra=True,
)


@gluon.constexpr_function
def _matrix_store_cast_layout(layout):
    # Native store packs each eight 16-bit elements as 0,4,1,5,2,6,3,7.
    # Rotate only register bases, preserving lane/warp ownership and addresses.
    """Match matrix_store register packing without changing lane/wave ownership."""
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:],
        layout.lane_bases,
        layout.warp_bases,
        layout.block_bases,
        layout.shape,
    )


@gluon.jit
def _cast_for_matrix_store(value, dtype: gl.constexpr, result: gl.constexpr):
    """Reorder FP32 registers, narrow to 16 bits, then restore the store layout."""
    linear: gl.constexpr = gl.to_linear_layout(value.type.layout, value.shape)
    packing: gl.constexpr = _matrix_store_cast_layout(linear)
    value = gl.convert_layout(value, packing)
    value = value.to(dtype)
    return gl.convert_layout(value, result)


@gluon.jit
def store_channel_output(
    C,
    acc,
    sa,
    sb,
    block_m,
    block_n,
    N: gl.constexpr,
    BN: gl.constexpr,
    PARTITION: gl.constexpr,
    WM: gl.constexpr,
    WN: gl.constexpr,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr,
):
    """Scale one M32/N64 quarter and write it with native matrix_store."""
    result: gl.constexpr = acc.type.layout
    gl.static_assert(acc.shape == [32, 64] and WM == 2 and WN == 2 and BN == 128)
    if acc.dtype == gl.int32:
        acc = acc.to(gl.float32)
    part = _cast_for_matrix_store(acc * sa[:, None] * sb[None, :], C.dtype.element_ty, result)
    C = C + (block_m.to(gl.int64) if USE_ADDR_OFFSET_INT64_C else block_m) * 32 * N
    hcu.matrix_store(C, part, (32, N), (N, 1),
                     (0, block_n * BN + PARTITION * 64), tile=(16, 32), boundary_check=())


@gluon.jit
def gemm1_pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: gl.constexpr):
    """Map a logical tile to M/N with grouped-M traversal, matching ordinary MoE."""
    if GROUP_SIZE_M == 1:
        return pid // num_pid_n, pid % num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    return first_pid_m + pid % group_size_m, (pid % num_pid_in_group) // group_size_m


@gluon.jit
def _empty():
    """Keep the default WASP partition idle while retaining the common entry/exit path."""
    pass


@gluon.constexpr_function
def _icache_label(partition, slot, kind):
    """Generate separate MMAC entry/resume labels for each consumer and ping/pong block."""
    return f".Lmoe_iwarm_c{partition}_b{slot}_{kind}"


@gluon.jit
def _warm_mmac_instructions(PARTITION: gl.constexpr, SLOT: gl.constexpr):
    # Visit the real MMAC block with EXEC=0, skipping LDS and handoff code.
    # Ping/pong are visited concurrently by separate waves; a shared consumer
    # rendezvous below waits for both visits before reusing the B ring.
    # Symbolic labels avoid fixed code addresses. Normal compute pays one
    # untaken EXEC=0 return guard per BK; scalar temporaries are allocated.
    """Visit one real MMAC block with EXEC=0; ping/pong warm concurrently on different SIMDs."""
    # ### ASM Note: The optional instruction-cache visit needs direct EXEC/branch control absent from Gluon.
    saved_exec = gl.inline_asm_elementwise(
        "s_mov_b64 $0, exec\n\ts_mov_b64 exec, 0",
        constraints="=&s,~{exec},~{memory}",
        args=[],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )
    # ### ASM Note: The optional instruction-cache visit needs direct EXEC/branch control absent from Gluon.
    gl.inline_asm_elementwise(
        "s_branch "
        + _icache_label(PARTITION, SLOT, "mmac")
        + "\n"
        + _icache_label(PARTITION, SLOT, "resume")
        + ":",
        constraints="=s,~{memory}",
        args=[],
        dtype=gl.uint32,
        is_pure=False,
        pack=1,
    )
    # ### ASM Note: The optional instruction-cache visit needs direct EXEC/branch control absent from Gluon.
    gl.inline_asm_elementwise(
        "s_mov_b64 exec, $1",
        constraints="=s,s,~{exec},~{memory}",
        args=[saved_exec],
        dtype=gl.uint32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _consumer_pingpong(ID, WAIT: gl.constexpr, WAVES: gl.constexpr):
    # One wave from each consumer contributes to each of the four groups.
    """Wait or notify the matching C0/C1 SIMD wave through a count2 ebarrier."""
    hcu.sched_barrier()
    if WAIT:
        hcu.ebarrier_sync(ID, 2)
    else:
        hcu.ebarrier_arrive(ID, 2)
    hcu.sched_barrier()


@gluon.jit
def _next_a_slot(slot, phase, STAGES: gl.constexpr):
    """Advance A0/A1, flipping phase when slot1 wraps back to slot0."""
    wrap = slot == STAGES - 1
    return gl.where(wrap, 0, slot + 1), phase ^ wrap.to(gl.int32)


@gluon.constexpr_function
def _abarrier_init_asm(stages):
    """Emit all 16 barrier initializers with A/B ownership counts."""
    return (
        "\n\t".join(f"s_abarrier_init {i}, {8 if 4<=i<4+stages else 4}" for i in range(16))
        + "\n\ts_mov_b32 $0, 0"
    )


@gluon.jit
def _initialize(STAGES: gl.constexpr):
    """Initialize READY/EMPTY generations and seed reusable A/B slots."""
    # ### ASM Note: Emit all 16 initializers as one fixed ordered batch rather than independent API operations.
    gl.inline_asm_elementwise(
        _abarrier_init_asm(STAGES), constraints="=s", args=[], dtype=gl.int32, is_pure=False, pack=1
    )
    gl.thread_barrier()
    for i in gl.static_range(4, 4 + STAGES):
        hcu.abarrier_arrive(i, 2)
    for i in gl.static_range(10, 13):
        hcu.abarrier_arrive(i)


@gluon.jit
def _load_b_panel(B, bm, base, k, slot, phase, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr):
    """Wait for a free FIFO slot, issue its native B MLS, then publish READY."""
    # BK256/N64 uses MLS waves [K,N]=[4,1]; two N32 repetitions.
    warps: gl.constexpr = (4, 1)
    hcu.abarrier_try_wait(10 + slot, phase)
    hcu.abarrier_seq(7 + slot)
    hcu.matrix_load(bm.index(slot), base + k * 256, (K, N), (1, K), (0, 0), warps=warps)
    hcu.abarrier_arrive(7 + slot)


@gluon.jit
def _load_a_tile(am, A, k, slot, phase, offsets, mask, USE_A64: gl.constexpr = False):
    """Take the P0 issue grant, fill one A slot, and publish READY."""
    hcu.abarrier_try_wait(4 + slot, phase)
    hcu.ebarrier_sync(3, 8)
    hcu.sched_barrier()
    hcu.set_priority(2)
    hcu.sched_barrier()
    hcu.abarrier_seq(1 + slot)
    if USE_A64:
        packed = am.index(slot)._reinterpret(
            gl.int32, [32, 64], gl.SwizzledSharedLayout(4, 1, 1, [1, 0])
        )
        source = (A + gl.cast(k, gl.int64) * 256).to(gl.pointer_type(gl.int32))
        packed.store(gl.load(source + offsets, mask, other=0))
        hcu.wait_lgkmcnt(0)
    else:
        async_copy.buffer_load_to_shared(
            am.index(slot), (A + k * 256).to(gl.pointer_type(am.dtype)), offsets, mask
        )
    hcu.abarrier_arrive(1 + slot)
    hcu.sched_barrier()
    hcu.set_priority(1)
    hcu.sched_barrier()
    hcu.ebarrier_arrive(4, 8)


@gluon.jit
def _produce_a_and_scales(
    A,
    SA,
    SB,
    am,
    meta_mem,
    scales,
    M: gl.constexpr,
    E: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_bse: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_asm: gl.constexpr,
    total: gl.constexpr,
    BN: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ID: gl.constexpr,
    num_tokens_post_padded_ptr=None,
    PERSISTENT_GRID: gl.constexpr = 0,
    USE_A64: gl.constexpr = False,
):
    """P1: load CK-selected A tiles and publish tail channel scales."""
    BM: gl.constexpr = 32
    BK: gl.constexpr = 256
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    if num_tokens_post_padded_ptr is not None:
        padded_rows = gl.load(num_tokens_post_padded_ptr)
        total = gl.minimum(total, gl.cdiv(padded_rows, 32) * gl.cdiv(N, BN))
    # ### ASM Note: Keep K bounds in SGPRs; Python constants permit unwanted loop unrolling.
    q = gl.inline_asm_elementwise(
        "s_mov_b32 $0, $1",
        constraints="=s,s",
        args=[K // BK],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    phase, iteration = (0, 0)
    for output in range(
        gl.program_id(0) + TILE_ID,
        total,
        PERSISTENT_GRID if PERSISTENT_GRID > 0 else gl.num_programs(0),
    ):
        row, col = gemm1_pid_grid(
            output, gl.cdiv(total, gl.cdiv(N, BN)), gl.cdiv(N, BN), GROUP_SIZE_M
        )
        bank = iteration & 1
        metadata = meta_mem.index(bank)
        hcu.ebarrier_sync(1, 8)
        expert = gl.sum(metadata.slice(BM, 1).load(layout), 0)
        safe = gl.maximum(0, gl.minimum(expert, E - 1))
        hcu.set_priority(1)
        al: gl.constexpr = gl.BlockedLayout([1, 4 if USE_A64 else 16], [4, 16], [4, 1], [1, 0])
        routes = metadata.slice(0, BM).load(gl.SliceLayout(1, al)).to(gl.uint32) & 16777215
        valid = (routes < M) & (expert >= 0) & (expert < E)
        width: gl.constexpr = BK // 4 if USE_A64 else BK
        vec: gl.constexpr = 4 if USE_A64 else 16
        rk = gl.arange(0, width, gl.SliceLayout(0, al))
        rr = gl.arange(0, BM, gl.SliceLayout(1, al))
        logical_k = (rk[None, :] // vec ^ rr[:, None] & 15) * vec + rk[None, :] % vec
        offsets = (
            routes.to(gl.int64)[:, None] * (K // 4) if USE_A64 else routes[:, None] * K
        ) + logical_k
        mask = valid[:, None] & gl.full((BM, width), True, gl.int1, al)
        scale_routes = metadata.slice(0, BM).load(layout).to(gl.uint32) & 16777215
        _load_a_tile(am, A, 0, 0, phase, offsets, mask, USE_A64)
        rn = col * BN + gl.arange(0, BN, layout)
        sa = gl.load(SA + scale_routes.to(gl.int64) * stride_asm, scale_routes < M, other=0)
        sb = gl.load(
            SB + safe.to(gl.int64) * stride_bse + rn.to(gl.int64) * stride_bsn, rn < N, other=0
        )
        hcu.sched_barrier()
        if iteration > 0:
            hcu.ebarrier_sync(5, 12)
        _load_a_tile(am, A, 1, 1, phase, offsets, mask, USE_A64)
        phase = phase ^ 1
        for pair in range(1, q // 2):
            for half in gl.static_range(2):
                k = pair * 2 + half
                _load_a_tile(am, A, k, half, phase, offsets, mask, USE_A64)
                if half == 1:
                    phase = phase ^ 1
        hcu.abarrier_try_wait(4, phase)
        hcu.abarrier_try_wait(5, phase)
        scales.index(bank).slice(0, BM).store(sa)
        scales.index(bank).slice(BN, BN).store(sb)
        hcu.wait_lgkmcnt(0)
        hcu.ebarrier_arrive(2, 8)
        hcu.ebarrier_arrive(0, 8)
        iteration += 1


@gluon.jit
def gemm_top_w8a8_channel_wasp_persistent(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    M: gl.constexpr,
    E: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    top_k: gl.constexpr,
    # Channel scales have a singleton K dimension; its strides are unused.
    stride_bse: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    stride_asm: gl.constexpr,
    stride_ask: gl.constexpr,
    # Capacity of logical output tiles; the device counter limits actual work.
    TOTAL_TILES: gl.constexpr,
    # Autotune configuration (subject to the kernel's tile/resource constraints).
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    # Caller selects flags from tensor.numel() * tensor.element_size() >= 4 GiB - 1.
    # B keeps the common API flag; its copy path always folds expert/N/K into
    # a 64-bit base and selects local offset width from the native half-tile span.
    USE_ADDR_OFFSET_INT64_A: gl.constexpr,
    USE_ADDR_OFFSET_INT64_B: gl.constexpr,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr,
    ck_sorting: gl.constexpr = True,
    GROUP_SIZE_M: gl.constexpr = 1,
    A_STAGES: gl.constexpr = 2,  # Fixed two-stage BK256 A pool.
    PINGPONG: gl.constexpr = True,
    INSN_WARMUP: gl.constexpr = False,
    # Fixed physical A layout; retained for common launch compatibility.
    LDS_WRAP: gl.constexpr = 32,
    COPY_BYTES: gl.constexpr = 16,
    # Launch/probe configuration; not autotune parameters.
    PARTITION_REGS: gl.constexpr = None,  # P0, P1, C0, C1; selected by width/copy type
    TILE_ID: gl.constexpr = 0,  # Single-TG probe offset
    PERSISTENT_GRID: gl.constexpr = 0,
    num_tokens_post_padded_ptr=None,  # Optional device-side CK work count.
):
    """Run BM32/BN256/BK256 using two N64 panels per consumer.

    TOTAL_TILES is a capacity bound narrowed by the optional GPU counter.
    Metadata and channel scales alias the two retired A slots.
    """
    partition_regs: gl.constexpr = (16, 16, 80, 80) if PARTITION_REGS is None else PARTITION_REGS
    gl.static_assert(M == 128, "this standalone specialization covers M128 only")
    gl.static_assert(ck_sorting, "this kernel requires CK-packed sorted token IDs")
    gl.static_assert(0 < top_k and top_k <= 256, "CK packs the route index in eight bits")
    gl.static_assert(GROUP_SIZE_M >= 1)
    gl.static_assert(BLOCK_SIZE_M == 32 and BLOCK_SIZE_N == 256)
    gl.static_assert(BLOCK_SIZE_K == 256)
    gl.static_assert(
        K >= 4 * BLOCK_SIZE_K and K % (2 * BLOCK_SIZE_K) == 0,
        "persistent K requires at least four complete BK tiles, in ping/pong pairs",
    )
    gl.static_assert(
        (a_ptr.dtype.element_ty.is_fp8() and b_ptr.dtype.element_ty.is_fp8())
        or (a_ptr.dtype.element_ty == gl.int8 and b_ptr.dtype.element_ty == gl.int8)
    )
    gl.static_assert(
        a_scale_ptr.dtype.element_ty == gl.float32 and b_scale_ptr.dtype.element_ty == gl.float32
    )
    gl.static_assert(WAVES % PRODUCER_WAVES == 0 and (not PINGPONG or WAVES == 4))
    gl.static_assert(PINGPONG, "BN256 requires consumer pingpong")
    _run(
        a_ptr,
        b_ptr,
        a_scale_ptr,
        b_scale_ptr,
        c_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        M,
        E,
        N,
        K,
        top_k,
        stride_bse,
        stride_bsn,
        0,
        stride_asm,
        0,
        TOTAL_TILES,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        USE_ADDR_OFFSET_INT64_A,
        USE_ADDR_OFFSET_INT64_B,
        USE_ADDR_OFFSET_INT64_C,
        INSN_WARMUP,
        GROUP_SIZE_M,
        partition_regs,
        A_STAGES,
        LDS_WRAP,
        COPY_BYTES,
        TILE_ID,
        num_tokens_post_padded_ptr,
        PERSISTENT_GRID,
    )


@gluon.jit
def _next_b_tickets(state, STEP: gl.constexpr):
    """Advance one of six FIFO generations: slot=ticket%3, phase=ticket//3."""
    advanced = state + STEP
    return gl.where(advanced >= 6, advanced - 6, advanced)


@gluon.jit
def _produce_b_and_routes(
    B,
    Routes,
    Experts,
    bm,
    meta_mem,
    E: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    total: gl.constexpr,
    BN: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ID: gl.constexpr,
    num_tokens_post_padded_ptr=None,
    PERSISTENT_GRID: gl.constexpr = 0,
):
    """P0: publish CK routes/expert, then stream four N64 B panels per K256."""
    BM: gl.constexpr = 32
    BK: gl.constexpr = 256
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    if GROUP_SIZE_M == 1:
        first_output = gl.program_id(0) + TILE_ID
        first_row = first_output // gl.cdiv(N, BN)
        seed_routes = gl.load(
            Routes + first_row * BM + gl.arange(0, BM, layout), first_output < total, other=0
        )
        seed_expert = gl.load(
            Experts + first_row + gl.arange(0, 1, layout), first_output < total, other=-1
        )
    if num_tokens_post_padded_ptr is not None:
        padded_rows = gl.load(num_tokens_post_padded_ptr)
        total = gl.minimum(total, gl.cdiv(padded_rows, 32) * gl.cdiv(N, BN))
    # ### ASM Note: Keep K bounds in SGPRs; Python constants permit unwanted loop unrolling.
    q = gl.inline_asm_elementwise(
        "s_mov_b32 $0, $1",
        constraints="=s,s",
        args=[K // BK],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    (iteration,) = (0,)
    b_state = 0
    for output in range(
        gl.program_id(0) + TILE_ID,
        total,
        PERSISTENT_GRID if PERSISTENT_GRID > 0 else gl.num_programs(0),
    ):
        row, col = gemm1_pid_grid(
            output, gl.cdiv(total, gl.cdiv(N, BN)), gl.cdiv(N, BN), GROUP_SIZE_M
        )
        bank = iteration & 1
        metadata = meta_mem.index(bank)
        hcu.abarrier_try_wait(4, iteration * (K // BK) // 2 & 1)
        if GROUP_SIZE_M == 1:
            routes, expert = (seed_routes, seed_expert)
        else:
            routes = gl.load(Routes + row * BM + gl.arange(0, BM, layout))
            expert = gl.load(Experts + row + gl.arange(0, 1, layout))
        metadata.slice(0, BM).store(routes)
        metadata.slice(BM, 1).store(expert)
        hcu.wait_lgkmcnt(0)
        hcu.ebarrier_arrive(1, 8)
        expert = gl.sum(metadata.slice(BM, 1).load(layout), 0)
        safe = gl.maximum(0, gl.minimum(expert, E - 1))
        hcu.set_priority(1)
        base = B + safe.to(gl.int64) * N * K + (col * BN).to(gl.int64) * K
        if GROUP_SIZE_M == 1:
            next_output = output + (PERSISTENT_GRID if PERSISTENT_GRID > 0 else gl.num_programs(0))
            next_row = next_output // gl.cdiv(N, BN)
            seed_routes = gl.load(Routes + next_row * BM + gl.arange(0, BM, layout),
                                  next_output < total, other=0)
            seed_expert = gl.load(Experts + next_row + gl.arange(0, 1, layout),
                                  next_output < total, other=-1)
            hcu.sched_barrier()
        for pair in range(q // 2):
            for half in gl.static_range(2):
                k = pair * 2 + half
                _load_b_panel(B, bm, base, k, b_state % 3, b_state // 3 & 1, N, K, 128)
                b_state = _next_b_tickets(b_state, 1)
                if half == 0:
                    if pair == 0:
                        hcu.ebarrier_arrive(3, 8)
                        hcu.ebarrier_sync(4, 8)
                _load_b_panel(B, bm, base + 128 * K, k, b_state % 3, b_state // 3 & 1, N, K, 128)
                b_state = _next_b_tickets(b_state, 1)
                if k + 1 < q:
                    hcu.ebarrier_arrive(3, 8)
                    hcu.ebarrier_sync(4, 8)
                _load_b_panel(B, bm, base + 64 * K, k, b_state % 3, b_state // 3 & 1, N, K, 128)
                b_state = _next_b_tickets(b_state, 1)
                _load_b_panel(B, bm, base + 192 * K, k, b_state % 3, b_state // 3 & 1, N, K, 128)
                b_state = _next_b_tickets(b_state, 1)
        iteration += 1


@gluon.jit
def _consume_panel(
    acc,
    am,
    bm,
    slot,
    phase,
    b_state,
    sync_id,
    arrive_id,
    MMA: gl.constexpr,
    PARTITION: gl.constexpr,
    SLOT: gl.constexpr,
    WARM: gl.constexpr,
    retained_a=None,
    final_k=False,
):
    """Read one N64 panel; retain A registers across both panels of this K256."""
    bslot = b_state % 3
    bphase = (b_state // 3) & 1
    if PARTITION == 0:
        hcu.abarrier_try_wait(7 + bslot, bphase)
        b = bm.index(bslot).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, MMA, 8)))
        hcu.abarrier_try_wait(1 + slot, phase)
        if SLOT == 0:
            a = am.index(slot).load(gl.DotOperandLayout(0, MMA, 8))
        else:
            a = retained_a
    else:
        hcu.abarrier_try_wait(1 + slot, phase)
        if SLOT == 0:
            a = am.index(slot).load(gl.DotOperandLayout(0, MMA, 8))
        else:
            a = retained_a
        if SLOT == 0:
            _consumer_pingpong(sync_id, True, 4)
        hcu.abarrier_try_wait(7 + bslot, bphase)
        b = bm.index(bslot).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, MMA, 8)))
    if bm.dtype == gl.int8:
        a = a.to(gl.int8, bitcast=True)
    # Start with complete waits; enable partial thresholds only after checking
    # this geometry's actual generated DS operations and numerical ordering.
    hcu.sched_barrier()
    hcu.wait_lgkmcnt(0)
    hcu.abarrier_arrive(10 + bslot)
    # The final A operand has no later shared-memory reader: both N quarters
    # reuse its registers. Return only after the complete A/B LDS wait above.
    if SLOT == 0:
        if final_k:
            hcu.abarrier_arrive(4 + slot)
    else:
        if not final_k:
            hcu.abarrier_arrive(4 + slot)
    hcu.sched_barrier()
    if PARTITION == 0:
        if SLOT == 0:
            _consumer_pingpong(arrive_id, False, 4)
        else:
            _consumer_pingpong(sync_id, True, 4)
    hcu.set_priority(1 + 2 * SLOT if PARTITION == 0 else 2 * SLOT)
    hcu.sched_barrier()
    if WARM:
        # ### ASM Note: The optional instruction-cache visit needs labels/branches absent from the Gluon API.
        gl.inline_asm_elementwise(
            _icache_label(PARTITION, SLOT, "mmac") + ":",
            constraints="=s,~{memory}",
            args=[],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    hcu.sched_barrier()
    acc = gl.amd.cdna4.mfma(a, b, acc)
    hcu.sched_barrier()
    if WARM:
        # ### ASM Note: The optional instruction-cache visit needs labels/branches absent from the Gluon API.
        gl.inline_asm_elementwise(
            "s_cbranch_execz " + _icache_label(PARTITION, SLOT, "resume"),
            constraints="=s,~{memory}",
            args=[],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    hcu.sched_barrier()
    hcu.set_priority(3 - PARTITION)
    hcu.sched_barrier()
    if PARTITION == 1 and SLOT == 0:
        _consumer_pingpong(arrive_id, False, 4)
    if SLOT == 1:
        slot, phase = _next_a_slot(slot, phase, am.shape[0])
    b_state = _next_b_tickets(b_state, 2)
    return acc, slot, phase, b_state, a


@gluon.jit
def _consumer(
    C,
    B,
    am,
    bm,
    scales,
    total: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    PARTITION: gl.constexpr,
    WARM: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ID: gl.constexpr,
    num_tokens_post_padded_ptr=None,
    PERSISTENT_GRID: gl.constexpr = 0,
    USE_C64: gl.constexpr = False,
):
    # Load inside each WASP role, as in the shared-ring worker. Capturing a
    # pre-WASP GPU scalar currently requires an unallocated compiler scratch slot.
    """C0/C1: accumulate an N128 half and store scaled sorted output."""
    if num_tokens_post_padded_ptr is not None:
        padded_rows = gl.load(num_tokens_post_padded_ptr)
        total = gl.minimum(total, gl.cdiv(padded_rows, 32) * gl.cdiv(N, BN))
    BM: gl.constexpr = 32
    CN: gl.constexpr = 64
    # Each N64 panel uses MMAC waves [M,N]=[2,2], two N tiles/wave.
    warps: gl.constexpr = (2, 2)
    WM: gl.constexpr = warps[0]
    WN: gl.constexpr = warps[1]
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], 3, [WM, WN], tiles_per_warp=[1, 2])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    wave = hcu.get_wave_id() & 3
    sync_id = 8 + PARTITION * 4 + wave
    arrive_id = 8 + (1 - PARTITION) * 4 + wave
    slot, phase, iteration = 0, 0, 0
    b_state = 0 if PARTITION == 0 else 1
    for output in range(
        gl.program_id(0) + TILE_ID,
        total,
        (PERSISTENT_GRID if PERSISTENT_GRID > 0 else gl.num_programs(0)),
    ):
        row, col = gemm1_pid_grid(
            output, gl.cdiv(total, gl.cdiv(N, BN)), gl.cdiv(N, BN), GROUP_SIZE_M
        )
        if WARM:
            if iteration == 0:
                for stage in gl.static_range(2):
                    if wave == stage * 2:
                        _warm_mmac_instructions(PARTITION, stage)
        acc0 = gl.full(
            (BM, CN), 0, gl.int32 if B.dtype.element_ty == gl.int8 else gl.float32, result
        )
        acc1 = gl.full(
            (BM, CN), 0, gl.int32 if B.dtype.element_ty == gl.int8 else gl.float32, result
        )
        # Initialize FP8 accumulators through the native zero-MMAC path.
        # This computes zeros and is distinct from optional EXEC=0 cache warmup.
        if PARTITION == 0 and B.dtype.element_ty != gl.int8:
            init_mma: gl.constexpr = gl.amd.AMDMFMALayout(
                4, [INST_M, INST_N, INIT_INST_K], MMAC_LAYOUT, [WM, WN], tiles_per_warp=[1, 2]
            )
            zero_a = gl.full(
                (BM, INIT_INST_K), 0, gl.bfloat16, gl.DotOperandLayout(0, init_mma, INIT_K_WIDTH)
            )
            zero_b = gl.full(
                (INIT_INST_K, 2 * CN),
                0,
                gl.bfloat16,
                hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, init_mma, INIT_K_WIDTH)),
            )
            # ### ASM Note: Materialize real zero operands; otherwise folding removes the zero-MMAC initialization.
            zero_a = gl.inline_asm_elementwise(
                "v_mov_b32 $0, $1",
                constraints="=v,v",
                args=[zero_a],
                dtype=gl.bfloat16,
                is_pure=False,
                pack=2,
            )
            # ### ASM Note: Materialize real zero operands; otherwise folding removes the zero-MMAC initialization.
            zero_b = gl.inline_asm_elementwise(
                "v_mov_b32 $0, $1",
                constraints="=v,v",
                args=[zero_b],
                dtype=gl.bfloat16,
                is_pure=False,
                pack=2,
            )
            zero_c = gl.full((BM, 2 * CN), 0, gl.float32, hcu.MmacInterleave2Layout(init_mma))
            initialized = gl.convert_layout(gl.amd.cdna4.mfma(zero_a, zero_b, zero_c), result)
            init_parts = gl.split(initialized.reshape((BM, 2, CN)).permute((0, 2, 1)))
            acc0 = gl.convert_layout(init_parts[0], result)
            acc1 = gl.convert_layout(init_parts[1], result)
        # ### ASM Note: Keep the loop bound/index dynamic in an SGPR; Python constants allow harmful unrolling.
        first = gl.inline_asm_elementwise(
            "s_mov_b32 $0, 0", constraints="=s", args=[], dtype=gl.int32, is_pure=False, pack=1
        )
        gl.assume(first < K // 256)
        for k in range(first, K // 256):
            acc0, slot, phase, b_state, retained_a = _consume_panel(
                acc0,
                am,
                bm,
                slot,
                phase,
                b_state,
                sync_id,
                arrive_id,
                mma,
                PARTITION,
                0,
                WARM,
                final_k=(k == K // 256 - 1),
            )
            acc1, slot, phase, b_state, _ = _consume_panel(
                acc1,
                am,
                bm,
                slot,
                phase,
                b_state,
                sync_id,
                arrive_id,
                mma,
                PARTITION,
                1,
                WARM,
                retained_a,
                final_k=(k == K // 256 - 1),
            )
        hcu.ebarrier_sync(2 if PARTITION == 0 else 0, 8)
        sa = scales.index(iteration & 1).slice(0, BM).load(gl.SliceLayout(1, result))
        sb0 = (
            scales.index(iteration & 1)
            .slice(BN + PARTITION * (2 * CN), CN)
            .load(gl.SliceLayout(0, result))
        )
        sb1 = (
            scales.index(iteration & 1)
            .slice(BN + PARTITION * (2 * CN) + CN, CN)
            .load(gl.SliceLayout(0, result))
        )
        if output + (PERSISTENT_GRID if PERSISTENT_GRID > 0 else gl.num_programs(0)) < total:
            hcu.wait_lgkmcnt(0)
            hcu.ebarrier_arrive(5, 12)
        store_channel_output(C, acc0, sa, sb0, row, 2 * col + PARTITION, N, 128, 0, WM, WN, USE_C64)
        store_channel_output(C, acc1, sa, sb1, row, 2 * col + PARTITION, N, 128, 1, WM, WN, USE_C64)
        iteration += 1


@gluon.jit
def _run(
    A,
    B,
    SA,
    SB,
    C,
    Routes,
    Experts,
    M: gl.constexpr,
    E: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    TOPK: gl.constexpr,
    stride_bse: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    stride_asm: gl.constexpr,
    stride_ask: gl.constexpr,
    TOTAL_TILES: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    USE_ADDR_OFFSET_INT64_A: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_B: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr = False,
    INSN_WARMUP: gl.constexpr = False,
    GROUP_SIZE_M: gl.constexpr = 1,
    PARTITION_REGS: gl.constexpr = (16, 16, 80, 80),
    A_STAGES: gl.constexpr = 2,
    LDS_WRAP: gl.constexpr = 32,
    COPY_BYTES: gl.constexpr = 16,
    TILE_ID: gl.constexpr = 0,
    num_tokens_post_padded_ptr=None,
    PERSISTENT_GRID: gl.constexpr = 0,
):
    """Allocate the 64-KiB A/B pools and launch four WASP roles."""
    BM: gl.constexpr = 32
    BK: gl.constexpr = 256
    BN: gl.constexpr = BLOCK_SIZE_N
    gl.static_assert(BLOCK_SIZE_M == BM and BLOCK_SIZE_K == BK and BN == 256)
    gl.static_assert(K >= 1024 and K % 512 == 0 and N % BN == 0)
    gl.static_assert(A_STAGES == 2)
    gl.static_assert(
        (A.dtype.element_ty.is_fp8() and B.dtype.element_ty.is_fp8())
        or (A.dtype.element_ty == gl.int8 and B.dtype.element_ty == gl.int8)
    )
    al: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    bl: gl.constexpr = hcu.MlsSharedLayout(1, (64, 32), alt=1, order=(0, 1))
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    storage_dtype: gl.constexpr = (
        gl.float8e4nv if A.dtype.element_ty == gl.int8 else A.dtype.element_ty
    )
    am = gl.allocate_shared_memory(storage_dtype, [A_STAGES, BM, BK], al)
    bm = gl.allocate_shared_memory(B.dtype.element_ty, [3, BK, 64], bl)
    # No extra LDS allocation: metadata in retired A0, scales in retired A1.
    meta = am.index(0)._reinterpret(gl.int32, [2, 128], sl)
    scales = am.index(1)._reinterpret(gl.float32, [2, 1024], sl)
    flat_am = am._reinterpret(
        storage_dtype, [A_STAGES, BM, BK], gl.SwizzledSharedLayout(16, 1, 1, [1, 0])
    )
    _initialize(A_STAGES)
    gl.thread_barrier()
    gl.warp_specialize(
        [
            (_empty, ()),
            (
                _produce_b_and_routes,
                (
                    B,
                    Routes,
                    Experts,
                    bm,
                    meta,
                    E,
                    N,
                    K,
                    TOTAL_TILES,
                    BN,
                    GROUP_SIZE_M,
                    TILE_ID,
                    num_tokens_post_padded_ptr,
                    PERSISTENT_GRID,
                ),
            ),
            (
                _produce_a_and_scales,
                (
                    A,
                    SA,
                    SB,
                    flat_am,
                    meta,
                    scales,
                    M,
                    E,
                    N,
                    K,
                    stride_bse,
                    stride_bsn,
                    stride_asm,
                    TOTAL_TILES,
                    BN,
                    GROUP_SIZE_M,
                    TILE_ID,
                    num_tokens_post_padded_ptr,
                    PERSISTENT_GRID,
                    USE_ADDR_OFFSET_INT64_A,
                ),
            ),
            (
                _consumer,
                (
                    C,
                    B,
                    am,
                    bm,
                    scales,
                    TOTAL_TILES,
                    M,
                    N,
                    K,
                    BN,
                    0,
                    INSN_WARMUP,
                    GROUP_SIZE_M,
                    TILE_ID,
                    num_tokens_post_padded_ptr,
                    PERSISTENT_GRID,
                    USE_ADDR_OFFSET_INT64_C,
                ),
            ),
            (
                _consumer,
                (
                    C,
                    B,
                    am,
                    bm,
                    scales,
                    TOTAL_TILES,
                    M,
                    N,
                    K,
                    BN,
                    1,
                    INSN_WARMUP,
                    GROUP_SIZE_M,
                    TILE_ID,
                    num_tokens_post_padded_ptr,
                    PERSISTENT_GRID,
                    USE_ADDR_OFFSET_INT64_C,
                ),
            ),
        ],
        [4, 4, 4, 4],
        PARTITION_REGS,
    )
    for id in gl.static_range(16):
        hcu.abarrier_invalidate(id)


def is_persistent_channel_gemm1_supported(
    a,
    b,
    a_scale,
    b_scale,
    *,
    arch,
    cu_count,
    top_k,
    config,
    c_sorted=True,
    a_sorted=False,
    output_dtype=None,
    ck_sorting=True,
    gemm_type="gemm1",
    scale_recipe="fp8_channel",
    has_bias=False,
    mul_routed_weight=False,
):
    """Query geometry/input compatibility; unsupported cases use ordinary Triton."""
    import torch

    if arch != "gfx946" or not isinstance(cu_count, int) or cu_count <= 0:
        return False
    if (
        gemm_type != "gemm1"
        or scale_recipe not in ("fp8_channel", "int8_channel")
        or not c_sorted
        or a_sorted
        or not ck_sorting
        or has_bias
        or mul_routed_weight
    ):
        return False
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if not (
        (a.dtype in fp8 and b.dtype in fp8 and scale_recipe == "fp8_channel")
        or (a.dtype == b.dtype == torch.int8 and scale_recipe == "int8_channel")
    ):
        return False
    if output_dtype is None:
        output_dtype = torch.bfloat16
    if (
        a_scale.dtype != torch.float32
        or b_scale.dtype != torch.float32
        or output_dtype not in (torch.bfloat16, torch.float16)
    ):
        return False
    if a.ndim != 2 or b.ndim != 3 or not a.is_contiguous() or not b.is_contiguous():
        return False
    m, k = a.shape
    e, n, bk = b.shape
    if min(m, e, n, k) <= 0 or bk != k or not 0 < top_k <= 256:
        return False
    if a_scale.shape not in ((m,), (m, 1)) or b_scale.shape not in ((e, n), (e, n, 1)):
        return False
    if (config.get("BLOCK_SIZE_M"), config.get("BLOCK_SIZE_N"), config.get("BLOCK_SIZE_K")) != (
        32,
        256,
        256,
    ):
        return False
    return m == 128 and n % 256 == 0 and k >= 1024 and k % 512 == 0
