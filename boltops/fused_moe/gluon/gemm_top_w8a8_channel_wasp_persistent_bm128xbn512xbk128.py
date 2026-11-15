# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent FP8/INT8 channel GEMM1 for gfx946, BM128/BN512/BK128.

CK-sorted routes select rows of contiguous A[M,K]; B[E,N,K] is K-contiguous.
FP32 channel scales; BF16/FP16 sorted output. Requires K>=512, K%512=0 and
N%512=0. Each P0/P1/C0/C1 role owns four waves; grid is caller-supplied.

    P0: B-left + B-right -> C0/C1 through three shared B FIFO slots
    P1: A -> C0 + C1 through two A slots; publishes channel scales at tail
    C0: left N256; C1: right N256. Both retain A across two N128 panels.

For each K256 pair, consumers read A0/A1 once, compute N panel0, then reuse
those registers for N panel1. Each B slot holds [K256,N128]; both K128 reads
must finish before EMPTY. B issue order is left/right panel0, then left/right
panel1. EB3/4 order A bursts between B bursts.
A2 x 16 KiB + B3 x 32 KiB fill 128 KiB LDS. Metadata aliases A slot0;
tail scales alias A slot1, protected by EMPTY and EB5. Tickets continue across
outputs. Native output fragments preserve wave ownership, without LDS exchange.
INT8 accumulates in INT32 and scales in FP32. Warmup is disabled by default.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
from triton.experimental.gluon.language.amd.cdna4 import async_copy

GIB = gl.constexpr(1024 * 1024 * 1024)
MMAC_LAYOUT = gl.constexpr(3)
INST_M = gl.constexpr(16)
INST_N = gl.constexpr(16)
INST_K = gl.constexpr(32)
INIT_INST_K = gl.constexpr(16)
INIT_K_WIDTH = gl.constexpr(4)
DOT_K_WIDTH = gl.constexpr(8)
WAVE_SIZE = gl.constexpr(64)
WAVES = gl.constexpr(4)
PRODUCER_WAVES = gl.constexpr(4)
CONSUMERS = gl.constexpr(2)
ABARRIER_FIRST = gl.constexpr(0)
A_READY = gl.constexpr(1)
DEFAULT_LAUNCH_OPTIONS = dict(
    num_warps=4, num_stages=2, use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4), wasp_wdra=True,
)


@gluon.constexpr_function
def _matrix_store_cast_layout(layout):
    # Native store packs each eight 16-bit elements as 0,4,1,5,2,6,3,7.
    # Rotate only register bases, preserving lane/warp ownership and addresses.
    """Pair native store halfwords while preserving lane/wave ownership."""
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:],
        layout.lane_bases, layout.warp_bases, layout.block_bases, layout.shape)


@gluon.jit
def _cast_for_matrix_store(value, dtype: gl.constexpr, result: gl.constexpr):
    """Reorder registers, narrow to 16 bits, then restore native store layout."""
    linear: gl.constexpr = gl.to_linear_layout(value.type.layout, value.shape)
    packing: gl.constexpr = _matrix_store_cast_layout(linear)
    value = gl.convert_layout(value, packing)
    value = value.to(dtype)
    return gl.convert_layout(value, result)


@gluon.jit
def store_channel_output(C, acc, sa, sb, block_m, block_n,
                         N: gl.constexpr, BN: gl.constexpr,
                         PARTITION: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr,
                         USE_C64: gl.constexpr):
    """Scale one consumer's [128,256] result and store sorted C.

            acc [128,256] + SA [128] -> four M32 groups
            each group -> N128 halves -> N32 fragments -> scale/cast/store

        Removing N bit5 exposes a native two-MMAC-tile fragment. Its N-wave
        stride is32 instead of64; wave_n*32 in the base preserves ownership.
        These are register views; no LDS or cross-wave conversion is introduced."""
    gl.static_assert(BN == 512 and WM == 2 and WN == 2 and N % BN == 0)
    native: gl.constexpr = hcu.MmacInterleave2Layout(gl.amd.AMDMFMALayout(
        4, [16,16,32], 3, [2,2], tiles_per_warp=[1,2]))
    # Original TN4 N-warp stride is 64 columns; the native TN2 fragment
    # has stride 32. Compensate in the per-wave global base, retaining
    # all values in their original wave (no LDS or cross-wave exchange).
    wave_n = hcu.get_wave_id() & 1
    if USE_C64:
        C = C + block_m.to(gl.int64) * 128 * N + wave_n * 32
    else:
        C = C + block_m * 128 * N + wave_n * 32
    if acc.dtype == gl.int32:
        acc = acc.to(gl.float32)
    m2 = gl.split(acc.reshape((2,64,256)).permute((1,2,0)))
    a2 = gl.split(sa.reshape((2,64)).trans())
    m40 = gl.split(m2[0].reshape((2,32,256)).permute((1,2,0)))
    m41 = gl.split(m2[1].reshape((2,32,256)).permute((1,2,0)))
    a40 = gl.split(a2[0].reshape((2,32)).trans())
    a41 = gl.split(a2[1].reshape((2,32)).trans())
    sb2 = gl.split(sb.reshape((2,128)).trans())
    for mi in gl.static_range(4):
        row = m40[mi] if mi < 2 else m41[mi-2]
        scale_a = a40[mi] if mi < 2 else a41[mi-2]
        row2 = gl.split(row.reshape((32,2,128)).permute((0,2,1)))
        sa_part = gl.convert_layout(scale_a, gl.SliceLayout(1,native))
        for ni in gl.static_range(2):
            # Remove N bit5, keeping N bit6 as the warp coordinate.
            pieces = gl.split(row2[ni].reshape((32,2,2,32)).permute((0,1,3,2)).reshape((32,64,2)))
            scales = gl.split(sb2[ni].reshape((2,2,32)).permute((0,2,1)).reshape((64,2)))
            for half in gl.static_range(2):
                part = gl.convert_layout(pieces[half], native)
                sb_part = gl.convert_layout(scales[half], gl.SliceLayout(0,native))
                value = _cast_for_matrix_store(part * sa_part[:,None] * sb_part[None,:], C.dtype.element_ty,native)
                hcu.matrix_store(C,value,(128,N),(N,1),
                                 (mi*32,block_n*BN+PARTITION*256+ni*128+half*32),
                                 tile=(16,32),boundary_check=())


@gluon.constexpr_function
def abarrier_count(a_stages):
    # Initialize the complete 16-ID hardware namespace, including reserved IDs.
    return 16


@gluon.constexpr_function
def abarrier_init_asm(a_stages):
    # A READY1/2, EMPTY8/9; shared B READY4/5/6, EMPTY11/12/13.
    lines = []
    for barrier_id in range(16):
        count = 8 if barrier_id >= 8 else 4
        lines.append(f"s_abarrier_init {barrier_id}, {count}")
    return "\n\t".join(lines) + "\n\ts_mov_b32 $0, 0"


@gluon.constexpr_function
def _buffer_lds_layout(rows, bk, element_bytes, waves, wrap_step, copy_bytes):
    """Build the shared-memory layout for a K-contiguous [rows, bk] tile.

    Interleave row chunks from the producer waves in SharedLinearLayout,
    then describe the buffer-to-LDS placement with BufferLdsSharedLayout.
    The consumer uses the same layout to find the copied data in LDS.

    waves is the producer wave count; copy_bytes is bytes copied per lane
    (0 selects the fallback below); wrap_step is the wave wrap step in bytes.
    This constructs the requested layout; it does not search for an optimum.
    """
    # FP8 uses 32 lanes per row chunk; wider elements use a full wave.
    lanes_per_row = WAVE_SIZE.value // 2 if element_bytes == 1 else WAVE_SIZE.value
    copy_bytes = copy_bytes or element_bytes * bk // lanes_per_row
    chunk = WAVE_SIZE.value // lanes_per_row
    bases = []
    tile_bytes = rows * bk * element_bytes
    for index in range((rows * bk).bit_length() - 1):
        wave, local = divmod((1 << index) * element_bytes, tile_bytes // waves)
        local_row, column_byte = divmod(local, bk * element_bytes)
        row = (local_row // chunk * waves + wave) * chunk + local_row % chunk
        bases.append([row, column_byte // element_bytes])
    return hcu.BufferLdsSharedLayout(
        gl.SharedLinearLayout(bases), element_bytes, 1, rows, bk * element_bytes, waves, copy_bytes,
        chunk, waves, wrap_step
    )


@gluon.jit
def _load_a_tile(A, am, a_offsets, a_mask, BK: gl.constexpr, tile, first_k_tile,
    WAIT_EMPTY: gl.constexpr = True, WAIT_GATE: gl.constexpr = False, USE_A64: gl.constexpr = False):
    """P1 waits/gates one K128 A copy into slot=(first_k_tile+tile)&1.

            ticket = first_k_tile + tile; phase = (ticket >> 1) & 1
            EMPTY = 8 + slot    P1 waits; C0/C1 each return one credit per wave
            READY = 1 + slot   P1 publishes after the copy

        EB3 grants the A burst; only the burst raises P1 to priority2. Narrow
        addresses use buffer-to-LDS; wide addresses use two packed K64 writes
        to limit temporary VGPRs. Both paths preserve raw FP8/INT8 bytes."""
    ticket = first_k_tile + tile
    slot = ticket & 1
    phase = (ticket >> 1) & 1
    if WAIT_EMPTY:
        hcu.abarrier_try_wait(8 + slot, phase)
    ready = 1 + slot
    dest = am.index(slot)
    memory_tile = tile
    source = A + gl.cast(memory_tile, gl.int64) * BK if USE_A64 else A + memory_tile * BK
    # Empty tied-register asm materializes the pointer before the issue gate;
    # it emits no instruction and leaves register assignment to LLVM.
    # ### ASM Note: Gluon cannot pin source preparation before EB3; tied SGPR ASM emits no instruction.
    source = gl.inline_asm_elementwise(
        "", constraints="=s,0", args=[source.to(gl.uint64)],
        dtype=gl.uint64, is_pure=False, pack=1).to(A.dtype)
    source = gl.multiple_of(source, 16)
    if WAIT_GATE:
        hcu.ebarrier_sync(3, 2 * WAVES)
    hcu.sched_barrier()
    # Only the exclusive A-copy burst needs elevated issue priority.
    # Pointer, ticket and EMPTY preparation compete equally with P0.
    hcu.set_priority(2)
    hcu.sched_barrier()
    hcu.abarrier_seq(ready)
    if USE_A64:
        packed_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 1, [1, 0])
        al: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0])
        ak = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
        packed = dest._reinterpret(gl.int32, [128, BK // 4], packed_layout)
        # Two K64 writes bound temporary VGPRs while retaining a single A slot.
        for part in gl.static_range(2):
            offsets = a_offsets[:, None] + part * (BK // 8) + ak[None, :]
            value = gl.load(source.to(gl.pointer_type(gl.int32)) + offsets, a_mask[:, None], other=0)
            packed.slice(part * (BK // 8), BK // 8, dim=1).store(value)
            hcu.wait_lgkmcnt(0)
            hcu.sched_barrier()
    else:
        async_copy.buffer_load_to_shared(dest, source.to(gl.pointer_type(dest.dtype)), a_offsets, a_mask)
    hcu.abarrier_arrive(ready)
    hcu.sched_barrier()
    hcu.set_priority(1)
    hcu.sched_barrier()


@gluon.jit
def _empty():
    """Keep the default WASP partition empty; P0/P1/C0/C1 own the work."""
    pass


@gluon.jit
def _prepare_b_load(B, bm, expert, block_n, N: gl.constexpr, K: gl.constexpr,
                    BN: gl.constexpr, BK: gl.constexpr, PARTITION: gl.constexpr,
                    tile, first_k_tile):
    """Prepare one B FIFO load; no data is copied here.

            ticket = 2*(first_k_tile+tile) + PARTITION
            slot = ticket % 3; phase = (ticket // 3) & 1
            READY = 4 + slot; EMPTY = 11 + slot

        PARTITION selects C0/C1's N256 half; tile&1 selects its N128 panel.
        tile//2 selects K256. Fold expert/N/K origins into the 64-bit base."""
    ticket = 2 * (first_k_tile + tile) + PARTITION
    slot = ticket % 3
    phase = (ticket // 3) & 1
    ready = 4 + slot
    empty = 11 + slot
    col = block_n * BN + PARTITION * (BN // 2) + (tile & 1) * 128
    source = B + expert.to(gl.int64) * N * K + col.to(gl.int64) * K + gl.cast(tile // 2, gl.int64) * 256
    # ### ASM Note: Tied SGPR ASM fixes pointer preparation before the gate; Gluon has no such placement constraint.
    source = gl.inline_asm_elementwise(
        "", constraints="=s,0", args=[source.to(gl.uint64)],
        dtype=gl.uint64, is_pure=False, pack=1).to(B.dtype)
    return bm.index(slot), source, phase, ready, empty


@gluon.jit
def _load_b_tile(dest, source, phase, ready, empty, N: gl.constexpr,
                  K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    """Wait B EMPTY, issue one [K256,N128] MLS load, then publish READY."""
    hcu.abarrier_try_wait(empty, phase)
    hcu.abarrier_seq(ready)
    hcu.matrix_load(dest, source, (K, N), (1, K), (0, 0), warps=(2, 2))
    hcu.abarrier_arrive(ready)


@gluon.jit
def _producer(A, B, SA, SB, Sorted, Experts, expert_mem, am, bm, sam, sbm, route_mem, SHARE_ROUTES: gl.constexpr,
             block_m, block_n, M, E, N: gl.constexpr, K: gl.constexpr,
             TOPK: gl.constexpr, stride_bse, stride_bsn: gl.constexpr, stride_asm: gl.constexpr,
             BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, WAVES: gl.constexpr,
             PARTITION: gl.constexpr, USE_MLS_LOAD: gl.constexpr, A_STAGES: gl.constexpr, first_k_tile=0, preissued_b_left=False, USE_A64: gl.constexpr = False, USE_B64: gl.constexpr = False):
    # P0 owns the complete B FIFO; P1 owns A and publishes tail scales.
    # Startup metadata aliases A slot0; both producers read it before A0 reuse.
    """P0 streams both B halves; P1 loads A and publishes tail scales.

            Startup:      P0 B-left0 -> EB3 -> P1 A0 -> EB4
            Each B tile:  P0 B-right -> optional A issue window -> next B-left
            Each K256:    P1 issues A0/A1 only for the first N panel
            Tail:         P1 wait A EMPTY -> store SA/SB -> notify C0/C1

        CK low24 token bits select A and SA. Tail scales alias A slot1; old
        readers finish before the next output writes A1 (EB5)."""
    layout: gl.constexpr = gl.BlockedLayout([1], [WAVE_SIZE], [WAVES], [0])
    expert = gl.sum(expert_mem.load(layout), 0)
    safe_expert = gl.maximum(0, gl.minimum(expert, E - 1))
    a_slot = first_k_tile % A_STAGES
    a_phase = (first_k_tile // A_STAGES) & 1
    if PARTITION == 1:
        if USE_A64:
            al: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0])
            copy_k: gl.constexpr = BK // 4
            copy_stride: gl.constexpr = K // 4
        else:
            al: gl.constexpr = am.layout.producer_layout()
            copy_k: gl.constexpr = BK
            copy_stride: gl.constexpr = K
        routes = route_mem.load(gl.SliceLayout(1, al)).to(gl.uint32) & 0x00ffffff
        valid = (routes < M) & (expert >= 0) & (expert < E)
        ak = gl.arange(0, copy_k, gl.SliceLayout(0, al))
        if USE_A64:
            a_offsets = routes.to(gl.int64) * copy_stride
        else:
            a_offsets = routes[:, None] * copy_stride + ak[None, :]
        if USE_A64:
            a_mask = valid
        else:
            a_mask = valid[:, None] & gl.full((BM, copy_k), True, gl.int1, al)
        # Retain one-dimensional route IDs/scale offsets in registers. These
        # values belong to this output and need no global reload after K.
        scale_routes = route_mem.load(layout).to(gl.uint32) & 0x00ffffff
        scale_valid = (scale_routes < M) & (expert >= 0) & (expert < E)
        sa_offsets = scale_routes.to(gl.int64) * stride_asm if USE_A64 else (scale_routes * stride_asm).to(gl.uint32)
        scale_rn = block_n * BN + gl.arange(0, BN, layout)
        sb_base = SB + safe_expert.to(gl.int64) * stride_bse
        sb_offsets = scale_rn.to(gl.int64) * stride_bsn if USE_B64 else (scale_rn * stride_bsn).to(gl.uint32)
    hcu.ebarrier_sync(3, 2 * WAVES)
    # ### ASM Note: An ordinary constant may unroll the loop; MOV makes the bound opaque to LLVM.
    num_k_tiles = gl.inline_asm_elementwise(
        "s_mov_b32 $0, $1", constraints="=s,s", args=[K // BK],
        dtype=gl.int32, is_pure=False, pack=1)
    # A READY dominates the control's long consumer waits. Give the A
    # producer the same issue priority as B; no producer handshake is added.
    hcu.sched_barrier()
    hcu.set_priority(1)
    hcu.sched_barrier()
    # Prologue: Bleft(0) -> A(0). P0 owns B; P1 owns A at priority 2.
    # EB3 grants an exclusive A issue window; EB4 closes it after all A waves.
    # The loop is Bright(k) -> A(k+1) -> Bleft(k+1), with no A(K/BK).
    if PARTITION == 0:
        if not preissued_b_left:
            left, source, phase, ready, empty = _prepare_b_load(
                B, bm, safe_expert, block_n, N, K, BN, BK, 0, 0, first_k_tile)
            _load_b_tile(left, source, phase, ready, empty, N, K, BN, BK)
        hcu.sched_barrier()
        hcu.ebarrier_arrive(3, 2 * WAVES)
        hcu.sched_barrier()
        hcu.ebarrier_sync(4, 2 * WAVES)
        hcu.sched_barrier()
        # Two N panels share one A0/A1 register pair. Only the first panel
        # needs fresh A; the second panel advances B without an A gate.
        for tile in range(num_k_tiles):
            right, source, phase, ready, empty = _prepare_b_load(
                B, bm, safe_expert, block_n, N, K, BN, BK, 1, tile, first_k_tile)
            _load_b_tile(right, source, phase, ready, empty, N, K, BN, BK)
            if (tile & 1) == 0:
                hcu.sched_barrier()
                hcu.ebarrier_arrive(3, 2 * WAVES)
                hcu.sched_barrier()
                hcu.ebarrier_sync(4, 2 * WAVES)
                hcu.sched_barrier()
            if tile + 1 < num_k_tiles:
                left, source, phase, ready, empty = _prepare_b_load(
                    B, bm, safe_expert, block_n, N, K, BN, BK, 0, tile + 1, first_k_tile)
                _load_b_tile(left, source, phase, ready, empty, N, K, BN, BK)
                if (tile & 1) == 1:
                    hcu.sched_barrier()
                    hcu.ebarrier_arrive(3, 2 * WAVES)
                    hcu.sched_barrier()
                    hcu.ebarrier_sync(4, 2 * WAVES)
                    hcu.sched_barrier()
    else:
        # A0 EMPTY also separates the previous scale wait's ID9 from A1 ID9.
        hcu.abarrier_try_wait(8 + (first_k_tile & 1), (first_k_tile >> 1) & 1)
        _load_a_tile(A, am, a_offsets, a_mask, BK, 0, first_k_tile, False, True, USE_A64)
        hcu.sched_barrier()
        hcu.ebarrier_arrive(4, 2 * WAVES)
        hcu.sched_barrier()
        for tile in range(1, num_k_tiles):
            ticket = first_k_tile + tile
            hcu.abarrier_try_wait(8 + (ticket & 1), (ticket >> 1) & 1)
            if first_k_tile > 0 and tile == 1:
                # B slots are independent of scales; protect only next A1.
                hcu.ebarrier_sync(5, 2 * WAVES + CONSUMERS * WAVES)
            _load_a_tile(A, am, a_offsets, a_mask, BK, tile, first_k_tile, False, True, USE_A64)
            hcu.sched_barrier()
            hcu.ebarrier_arrive(4, 2 * WAVES)
            hcu.sched_barrier()
    if PARTITION == 1:
        # Tail scales reuse A slot1, whose last reader is A(Q-1).
        last_ticket = 3 * (first_k_tile + K // BK) - 1
        # Prefetch into registers before waiting for the old A/B owner.
        # Only the following LDS stores require A slot1 EMPTY permission.
        if USE_A64:
            scales_a = gl.load(SA + sa_offsets, scale_valid, other=0)
        else:
            scales_a = buffer_load(SA, sa_offsets, scale_valid, other=0)
        if USE_B64:
            scales_b = gl.load(sb_base + sb_offsets, scale_rn < N, other=0)
        else:
            scales_b = buffer_load(sb_base, sb_offsets, scale_rn < N, other=0)
        hcu.sched_barrier()
        # A0 and A1 must have no outstanding readers before metadata/scale aliasing.
        # Last A copy waits EMPTY9; visit 8 then 9 to preserve distinct wait IDs.
        next_phase = ((first_k_tile + K // BK) >> 1) & 1
        hcu.abarrier_try_wait(8, next_phase)
        hcu.abarrier_try_wait(9, next_phase)
        sam.store(scales_a)
        sbm.store(scales_b)
        # Ordinary LDS stores must complete before notifying the output readers.
        hcu.wait_lgkmcnt(0)
        # Independent readers: C0 need not wait for C1's last MMAC.
        # The explicit LDS wait above publishes scales before either signal.
        hcu.ebarrier_arrive(2, 2 * WAVES)
        hcu.ebarrier_arrive(0, 2 * WAVES)


@gluon.constexpr_function
def _icache_label(partition, slot, kind):
    """Create a unique per-consumer/slot warmup label."""
    return f".Lmoe_iwarm_c{partition}_b{slot}_{kind}"


@gluon.jit
def _warm_mmac_instructions(PARTITION: gl.constexpr, SLOT: gl.constexpr):
    # Visit the real MMAC block with EXEC=0, skipping LDS and handoff code.
    # Ping/pong are visited concurrently by separate waves; a shared consumer
    # rendezvous below waits for both visits before reusing the B ring.
    # Symbolic labels avoid fixed code addresses. Normal compute pays one
    # untaken EXEC=0 return guard per BK; scalar temporaries are allocated.
    """Visit the real MMAC block with EXEC=0; restore EXEC before normal work."""
    # ### ASM Note: Gluon has no EXEC-save or branch-to-MMAC-label operation; ASM implements the optional warmup.
    saved_exec = gl.inline_asm_elementwise(
        "s_mov_b64 $0, exec\n\ts_mov_b64 exec, 0", constraints="=&s,~{exec},~{memory}",
        args=[], dtype=gl.uint64, is_pure=False, pack=1)
    # ### ASM Note: Gluon has no EXEC-save or branch-to-MMAC-label operation; ASM implements the optional warmup.
    gl.inline_asm_elementwise(
        "s_branch " + _icache_label(PARTITION, SLOT, "mmac") + "\n"
        + _icache_label(PARTITION, SLOT, "resume") + ":",
        constraints="=s,~{memory}", args=[], dtype=gl.uint32, is_pure=False, pack=1)
    # ### ASM Note: Gluon has no EXEC-save or branch-to-MMAC-label operation; ASM implements the optional warmup.
    gl.inline_asm_elementwise(
        "s_mov_b64 exec, $1", constraints="=s,s,~{exec},~{memory}", args=[saved_exec],
        dtype=gl.uint32, is_pure=False, pack=1)


@gluon.jit
def _initialize(PRODUCER_WAVES: gl.constexpr, WAVES: gl.constexpr, A_STAGES: gl.constexpr):
    """Initialize READY/EMPTY before splitting into the four roles.

            Buffer       Slots      READY       EMPTY       Consumer credits
            A K128       0/1        1/2         8/9         C0 x4 + C1 x4
            B K256,N128  0/1/2      4/5/6       11/12/13    its reader: 4 x2

        Each READY counts four producer waves. Startup grants eight EMPTY credits
        per slot. IDs 0/3/7/10/14/15 are reserved. Scale READY uses EB2 for C0 and
        EB0 for C1; EB5 protects old scale reads before next-output A1 reuse."""
    # ### ASM Note: Builtin init adds a rendezvous per ID; batch ASM needs only the one barrier below.
    gl.inline_asm_elementwise(abarrier_init_asm(A_STAGES), constraints="=s", args=[], dtype=gl.int32,
        is_pure=False, pack=1)
    gl.thread_barrier()
    for slot in gl.static_range(8):
        hcu.abarrier_arrive(8 + slot, 2)


@gluon.jit
def _consumer_pingpong(ID, WAIT: gl.constexpr, WAVES: gl.constexpr):
    # One wave from each consumer contributes to each of the four groups.
    """Pass the MMAC handoff between matching SIMD waves of C0/C1."""
    hcu.sched_barrier()
    if WAIT:
        hcu.ebarrier_sync(ID, 2)
    else:
        hcu.ebarrier_arrive(ID, 2)
    hcu.sched_barrier()


@gluon.jit
def _consume_retained(acc, am, bm, a0, a1, a_ticket, b_ticket, sync_id, arrive_id,
                      MMA: gl.constexpr, PARTITION: gl.constexpr, SLOT: gl.constexpr,
                      WAVES: gl.constexpr, PINGPONG: gl.constexpr,
                      INSN_WARMUP: gl.constexpr, N_PANEL: gl.constexpr):
    """Consume one K128 half; retain A for the second N128 panel.

            C0: wait/read B -> wait/read A (panel0 only) -> release -> MMAC
            C1: wait/read A -> ping handoff -> wait/read B -> release -> MMAC
            Panel1: both reuse a0/a1 registers instead of reading A again.

        B EMPTY is returned only after SLOT1, when both K halves have been read.
        For panel0/SLOT1, split LDS waits release the earlier operand first:
        C0 B-before-A: lgkmcnt(8) -> B EMPTY -> lgkmcnt(0) -> A EMPTY.
        C1 A-before-B: lgkmcnt(6) -> A EMPTY -> lgkmcnt(0) -> B EMPTY."""
    a_slot = a_ticket & 1
    a_phase = (a_ticket >> 1) & 1
    b_slot = b_ticket % 3
    b_phase = (b_ticket // 3) & 1
    operand_b: gl.constexpr = hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, MMA, DOT_K_WIDTH))
    if PARTITION == 0:
        if SLOT == 0:
            hcu.abarrier_try_wait(4 + b_slot, b_phase)
        b = bm.index(b_slot).slice(SLOT * 128, 128).load(operand_b)
        hcu.sched_barrier()
        if N_PANEL == 0:
            hcu.abarrier_try_wait(1 + a_slot, a_phase)
            a = am.index(a_slot).load(gl.DotOperandLayout(0, MMA, DOT_K_WIDTH))
            if acc.dtype == gl.int32:
                a = a.to(gl.int8, bitcast=True)
        else:
            a = a0 if SLOT == 0 else a1
    else:
        if N_PANEL == 0:
            hcu.abarrier_try_wait(1 + a_slot, a_phase)
            a = am.index(a_slot).load(gl.DotOperandLayout(0, MMA, DOT_K_WIDTH))
            if acc.dtype == gl.int32:
                a = a.to(gl.int8, bitcast=True)
        else:
            a = a0 if SLOT == 0 else a1
        if PINGPONG and SLOT == 0:
            _consumer_pingpong(sync_id, True, WAVES)
        if SLOT == 0:
            hcu.abarrier_try_wait(4 + b_slot, b_phase)
        b = bm.index(b_slot).slice(SLOT * 128, 128).load(operand_b)
    hcu.sched_barrier()
    # All accesses to a B slot finish only after the second K128 half read.
    # A EMPTY tracks LDS read completion, independently of later REG reuse.
    if N_PANEL == 0:
        if SLOT == 1 and PARTITION == 0:
            hcu.wait_lgkmcnt(8)
            # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
            gl.inline_asm_elementwise("s_abarrier_arrive $0, $1, 2", constraints="=&s,s,~{memory}",
                args=[11 + b_slot], dtype=gl.int32, is_pure=False, pack=1)
            hcu.wait_lgkmcnt(0)
            # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
            gl.inline_asm_elementwise("s_abarrier_arrive $0, $1, 1", constraints="=&s,s,~{memory}",
                args=[8 + a_slot], dtype=gl.int32, is_pure=False, pack=1)
        elif SLOT == 1 and PARTITION == 1:
            hcu.wait_lgkmcnt(6)
            # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
            gl.inline_asm_elementwise("s_abarrier_arrive $0, $1, 1", constraints="=&s,s,~{memory}",
                args=[8 + a_slot], dtype=gl.int32, is_pure=False, pack=1)
            hcu.wait_lgkmcnt(0)
            # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
            gl.inline_asm_elementwise("s_abarrier_arrive $0, $1, 2", constraints="=&s,s,~{memory}",
                args=[11 + b_slot], dtype=gl.int32, is_pure=False, pack=1)
        else:
            hcu.wait_lgkmcnt(0)
            # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
            gl.inline_asm_elementwise("s_abarrier_arrive $0, $1, 1", constraints="=&s,s,~{memory}",
                args=[8 + a_slot], dtype=gl.int32, is_pure=False, pack=1)
        if SLOT == 0:
            a0 = a
        else:
            a1 = a
    else:
        hcu.wait_lgkmcnt(0)
        if SLOT == 1:
            # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
            gl.inline_asm_elementwise("s_abarrier_arrive $0, $1, 2", constraints="=&s,s,~{memory}",
                args=[11 + b_slot], dtype=gl.int32, is_pure=False, pack=1)
    hcu.sched_barrier()
    if PINGPONG and PARTITION == 0:
        if SLOT == 0:
            _consumer_pingpong(arrive_id, False, WAVES)
        else:
            _consumer_pingpong(sync_id, True, WAVES)
    priority: gl.constexpr = 1 + 2 * SLOT if PARTITION == 0 else 2 * SLOT
    hcu.set_priority(priority)
    hcu.sched_barrier()
    if INSN_WARMUP:
        # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
        gl.inline_asm_elementwise(_icache_label(PARTITION, 2 * N_PANEL + SLOT, "mmac") + ":",
            constraints="=s,~{memory}", args=[], dtype=gl.uint32, is_pure=False, pack=1)
    hcu.sched_barrier()
    acc = gl.amd.cdna4.mfma(a, b, acc)
    hcu.sched_barrier()
    if INSN_WARMUP:
        # ### ASM Note: Builtin arrive drains all LDS reads; ASM preserves the explicit split wait. Warmup labels also require ASM.
        gl.inline_asm_elementwise("s_cbranch_execz " + _icache_label(PARTITION, 2 * N_PANEL + SLOT, "resume"),
            constraints="=s,~{memory}", args=[], dtype=gl.uint32, is_pure=False, pack=1)
    hcu.sched_barrier()
    hcu.set_priority(3 - PARTITION)
    hcu.sched_barrier()
    if PINGPONG and PARTITION == 1 and SLOT == 0:
        _consumer_pingpong(arrive_id, False, WAVES)
    return acc, a0, a1


@gluon.jit
def _consumer(C, Sorted, Experts, am, bm, sam, sbm, route_mem, SHARE_ROUTES: gl.constexpr, block_m, block_n,
              M, E, N: gl.constexpr, K: gl.constexpr, TOPK: gl.constexpr,
              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
              WAVES: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr,
              MMA_TILES_PER_WAVE_N: gl.constexpr,
              PARTITION: gl.constexpr, PINGPONG: gl.constexpr, USE_MLS_LOAD: gl.constexpr,
              A_STAGES: gl.constexpr, INSN_WARMUP: gl.constexpr,
              first_k_tile=0, output_phase=0, warmup_this_tile=True, has_next_output=False,
              USE_C64: gl.constexpr = False, IS_INT8: gl.constexpr = False):
    """Keep a full [128,256] accumulator and stream native output fragments.

            acc [128,256] -> two N128 accumulators
            each K256:    read A0/A1 for N0 -> reuse A0/A1 for N1
            tail:         join acc -> wait scales -> native fragment stores

        A0/A1 are register copies after EMPTY, so P1 may overwrite retired LDS.
        Each native store preserves the original N-wave ownership."""
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        4, [INST_M, INST_N, INST_K], MMAC_LAYOUT,
        [WM, WN], tiles_per_warp=[1, MMA_TILES_PER_WAVE_N])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    CM: gl.constexpr = BM
    CN: gl.constexpr = BN // CONSUMERS
    col: gl.constexpr = PARTITION * CN
    if INSN_WARMUP:
        # Instruction cache is shared across the consumer's four waves.
        # Its fixed 4-wave partition starts at a multiple of WAVES.
        if warmup_this_tile:
            wave = hcu.get_wave_id() % WAVES
            for slot in gl.static_range(4):
                if wave == slot:
                    _warm_mmac_instructions(PARTITION, slot)
    ACC: gl.constexpr = gl.int32 if IS_INT8 else gl.float32
    acc = gl.full((CM, CN), 0, ACC, result)
    if PARTITION == 0 and not IS_INT8:
        # BF16 MMACs distinguish zero initialization from FP8 GEMM in traces.
        # M/N ownership and LTS stay identical; BF16 has instruction K=16.
        init_mma: gl.constexpr = gl.amd.AMDMFMALayout(
            4, [INST_M, INST_N, INIT_INST_K], MMAC_LAYOUT,
            [WM, WN], tiles_per_warp=[1, MMA_TILES_PER_WAVE_N])
        # Distinct zero groups prevent CSE from replacing MMACs with MOV copies.
        zero_a = gl.full((CM, INIT_INST_K), 0, gl.bfloat16, gl.DotOperandLayout(0, init_mma, INIT_K_WIDTH))
        zero_b = gl.full((INIT_INST_K, CN), 0, gl.bfloat16,
            hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, init_mma, INIT_K_WIDTH)))
        # ### ASM Note: Opaque MOV prevents CSE of initialization or cloning of symbolic warmup blocks.
        zero_a = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", constraints="=v,v", args=[zero_a],
            dtype=gl.bfloat16, is_pure=False, pack=2)
        # ### ASM Note: Opaque MOV prevents CSE of initialization or cloning of symbolic warmup blocks.
        zero_b = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", constraints="=v,v", args=[zero_b],
            dtype=gl.bfloat16, is_pure=False, pack=2)
        zero_c = gl.full((CM, CN), 0, gl.float32, hcu.MmacInterleave2Layout(init_mma))
        acc = gl.convert_layout(gl.amd.cdna4.mfma(zero_a, zero_b, zero_c), result)
    # Partition membership is invariant across K. Compute each physical-wave
    # pair ID once, outside the loop; retain four independent SIMD handoffs.
    sync_id = 0
    arrive_id = 0
    if PINGPONG:
        wave = hcu.get_wave_id() & (WAVES - 1)
        sync_id = 8 + PARTITION * WAVES + wave
        arrive_id = 8 + (1 - PARTITION) * WAVES + wave
    # Operand EMPTY credits already count each reader wave independently.
    # No consumer rendezvous is needed here. EB0 belongs to C1 scale READY;
    # EB2 belongs to C0 scale READY, and both complete before slot1 reuse.
    # Two complete BK tiles per iteration; no remainder or separate tail.
    gl.static_assert(K % (2 * BK) == 0)
    a_slot = first_k_tile % A_STAGES
    a_phase = (first_k_tile // A_STAGES) & 1
    first_pair = 0
    if INSN_WARMUP:
        # Preserve the committed short-loop warmup fix: LLVM must not clone
        # the inline assembly's symbolic MMAC-entry labels.
        # ### ASM Note: Opaque MOV prevents CSE of initialization or cloning of symbolic warmup blocks.
        first_pair = gl.inline_asm_elementwise(
            "s_mov_b32 $0, 0", constraints="=s", args=[],
            dtype=gl.int32, is_pure=False, pack=1)
        gl.static_assert(K >= 2 * BK)
        gl.assume(first_pair < K // 256)
    operand_a: gl.constexpr = gl.DotOperandLayout(0, mma, DOT_K_WIDTH)
    acc_halves = gl.split(acc.reshape((CM, 2, CN // 2)).permute((0, 2, 1)))
    acc0 = gl.convert_layout(acc_halves[0], result)
    acc1 = gl.convert_layout(acc_halves[1], result)
    a0 = gl.full((CM, BK), 0, gl.int8 if IS_INT8 else am.dtype, operand_a)
    a1 = gl.full((CM, BK), 0, gl.int8 if IS_INT8 else am.dtype, operand_a)
    for pair in range(first_pair, K // 256):
        for panel in gl.static_range(2):
            panel_acc = acc0 if panel == 0 else acc1
            a_ticket = first_k_tile + 2 * pair
            b_ticket = 2 * (first_k_tile + 2 * pair + panel) + PARTITION
            panel_acc, a0, a1 = _consume_retained(
                panel_acc, am, bm, a0, a1, a_ticket, b_ticket, sync_id, arrive_id,
                mma, PARTITION, 0, WAVES, PINGPONG, INSN_WARMUP, panel)
            panel_acc, a0, a1 = _consume_retained(
                panel_acc, am, bm, a0, a1, a_ticket + 1, b_ticket, sync_id, arrive_id,
                mma, PARTITION, 1, WAVES, PINGPONG, INSN_WARMUP, panel)
            if panel == 0:
                acc0 = panel_acc
            else:
                acc1 = panel_acc
    acc = gl.convert_layout(gl.join(acc0, acc1).permute((0, 2, 1)).reshape((CM, CN)), result)
    if PARTITION == 0:
        hcu.ebarrier_sync(2, 2 * WAVES)
    else:
        hcu.ebarrier_sync(0, 2 * WAVES)
    sa = sam.load(gl.SliceLayout(1, result))
    sb = sbm.slice(col, CN).load(gl.SliceLayout(0, result))
    if has_next_output:
        # A slot1 holds old scales; both readers release it before next A1.
        # Signal its last LDS read, without waiting for this output's stores.
        hcu.sched_barrier()
        hcu.wait_lgkmcnt(0)
        hcu.ebarrier_arrive(5, 2 * PRODUCER_WAVES + CONSUMERS * WAVES)
        hcu.sched_barrier()
    store_channel_output(C, acc, sa, sb, block_m, block_n, N, BN, PARTITION, WM, WN, USE_C64)


@gluon.jit
def gemm1_pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: gl.constexpr):
    """Match non-Gluon MoE pid_grid, including its final partial M group.

    For four N tiles: GROUP_SIZE_M=1 maps IDs 0..3 to (M0, N0..3);
    GROUP_SIZE_M=8 maps IDs 0..7 to (M0..7, N0), then advances N.
    This changes output-tile traversal, not the consumer N split.
    """
    if GROUP_SIZE_M == 1:
        return pid // num_pid_n, pid % num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    return first_pid_m + pid % group_size_m, (pid % num_pid_in_group) // group_size_m


@gluon.jit
def _persistent_worker(A, B, SA, SB, C, Sorted, Experts, expert_mem, route_mem,
                       am, bm, sam, sbm, SHARE_ROUTES: gl.constexpr, first_tile, total_tiles,
                       M, E, N: gl.constexpr, K: gl.constexpr, TOPK: gl.constexpr,
                       stride_bse, stride_bsn: gl.constexpr, stride_asm: gl.constexpr,
                       BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                       WM: gl.constexpr, WN: gl.constexpr, MMA_TILES_PER_WAVE_N: gl.constexpr,
                       PARTITION: gl.constexpr, PINGPONG: gl.constexpr, USE_MLS_LOAD: gl.constexpr,
                       A_STAGES: gl.constexpr, INSN_WARMUP: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                       USE_A64: gl.constexpr, USE_B64: gl.constexpr, USE_C64: gl.constexpr,
                       IS_INT8: gl.constexpr, num_tokens_post_padded_ptr):
    """Run one role over tile_id += grid; retain tickets across outputs.

        The optional CK device counter caps work inside each WASP role. P0
        prepares next metadata in retired A0 and preissues its first B-left;
        P1 waits metadata READY. Consumers release old scales before their stores."""
    if num_tokens_post_padded_ptr is not None:
        total_tiles = gl.minimum(total_tiles, gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM) * gl.cdiv(N, BN))
    iteration = 0
    num_pid_n = gl.cdiv(N, BN)
    for tile_id in range(first_tile, total_tiles, gl.num_programs(0)):
        block_m, block_n = gemm1_pid_grid(tile_id, gl.cdiv(total_tiles, num_pid_n), num_pid_n, GROUP_SIZE_M)
        first_k_tile = iteration * (K // BK)
        next_tile = tile_id + gl.num_programs(0)
        has_next_output = next_tile < total_tiles
        if PARTITION < 2:
            _producer(A, B, SA, SB, Sorted, Experts, expert_mem, am, bm, sam, sbm, route_mem,
                     SHARE_ROUTES, block_m, block_n, M, E, N, K, TOPK,
                     stride_bse, stride_bsn, stride_asm, BM, BN, BK, PRODUCER_WAVES,
                     PARTITION, USE_MLS_LOAD, A_STAGES, first_k_tile, iteration > 0, USE_A64, USE_B64)
        else:
            _consumer(C, Sorted, Experts, am, bm, sam, sbm, route_mem, SHARE_ROUTES, block_m, block_n,
                     M, E, N, K, TOPK, BM, BN, BK, WAVES, WM, WN, MMA_TILES_PER_WAVE_N,
                     PARTITION - 2, PINGPONG, USE_MLS_LOAD, A_STAGES, INSN_WARMUP,
                     first_k_tile, iteration & 1, iteration == 0, has_next_output, USE_C64, IS_INT8)
        if has_next_output:
            if PARTITION == 0:
                next_m, next_n = gemm1_pid_grid(next_tile, gl.cdiv(total_tiles, num_pid_n), num_pid_n, GROUP_SIZE_M)
                metadata_layout: gl.constexpr = gl.BlockedLayout([1], [WAVE_SIZE], [WAVES], [0])
                routes = gl.load(Sorted + next_m.to(gl.int64) * BM + gl.arange(0, BM, metadata_layout))
                expert = gl.load(Experts + next_m + gl.arange(0, 1, metadata_layout))
                next_expert = gl.maximum(0, gl.minimum(gl.sum(expert, 0), E - 1))
                # Metadata aliases A slot0; tail scales independently alias A slot1.
                # Old A(Q-2) releases A slot0; it then holds next metadata.
                # Next metadata overwrites A0 only after all current A0 reads.
                hcu.abarrier_try_wait(8, ((first_k_tile + K // BK) >> 1) & 1)
                route_mem.store(routes)
                expert_mem.store(expert)
                # Publish only completed metadata writes. P1 can prepare A0
                # while P0 issues Bleft0; EB3 still orders the actual copies.
                hcu.wait_lgkmcnt(0)
                hcu.ebarrier_arrive(1, 2 * PRODUCER_WAVES)
                hcu.ebarrier_arrive(5, 2 * PRODUCER_WAVES + CONSUMERS * WAVES)
                left, source, phase, ready, empty = _prepare_b_load(
                    B, bm, next_expert, next_n, N, K, BN, BK, 0, 0, first_k_tile + K // BK)
                _load_b_tile(left, source, phase, ready, empty, N, K, BN, BK)
            elif PARTITION == 1:
                hcu.ebarrier_sync(1, 2 * PRODUCER_WAVES)
        iteration += 1


@gluon.jit
def gemm_top_w8a8_channel_wasp_persistent(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    M, E, N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    # Channel scales have a singleton K dimension; its strides are unused.
    stride_bse, stride_bsn: gl.constexpr, stride_bsk: gl.constexpr,
    stride_asm: gl.constexpr, stride_ask: gl.constexpr,
    TOTAL_TILES,
    # Autotune configuration (subject to the kernel's tile/resource constraints).
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr = 1,
    USE_ADDR_OFFSET_INT64_A: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_B: gl.constexpr = False,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr = False,
    ck_sorting: gl.constexpr = True,
    PINGPONG: gl.constexpr = True, INSN_WARMUP: gl.constexpr = False,
    LDS_WRAP: gl.constexpr = 32, COPY_BYTES: gl.constexpr = 16,
    # Launch/probe configuration; not autotune parameters.
    PARTITION_REGS: gl.constexpr = None,  # P0, P1, C0, C1
    TILE_ID: gl.constexpr = 0,  # Single-TG probe offset
    num_tokens_post_padded_ptr=None,
):
    """Allocate BN512 buffers and launch four persistent roles."""
    use_mls_load_b: gl.constexpr = True
    share_routes: gl.constexpr = True
    gl.static_assert(ck_sorting and top_k > 0 and top_k <= 256)
    gl.static_assert(BLOCK_SIZE_M == 128 and BLOCK_SIZE_N == 512 and BLOCK_SIZE_K == 128)
    gl.static_assert(K >= 512 and K % 512 == 0 and N > 0 and N % 512 == 0)
    gl.static_assert((a_ptr.dtype.element_ty.is_fp8() and b_ptr.dtype.element_ty.is_fp8())
                     or (a_ptr.dtype.element_ty == gl.int8 and b_ptr.dtype.element_ty == gl.int8))
    gl.static_assert(a_scale_ptr.dtype.element_ty == gl.float32 and b_scale_ptr.dtype.element_ty == gl.float32)
    A_STAGES: gl.constexpr = 2
    CM: gl.constexpr = 128
    CN: gl.constexpr = 256
    mls_tile: gl.constexpr = (128, 64)
    tiles_per_warp: gl.constexpr = (1, 4)
    WM: gl.constexpr = 2
    WN: gl.constexpr = 2
    partition_regs: gl.constexpr = ((16, 48, 224, 224) if USE_ADDR_OFFSET_INT64_A
                                   else (32, 48, 216, 216)) if PARTITION_REGS is None else PARTITION_REGS
    pid = gl.program_id(0) + TILE_ID
    block_m, block_n = gemm1_pid_grid(pid, gl.cdiv(TOTAL_TILES, gl.cdiv(N, BLOCK_SIZE_N)),
                                     gl.cdiv(N, BLOCK_SIZE_N), GROUP_SIZE_M)
    if USE_ADDR_OFFSET_INT64_A:
        al: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 1, [1, 0])
    else:
        al: gl.constexpr = _buffer_lds_layout(CM, BLOCK_SIZE_K, 1, PRODUCER_WAVES, LDS_WRAP, COPY_BYTES)
    metadata_layout: gl.constexpr = gl.BlockedLayout([1], [WAVE_SIZE], [WAVES], [0])
    expert_prefetch = gl.load(expert_ids_ptr + block_m + gl.arange(0, 1, metadata_layout), pid < TOTAL_TILES, other=-1)
    routes_prefetch = gl.load(sorted_token_ids_ptr + block_m.to(gl.int64) * CM + gl.arange(0, CM, metadata_layout),
        pid < TOTAL_TILES, other=-1)
    storage_dtype: gl.constexpr = gl.float8e4nv if a_ptr.dtype.element_ty == gl.int8 else a_ptr.dtype.element_ty
    am = gl.allocate_shared_memory(storage_dtype, [2, CM, BLOCK_SIZE_K], al)
    bl: gl.constexpr = hcu.MlsSharedLayout(1, mls_tile, alt=1, order=(0, 1))
    # B-left and B-right share the same three physical B slots.
    b0 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [3, 256, 128], bl)
    b1 = b0
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    tail = am.index(1)._reinterpret(gl.float32, [4096], sl)
    sam = tail.slice(BLOCK_SIZE_N, BLOCK_SIZE_M)
    sbm = tail.slice(0, BLOCK_SIZE_N)
    raw = am.index(0)._reinterpret(gl.int32, [4096], sl)
    route_mem = raw.slice(2 * BLOCK_SIZE_N, BLOCK_SIZE_M)
    expert_mem = raw.slice(2 * BLOCK_SIZE_N + BLOCK_SIZE_M, 1)
    _initialize(PRODUCER_WAVES, WAVES, A_STAGES)
    route_mem.store(routes_prefetch)
    expert_mem.store(expert_prefetch)
    gl.thread_barrier()
    gl.warp_specialize([
        (_empty, ()),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b0, sam, sbm,
            share_routes, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            0, PINGPONG, use_mls_load_b, A_STAGES, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_A, USE_ADDR_OFFSET_INT64_B, USE_ADDR_OFFSET_INT64_C,
            a_ptr.dtype.element_ty == gl.int8, num_tokens_post_padded_ptr)),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b1, sam, sbm,
            share_routes, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            1, PINGPONG, use_mls_load_b, A_STAGES, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_A, USE_ADDR_OFFSET_INT64_B, USE_ADDR_OFFSET_INT64_C,
            a_ptr.dtype.element_ty == gl.int8, num_tokens_post_padded_ptr)),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b0, sam, sbm,
            share_routes, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            2, PINGPONG, use_mls_load_b, A_STAGES, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_A, USE_ADDR_OFFSET_INT64_B, USE_ADDR_OFFSET_INT64_C,
            a_ptr.dtype.element_ty == gl.int8, num_tokens_post_padded_ptr)),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b1, sam, sbm,
            share_routes, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            3, PINGPONG, use_mls_load_b, A_STAGES, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_A, USE_ADDR_OFFSET_INT64_B, USE_ADDR_OFFSET_INT64_C,
            a_ptr.dtype.element_ty == gl.int8, num_tokens_post_padded_ptr)),
    ], [PRODUCER_WAVES, PRODUCER_WAVES, WAVES, WAVES], partition_regs)
    for id in gl.static_range(ABARRIER_FIRST, ABARRIER_FIRST + abarrier_count(A_STAGES)):
        hcu.abarrier_invalidate(id)


def is_persistent_channel_gemm1_supported(a, b, a_scale, b_scale, *, arch, cu_count, top_k, config,
                                          c_sorted=True, a_sorted=False, output_dtype=None,
                                          ck_sorting=True, gemm_type="gemm1",
                                          scale_recipe="fp8_channel", has_bias=False,
                                          mul_routed_weight=False):
    """Check input/tile compatibility on CPU, without compiling or launching.

    Internal pipeline/register settings are validated by the kernel/compiler.
    The caller checks device placement, routing and downstream sorted layouts,
    and supplies independent A/B/C address-width launch constants."""
    import torch

    if arch != "gfx946" or type(cu_count) is not int or cu_count <= 0:
        return False
    if (gemm_type != "gemm1" or scale_recipe not in ("fp8_channel", "int8_channel")
            or not c_sorted or a_sorted or not ck_sorting or has_bias or mul_routed_weight):
        return False
    if output_dtype is None:
        output_dtype = torch.bfloat16
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    valid_inputs = ((a.dtype in fp8_dtypes and b.dtype in fp8_dtypes and scale_recipe == "fp8_channel")
                    or (a.dtype == b.dtype == torch.int8 and scale_recipe == "int8_channel"))
    if (not valid_inputs
            or a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32
            or output_dtype not in (torch.bfloat16, torch.float16)):
        return False
    if a.ndim != 2 or b.ndim != 3 or not a.is_contiguous() or not b.is_contiguous():
        return False
    m, k = a.shape
    e, n, b_k = b.shape
    # CK reserves low 24 bits for tokens (including padding M), high 8 for routes.
    if min(m, e, n, k) <= 0 or b_k != k or not (0 < top_k <= 256):
        return False
    if a_scale.shape != (m, 1) or b_scale.shape not in ((e, n), (e, n, 1)):
        return False
    bm = config.get("BLOCK_SIZE_M")
    bn = config.get("BLOCK_SIZE_N")
    bk = config.get("BLOCK_SIZE_K")
    if bm != 128 or bn != 512 or bk != 128:
        return False
    # BK128 consumers process complete ping/pong pairs, with at least four BKs.
    if k < 512 or k % 512:
        return False
    if n % 512:
        return False
    return True
