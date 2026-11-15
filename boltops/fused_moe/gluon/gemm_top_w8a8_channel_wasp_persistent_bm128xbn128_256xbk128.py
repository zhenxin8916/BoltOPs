# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent FP8/INT8 channel GEMM1 for gfx946, BM128/BN128,256/BK128.

Contiguous FP8 (E4M3FN/E5M2, including mixed formats) or signed INT8 A/B;
FP32 channel scales; BF16/FP16 sorted output. CK routes select A by low24 bits.
Requires K>=512 and K%256=0. Each role owns four waves.

All supported geometries use the same eight-slot A/B ring.
Copy issue order (arrows do not imply waiting for data to return):
    First BK: Bleft0(P0) -> A0(P1) -> Bright0(P0)
    Later BKs: A1(P1) -> Bleft1(P0) -> Bright1(P0) -> A2(P1) -> ...
    EB3/EB4 gate each A burst between P0's B bursts.
Consumer order, with four waves per role:
    C0: wait/read B -> wait/read A -> release B/A -> handoff -> MMAC
    C1: wait/read A -> ping handoff -> wait/read B -> release A/B -> MMAC
    The pre-B C1 handoff applies to ping only; C0 waits before pong MMAC.
    A reuse requires both consumers' EMPTY credits; B has one consumer.

A and both B halves alias eight 16-KiB LDS slots with 16 abarriers. P0 owns B;
P1 owns A/scales. B uses MLS global->LDS copies; A uses buffer-to-LDS or typed loads/stores
for wide addresses. Each consumer owns an N half; both share A.
Continuous tickets determine slot/phase across outputs, including short K.
Metadata aliases the next A0 slot; scales alias the next Bright0 slot.
P0 publishes completed next-output metadata before Bleft0 so P1 can prepare A0.
Consumers release scales before output stores, overlapping the next Bleft/A.
Each consumer K-loop iteration processes two BKs (ping/pong).
Grid is caller-supplied; persistent TGs advance tile_id += grid.
Instruction warmup is disabled by default.
All transport preserves input bytes. INT8 accumulates in INT32; FP8 in FP32.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
from triton.experimental.gluon.language.amd.cdna4 import async_copy


GIB = gl.constexpr(1024 * 1024 * 1024)  # Binary gigabyte, in bytes.

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
    num_warps=4, num_stages=2, use_async_copy=True,
    wasp_partition_warps=(4, 4, 4, 4), wasp_wdra=True,
)


@gluon.constexpr_function
def _matrix_store_cast_layout(layout):
    # Native store packs each eight 16-bit elements as 0,4,1,5,2,6,3,7.
    # Rotate only register bases, preserving lane/warp ownership and addresses.
    """Match matrix_store register packing without changing lane/wave ownership."""
    regs = layout.reg_bases
    return gl.DistributedLinearLayout(
        [regs[2], regs[0], regs[1]] + regs[3:],
        layout.lane_bases, layout.warp_bases, layout.block_bases, layout.shape)


@gluon.jit
def _cast_for_matrix_store(value, dtype: gl.constexpr, result: gl.constexpr):
    """Reorder FP32 registers, narrow to 16 bits, then restore the store layout."""
    linear: gl.constexpr = gl.to_linear_layout(value.type.layout, value.shape)
    packing: gl.constexpr = _matrix_store_cast_layout(linear)
    value = gl.convert_layout(value, packing)
    value = value.to(dtype)
    return gl.convert_layout(value, result)


@gluon.jit
def store_channel_output(C, acc, sa, sb, block_m, block_n, N: gl.constexpr, BN: gl.constexpr,
                         PARTITION: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr,
                         USE_ADDR_OFFSET_INT64_C: gl.constexpr):
    """Apply channel scales to this consumer's N half and write sorted C.

acc[128,CN] -> scale/narrow each panel -> matrix_store, without extra LDS.
PARTITION selects the N half; the caller zeroes scales for invalid rows."""
    CM: gl.constexpr = acc.shape[0]
    CN: gl.constexpr = acc.shape[1]
    BM: gl.constexpr = CM
    result: gl.constexpr = acc.type.layout
    col: gl.constexpr = PARTITION * CN
    gl.static_assert(CM == 128 and (CN == 64 or CN == 128))
    gl.static_assert((CN == 64 and WM == 4 and WN == 1)
                     or (CN == 128 and WM == 2 and WN == 2))
    gl.static_assert(BN == 2 * CN and N > 0)
    gl.static_assert(acc.dtype == gl.float32 or acc.dtype == gl.int32)
    # INT8 keeps exact integer accumulation through K; scale in FP32 once.
    if acc.dtype == gl.int32:
        acc = acc.to(gl.float32)
    gl.static_assert(C.dtype.element_ty == gl.bfloat16 or C.dtype.element_ty == gl.float16)
    # M stays interleaved across WM waves for paired A reads. Slice registers
    # into native store groups, then scale/convert/store each group immediately.
    # BN128 [4,1]: four [64,32] groups per consumer.
    # BN256 [2,2]: eight [32,64] groups per consumer.
    # Match ordinary Triton: widen before multiplying for large outputs.
    if USE_ADDR_OFFSET_INT64_C:
        C = C + block_m.to(gl.int64) * BM * N
    else:
        C = C + block_m * BM * N
    # Split along M, keeping each accumulator group paired with its A scales:
    #
    #     acc [128, CN]                  sa [128]
    #           |                          |
    #           +--- split into halves ----+
    #           |                          |
    #     2 x [64, CN]                  2 x [64]       <- WM=4: ready to store
    #           |                          |
    #           +--- WM=2: split again ----+
    #           |                          |
    #     4 x [32, CN]                  4 x [32]       <- WM=2: ready to store
    #
    # Each store group covers 16 * WM rows. N is split into panels below.
    # These are register views; no global/LDS copy is performed here.
    m_halves = gl.split(acc.reshape((2, CM // 2, CN)).permute((1, 2, 0)))
    sa_halves = gl.split(sa.reshape((2, CM // 2)).trans())
    if WM == 2:
        first = gl.split(m_halves[0].reshape((2, CM // 4, CN)).permute((1, 2, 0)))
        second = gl.split(m_halves[1].reshape((2, CM // 4, CN)).permute((1, 2, 0)))
        sa_first = gl.split(sa_halves[0].reshape((2, CM // 4)).trans())
        sa_second = gl.split(sa_halves[1].reshape((2, CM // 4)).trans())
    for mi in gl.static_range(CM // (INST_M * WM)):
        if WM == 2:
            if mi < 2:
                m_part = first[mi]
                sa_chunk = sa_first[mi]
            else:
                m_part = second[mi - 2]
                sa_chunk = sa_second[mi - 2]
        else:
            m_part = m_halves[mi]
            sa_chunk = sa_halves[mi]
        sa_part = gl.convert_layout(sa_chunk, gl.SliceLayout(1, result))
        if WN == 1:
            # BN128: each 64-row group emits two 32-column store panels.
            parts = gl.split(m_part.reshape((INST_M * WM, 2, CN // 2)).permute((0, 2, 1)))
            scales = gl.split(sb.reshape((2, CN // 2)).trans())
            for ni in gl.static_range(2):
                part = gl.convert_layout(parts[ni], result)
                sb_part = gl.convert_layout(scales[ni], gl.SliceLayout(0, result))
                part = _cast_for_matrix_store(part * sa_part[:, None] * sb_part[None, :], C.dtype.element_ty, result)
                store_col = block_n * BN + col + ni * 32
                if N % BN == 0 or store_col < N:
                    hcu.matrix_store(C, part, (BM, N), (N, 1),
                                     (mi * INST_M * WM, store_col),
                                     tile=(16, 32), boundary_check=(1,))
        else:
            # BN256 MLS. Former buffer-path conversion retained for reference:
            # part = gl.convert_layout(m_part, result)
            # Two MMAC N tiles per wave own 32 columns. Stream two 64-column CTA
            # panels so matrix_store retains the same per-wave ownership.
            parts = gl.split(m_part.reshape((INST_M * WM, 2, CN // 2)).permute((0, 2, 1)))
            scales = gl.split(sb.reshape((2, CN // 2)).trans())
            for ni in gl.static_range(2):
                part = gl.convert_layout(parts[ni], result)
                sb_part = gl.convert_layout(scales[ni], gl.SliceLayout(0, result))
                part = _cast_for_matrix_store(part * sa_part[:, None] * sb_part[None, :], C.dtype.element_ty, result)
                hcu.matrix_store(C, part, (BM, N), (N, 1),
                                 (mi * INST_M * WM, block_n * BN + col + ni * (CN // 2)),
                                 tile=(16, 32), boundary_check=(1,))

@gluon.constexpr_function
def gemm1_mls_supports_n(block_n, n):
    """Check the N tails handled by the current native MLS boundary lowering."""
    half_n = block_n // 2
    remainder = n % half_n
    return remainder == 0 or remainder >= half_n - 32


@gluon.constexpr_function
def gemm1_partition_regs(USE_ADDR_OFFSET_INT64_A=False):
    """P0/P1/C0/C1 quotas: normal A copies or 64-bit A offsets; B uses MLS."""
    if not USE_ADDR_OFFSET_INT64_A:
        return (32, 48, 216, 216)
    return (64, 96, 176, 176)


@gluon.constexpr_function
def gemm1_mls_warps(shape, mls_tile, num_warps):
    """MLS waves (K,N), with BK128 and tile64x32:

    BN128: [2,2], one N repetition per B half.
    BN256: [2,2], two N repetitions per B half.
    """
    warps_k = min(num_warps, max(1, shape[0] // mls_tile[0]))
    return warps_k, num_warps // warps_k


@gluon.constexpr_function
def gemm1_mma_warps(m, n, waves=4, mma_tiles_per_wave_n=2):
    """Consumer MMAC waves (WM,WN), with two N tiles per wave:

    BN128: consumer tile128x64  -> [4,1].
    BN256: consumer tile128x128 -> [2,2].
    """
    if m <= 0 or n <= 0 or waves not in (1, 2, 4) or mma_tiles_per_wave_n not in (1, 2, 4):
        raise ValueError("invalid consumer tile, wave count or MMA_TILES_PER_WAVE_N")
    tile_m, tile_n = INST_M.value, INST_N.value * mma_tiles_per_wave_n
    m_slots, n_slots = m // tile_m, n // tile_n
    if m_slots > 0 and n_slots > 0 and waves > m_slots * n_slots:
        return waves // n_slots, n_slots
    wm, wn = 1, 1
    while wm * wn < waves:
        if m // (tile_m * 2) // wm >= n // tile_n // wn and wm < m_slots:
            wm *= 2
        else:
            wn *= 2
    if wn * tile_n > n:
        repeats = max(1, wm * tile_m // m) * max(1, wn * tile_n // n)
        swapped = max(1, wn * tile_m // m) * max(1, wm * tile_n // n)
        if swapped <= repeats:
            return wn, wm
    return wm, wn


@gluon.constexpr_function
def _buffer_lds_layout(rows, bk, element_bytes, waves, wrap_step, copy_bytes):
    """A copy layout; BN128 and BN256 use the same defaults:

    rows=128, bk=128, element_bytes=1, waves=4, wrap_step=32, copy_bytes=16.
    A[128,128] = 16 KiB per BK; each wave copies 32 rows = 4 KiB.

        wave0: rows 0,1,  8,9,   ... 120,121
        wave1: rows 2,3, 10,11,  ... 122,123
        wave2: rows 4,5, 12,13,  ... 124,125
        wave3: rows 6,7, 14,15,  ... 126,127

    Each row uses 8 lanes x 16 bytes. Rows interleave in pairs across waves.
    BufferLdsSharedLayout also records the physical copy placement/rotation;
    copy_bytes and wrap_step are byte counts. FP8 and INT8 share this byte view.
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
    # ### ASM Note: Gluon has no direct EXEC-mask API; save/clear EXEC for warmup.
    saved_exec = gl.inline_asm_elementwise(
        "s_mov_b64 $0, exec\n\ts_mov_b64 exec, 0", constraints="=&s,~{exec},~{memory}",
        args=[], dtype=gl.uint64, is_pure=False, pack=1)
    # ### ASM Note: Gluon cannot jump to named MMAC labels; ASM supplies entry/return control.
    gl.inline_asm_elementwise(
        "s_branch " + _icache_label(PARTITION, SLOT, "mmac") + "\n"
        + _icache_label(PARTITION, SLOT, "resume") + ":",
        constraints="=s,~{memory}", args=[], dtype=gl.uint32, is_pure=False, pack=1)
    # ### ASM Note: Gluon has no direct EXEC-mask API; restore the mask after warmup.
    gl.inline_asm_elementwise(
        "s_mov_b64 exec, $1", constraints="=s,s,~{exec},~{memory}", args=[saved_exec],
        dtype=gl.uint32, is_pure=False, pack=1)


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


@gluon.constexpr_function
def _abarrier_init_asm():
    """Initialize eight READY(count4)/EMPTY(count8) slot pairs."""
    lines = []
    for barrier_id in range(16):
        count = 8 if barrier_id >= 8 else 4
        lines.append(f"s_abarrier_init {barrier_id}, {count}")
    return "\n\t".join(lines) + "\n\ts_mov_b32 $0, 0"


@gluon.jit
def _initialize():
    """Initialize the eight shared A/B slots before splitting into P0/P1/C0/C1.

        LDS slot s       0   1   2   3   4   5   6   7
        READY ID         1   2   3   4   5   6   7   0    count=4
        EMPTY ID         8   9  10  11  12  13  14  15    count=8

        producer                         consumer(s)
          wait EMPTY -> copy -> READY ----> wait READY -> read
               ^                                           |
               +---------------- EMPTY <-------------------+

    P0: copies B-left for C0 and B-right for C1; publishes their READY.
    P1: copies shared A for C0/C1; publishes its READY.
    C0: reads B-left/A; returns EMPTY credits: B-left = 4 waves x 2, A = 4 waves x 1.
    C1: reads A/B-right; returns EMPTY credits: A = 4 waves x 1, B-right = 4 waves x 2.
    A needs both consumers' credits (4 + 4); each B half needs only its reader (8).
    READY always counts the four waves of the producer owning this copy.
    Slots rotate between A/B; they are not permanently assigned to a role.

    Startup runs on four waves: each arrives with 2 EMPTY credits per slot,
    giving 4 x 2 = 8 so the first copy can start. READY is not seeded;
    consumers must wait for a real copy. The thread barrier separates barrier
    initialization from seeding these first EMPTY credits.
    """
    # ### ASM Note: Builtin init adds a rendezvous per ID; batch ASM needs only one below.
    gl.inline_asm_elementwise(_abarrier_init_asm(), constraints="=s", args=[], dtype=gl.int32, is_pure=False, pack=1)
    gl.thread_barrier()
    for slot in gl.static_range(8):
        hcu.abarrier_arrive(8 + slot, 2)


@gluon.jit
def _load_a_tile(A, am, a_offsets, a_mask, BK: gl.constexpr, tile, first_k_tile,
                 WAIT_EMPTY: gl.constexpr = True, WAIT_GATE: gl.constexpr = False,
                 USE_ADDR_OFFSET_INT64_A: gl.constexpr = False):
    """Copy one A BK into the shared eight-slot buffer.

    first_k_tile = BKs processed by this TG in earlier outputs; tile = local BK.
    Three tickets per BK reserve slots in copy issue order:

        first BK:  B-left (+0) -> A (+1) -> B-right (+2)
        later BK:  A      (+0) -> B-left (+1) -> B-right (+2)
        ticket = 3 * (first_k_tile + tile) + (1 if tile == 0 else 0)
                              |
                +-------------+--------------+
                |                            |
          slot = ticket & 7          phase = (ticket >> 3) & 1
                |                    flips each eight tickets
          LDS = am[slot]
          EMPTY ID = 8 + slot        P1 waits; C0/C1 return credits
          READY ID = (slot + 1) & 7  P1 publishes; C0/C1 wait

    First output (first_k_tile=0):
        A tile       0    1    2    3    4    5
        ticket       1    3    6    9   12   15
        LDS slot     1    3    6    1    4    7
        phase        0    0    0    1    1    1
        EMPTY ID     9   11   14    9   12   15
        READY ID     2    4    7    2    5    0

    A3 reuses A0's slot1 with the opposite phase, after both readers release it.
    READY uses rotated IDs; READY2 still protects slot1, not slot2.
    Tickets continue across outputs instead of resetting the buffer phases.
    WAIT_EMPTY=False means ownership was ensured by the caller/startup handoff.
    EB3 gates the copy burst; only that burst raises P1's issue priority.
    """
    ticket = 3 * (first_k_tile + tile) + gl.where(tile == 0, 1, 0)
    slot = ticket & 7
    phase = (ticket >> 3) & 1
    # The startup/outer-tile rendezvous guarantees all previous users have
    # finished before A0; avoid repeating tail EMPTY1 as the next try-wait ID.
    if WAIT_EMPTY:
        hcu.abarrier_try_wait(8 + slot, phase)
    ready = (slot + 1) & 7
    dest = am.index(slot)
    source = A + gl.cast(tile, gl.int64) * BK if USE_ADDR_OFFSET_INT64_A else A + tile * BK
    # ### ASM Note: Gluon cannot pin source preparation to SGPRs before the gate;
    # empty tied-register ASM enforces this without emitting an instruction.
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
    if USE_ADDR_OFFSET_INT64_A:
        packed = dest._reinterpret(gl.int32, [128, BK // 4],
                                   gl.SwizzledSharedLayout(2,1,8,[1,0]))
        value = gl.load(source.to(gl.pointer_type(gl.int32)) + a_offsets, a_mask, other=0)
        packed.store(value)
        hcu.wait_lgkmcnt(0)
    else:
        async_copy.buffer_load_to_shared(dest, source.to(gl.pointer_type(dest.dtype)), a_offsets, a_mask)
    hcu.abarrier_arrive((slot + 1) & 7)
    hcu.sched_barrier()
    hcu.set_priority(1)
    hcu.sched_barrier()


@gluon.jit
def _b_slot(bm, slot, BN: gl.constexpr):
    """View a shared slot as one B half; no allocation or copy.

    bm[slot] (16 KiB) -> [BK, BN/2]
        BN128: [128,  64] =  8 KiB
        BN256: [128, 128] = 16 KiB
    P0 fills each half; C0 reads left, C1 reads right.
    """
    return bm.index(slot)._reinterpret(bm.dtype, [128, BN // 2], bm.layout)


@gluon.jit
def _prepare_b_load(B, bm, expert, block_n, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr,
                    BK: gl.constexpr, PARTITION: gl.constexpr, tile, first_k_tile):
    """Prepare P0's B-copy arguments; no data is copied here.

    expert / block_n / tile -> 64-bit global B source
    ticket                 -> LDS half-slot, phase, READY / EMPTY IDs
    return these to _load_b_tile(), which waits EMPTY and issues MLS.
    PARTITION=0/1 selects the left/right B half, both copied by P0.
    """
    ticket = 3 * (first_k_tile + tile) + (2 if PARTITION == 1 else gl.where(tile == 0, 0, 1))
    slot = ticket & 7
    phase = (ticket >> 3) & 1
    ready = (slot + 1) & 7
    empty = 8 + slot
    # ### ASM Note: Gluon cannot pin slot/barrier preparation to SGPRs before the gate;
    # empty tied-register ASM enforces this without emitting an instruction.
    slot, phase, ready, empty = gl.inline_asm_elementwise(
        "", constraints="=s,=s,=s,=s,0,1,2,3", args=[slot, phase, ready, empty],
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32), is_pure=False, pack=1)
    col = block_n * BN + PARTITION * (BN // CONSUMERS)
    source = B + expert.to(gl.int64) * N * K + col.to(gl.int64) * K
    source += tile * BK if K < 2 * GIB else gl.cast(tile, gl.int64) * BK
    # ### ASM Note: Gluon cannot pin the B source to SGPRs before the gate;
    # empty tied-register ASM enforces this without emitting an instruction.
    source = gl.inline_asm_elementwise(
        "", constraints="=s,0", args=[source.to(gl.uint64)],
        dtype=gl.uint64, is_pure=False, pack=1).to(B.dtype)
    source = gl.multiple_of(source, 16)
    return _b_slot(bm, slot, BN), source, phase, ready, empty


@gluon.jit
def _load_b_tile(dest, source, phase, ready, empty, N: gl.constexpr, K: gl.constexpr,
                 BN: gl.constexpr, BK: gl.constexpr, block_n, PARTITION: gl.constexpr):
    """Wait EMPTY, copy a B half, then publish READY; mask incomplete N blocks."""
    hcu.abarrier_try_wait(empty, phase)
    hcu.abarrier_seq(ready)
    warps: gl.constexpr = gemm1_mls_warps((BK, BN // 2), dest.layout.tile, WAVES)
    if N % (BN // 2) == 0:
        # Complete native halves: no boundary descriptor or extra arithmetic.
        hcu.matrix_load(dest, source, (K, N), (1, K), (0, 0), warps=warps,
                        pred=True if N % BN == 0 else block_n * BN + PARTITION * (BN // 2) < N)
    else:
        # source already includes this half's N origin. Bounds are relative
        # to that folded base; K remains complete because K is BK-aligned.
        col = block_n * BN + PARTITION * (BN // 2)
        remaining_n = gl.maximum(N - col, 0)
        hcu.matrix_load(dest, source, (K, remaining_n), (1, K), (0, 0),
                        warps=warps, pred=col < N, boundary_check=(1,))
    hcu.abarrier_arrive(ready)


@gluon.jit
def _producer(A, B, SA, SB, Sorted, Experts, expert_mem, am, bm, sam, sbm, route_mem,
              SHARE_ROUTES: gl.constexpr, block_m, block_n, M, E, N: gl.constexpr, K: gl.constexpr,
              TOPK: gl.constexpr, stride_bse, stride_bsn: gl.constexpr, stride_asm: gl.constexpr,
              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, WAVES: gl.constexpr,
              PARTITION: gl.constexpr, first_k_tile=0, preissued_b_left=False,
              USE_ADDR_OFFSET_INT64_A: gl.constexpr = False):
    """P0 issues both B halves; P1 issues A and publishes scales to each consumer."""
    # P0 owns the complete B FIFO; P1 owns A and publishes tail scales.
    # Both producers capture metadata before A0 reuses its slot.
    layout: gl.constexpr = gl.BlockedLayout([1], [WAVE_SIZE], [WAVES], [0])
    expert = gl.sum(expert_mem.load(layout), 0)
    safe_expert = gl.maximum(0, gl.minimum(expert, E - 1))
    if PARTITION == 1:
        if USE_ADDR_OFFSET_INT64_A:
            al: gl.constexpr = gl.BlockedLayout([1,4], [8,8], [4,1], [1,0])
            copy_k: gl.constexpr = BK // 4
            copy_stride: gl.constexpr = K // 4
        else:
            al: gl.constexpr = am.layout.producer_layout()
            copy_k: gl.constexpr = BK
            copy_stride: gl.constexpr = K
        routes = route_mem.load(gl.SliceLayout(1, al)).to(gl.uint32) & 0x00FFFFFF
        valid = (routes < M) & (expert >= 0) & (expert < E)
        ak = gl.arange(0, copy_k, gl.SliceLayout(0, al))
        if USE_ADDR_OFFSET_INT64_A:
            a_offsets = routes[:, None].to(gl.int64) * copy_stride + ak[None, :]
        else:
            a_offsets = routes[:, None].to(gl.uint32) * copy_stride + ak[None, :]
        a_mask = valid[:, None] & gl.full((BM, copy_k), True, gl.int1, al)
        # Retain one-dimensional route IDs/scale offsets in registers. These
        # values belong to this output and need no global reload after K.
        scale_routes = route_mem.load(layout).to(gl.uint32) & 0x00FFFFFF
        scale_valid = (scale_routes < M) & (expert >= 0) & (expert < E)
        sa_offsets = (scale_routes * stride_asm).to(gl.uint32)
        scale_rn = block_n * BN + gl.arange(0, BN, layout)
        sb_base = SB + safe_expert.to(gl.int64) * stride_bse
        sb_offsets = (scale_rn * stride_bsn).to(gl.uint32)
    hcu.ebarrier_sync(3, 2 * WAVES)
    # ### ASM Note: A normal constant may trigger unrolling; MOV makes the bound opaque.
    num_k_tiles = gl.inline_asm_elementwise(
        "s_mov_b32 $0, $1", constraints="=s,s", args=[K // BK],
        dtype=gl.int32, is_pure=False, pack=1)
    # Both producers prepare/wait at priority1. The EB3/4 issue window raises
    # only P1's A-copy burst to priority2 and orders it before the next B half.
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
            _load_b_tile(left, source, phase, ready, empty, N, K, BN, BK, block_n, 0)
        hcu.sched_barrier()
        hcu.ebarrier_arrive(3, 2 * WAVES)
        hcu.sched_barrier()
        hcu.ebarrier_sync(4, 2 * WAVES)
        hcu.sched_barrier()
        for tile in range(num_k_tiles):
            right, source, phase, ready, empty = _prepare_b_load(
                B, bm, safe_expert, block_n, N, K, BN, BK, 1, tile, first_k_tile)
            _load_b_tile(right, source, phase, ready, empty, N, K, BN, BK, block_n, 1)
            if tile + 1 < num_k_tiles:
                left, source, phase, ready, empty = _prepare_b_load(
                    B, bm, safe_expert, block_n, N, K, BN, BK, 0, tile + 1, first_k_tile)
                hcu.sched_barrier()
                hcu.ebarrier_arrive(3, 2 * WAVES)
                hcu.sched_barrier()
                hcu.ebarrier_sync(4, 2 * WAVES)
                hcu.sched_barrier()
                _load_b_tile(left, source, phase, ready, empty, N, K, BN, BK, block_n, 0)
    else:
        if first_k_tile == 0:
            hcu.abarrier_try_wait(9, 0)
        _load_a_tile(A, am, a_offsets, a_mask, BK, 0, first_k_tile, False, True, USE_ADDR_OFFSET_INT64_A)
        if first_k_tile > 0:
            # Next A0 and the previous scales occupy different slots. Only
            # Bright0 must wait for both old scale readers to release the scale slot.
            hcu.ebarrier_sync(5, 2 * WAVES + CONSUMERS * WAVES)
        hcu.sched_barrier()
        hcu.ebarrier_arrive(4, 2 * WAVES)
        hcu.sched_barrier()
        for tile in range(1, num_k_tiles):
            ticket = 3 * (first_k_tile + tile) + gl.where(tile == 0, 1, 0)
            hcu.abarrier_try_wait(8 + (ticket & 7), (ticket >> 3) & 1)
            _load_a_tile(A, am, a_offsets, a_mask, BK, tile, first_k_tile, False, True, USE_ADDR_OFFSET_INT64_A)
            hcu.sched_barrier()
            hcu.ebarrier_arrive(4, 2 * WAVES)
            hcu.sched_barrier()
    if PARTITION == 1:
        # Scales occupy the next Bright0 slot until both consumers capture them in registers.
        last_ticket = 3 * (first_k_tile + K // BK) - 1
        # Prefetch into registers before waiting for the old A/B owner.
        # Only the following LDS stores require the scale slot's EMPTY permission.
        if (2**24 - 1) * stride_asm * 4 >= 4 * GIB:
            scales_a = gl.load(SA + scale_routes.to(gl.int64) * stride_asm, scale_valid, other=0)
        else:
            scales_a = buffer_load(SA, sa_offsets, scale_valid, other=0)
        if (N - 1) * stride_bsn * 4 >= 4 * GIB:
            scales_b = gl.load(sb_base + scale_rn.to(gl.int64) * stride_bsn,
                               (scale_rn < N) & (expert >= 0) & (expert < E), other=0)
        else:
            scales_b = buffer_load(sb_base, sb_offsets,
                                   (scale_rn < N) & (expert >= 0) & (expert < E), other=0)
        hcu.sched_barrier()
        scale_slot = (3 * (first_k_tile + K // BK) + 2) & 7
        last_scale = last_ticket - ((last_ticket - scale_slot) & 7)
        hcu.abarrier_try_wait(8 + scale_slot, ((last_scale >> 3) & 1) ^ 1)
        sam.store(scales_a)
        sbm.store(scales_b)
        # Ordinary LDS stores must complete before notifying the output readers.
        hcu.wait_lgkmcnt(0)
        # Independent readers: C0 need not wait for C1's last MMAC.
        # The explicit LDS wait above publishes scales before either signal.
        hcu.ebarrier_arrive(2, 2 * WAVES)
        hcu.ebarrier_arrive(0, 2 * WAVES)


@gluon.jit
def _read_b(bm, slot, BN: gl.constexpr, MMA: gl.constexpr):
    """Read the B-half view with the layout consumed by MMAC."""
    desc = _b_slot(bm, slot, BN)
    return desc.load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, MMA, DOT_K_WIDTH)))


@gluon.jit
def _consume_block(acc, am, bm, b_tile, sync_id, arrive_id, MMA: gl.constexpr,
                   PARTITION: gl.constexpr, SLOT: gl.constexpr, WAVES: gl.constexpr,
                   PINGPONG: gl.constexpr, INSN_WARMUP: gl.constexpr,
                   PERSISTENT: gl.constexpr = True, first_k_tile=0):
    """Consume one BK; arrows show issue order rather than cycle duration.

    C0: wait B -> read B -> wait A -> read A -> release B/A -> handoff -> MMAC
    C1: wait A -> read A -> ping handoff -> wait B -> read B -> release A/B -> MMAC

    C1 can issue A while C0 waits for LDS reads. On pong, C1 skips the
    pre-B handoff and C0 waits for C1 before MMAC. EMPTY counts both A
    readers; an A slot cannot be overwritten after only one returns it.
    """
    a_ticket = 3 * b_tile + gl.where(b_tile == first_k_tile, 1, 0)
    a_slot = a_ticket & 7
    a_phase = (a_ticket >> 3) & 1
    a_ready = (a_slot + 1) & 7
    b_ticket = 3 * b_tile + (2 if PARTITION == 1 else gl.where(b_tile == first_k_tile, 0, 1))
    b_slot = b_ticket & 7
    b_phase = (b_ticket >> 3) & 1
    b_ready = (b_slot + 1) & 7
    b_empty = 8 + b_slot
    if PARTITION == 0:
        # C0: private B first, then shared A. Producer ordering is unchanged.
        hcu.abarrier_try_wait(b_ready, b_phase)
        b = _read_b(bm, b_slot, acc.shape[1] * 2, MMA)
        hcu.sched_barrier()
        hcu.abarrier_try_wait(a_ready, a_phase)
        a = am.index(a_slot).load(gl.DotOperandLayout(0, MMA, DOT_K_WIDTH))
        if bm.dtype == gl.int8:
            a = a.to(gl.int8, bitcast=True)
    else:
        # Issue shared A reads before the ping handoff; C0 may still be
        # waiting for its LDS reads. Private B and MMAC keep their handoff.
        hcu.abarrier_try_wait(a_ready, a_phase)
        a = am.index(a_slot).load(gl.DotOperandLayout(0, MMA, DOT_K_WIDTH))
        if bm.dtype == gl.int8:
            a = a.to(gl.int8, bitcast=True)
        if PINGPONG and SLOT == 0:
            _consumer_pingpong(sync_id, True, WAVES)
        hcu.abarrier_try_wait(b_ready, b_phase)
        b = _read_b(bm, b_slot, acc.shape[1] * 2, MMA)
    # BN256 MLS has eight reads per operand: release at lgkmcnt(8)/0.
    # BN128 drains all reads before returning either EMPTY credit.
    # Visible waits let LLVM track completion; inline arrivals preserve
    # independent EMPTY publication without an extra full LDS drain.
    a_empty_id = 8 + a_slot
    b_empty_id = b_empty
    hcu.sched_barrier()
    if PARTITION == 0:
        first_empty_id = b_empty_id
        second_empty_id = a_empty_id
    else:
        first_empty_id = a_empty_id
        second_empty_id = b_empty_id
    if acc.shape[1] == 128:
        hcu.wait_lgkmcnt(8)
    else:
        hcu.wait_lgkmcnt(0)
    # ### ASM Note: Builtin arrive lowering can add a full LDS wait;
    # ASM preserves the explicit partial wait for separate operand release.
    gl.inline_asm_elementwise(
        "s_abarrier_arrive $0, $1, $2", constraints="=&s,s,n,~{memory}",
        args=[first_empty_id, gl.constexpr(2 if PARTITION == 0 else 1)], dtype=gl.int32, is_pure=False, pack=1)
    hcu.wait_lgkmcnt(0)
    # ### ASM Note: Avoid builtin arrive's extra LDS wait; completion is already explicit above.
    gl.inline_asm_elementwise(
        "s_abarrier_arrive $0, $1, $2", constraints="=&s,s,n,~{memory}",
        args=[second_empty_id, gl.constexpr(1 if PARTITION == 0 else 2)], dtype=gl.int32, is_pure=False, pack=1)
    hcu.sched_barrier()
    if PINGPONG and PARTITION == 0:
        if SLOT == 0:
            # Nonblocking: C1 can read/compute ping while C0 issues MMACs.
            _consumer_pingpong(arrive_id, False, WAVES)
        else:
            # C1 has finished ping and can cover C0's pong instruction fetch.
            _consumer_pingpong(sync_id, True, WAVES)
    # TileLang's asymmetric priorities: C0 ping/pong=1/3, C1 ping/pong=0/2.
    priority: gl.constexpr = 1 + 2 * SLOT if PARTITION == 0 else 2 * SLOT
    hcu.set_priority(priority)
    hcu.sched_barrier()
    if INSN_WARMUP:
        # Warmup enters after the handoff/priority setup and returns before
        # priority reset and arrive. EXEC=0 visits only these MMAC instructions.
        # ### ASM Note: Gluon cannot define branch-target labels; mark the warmup MMAC entry.
        gl.inline_asm_elementwise(
            _icache_label(PARTITION, SLOT, "mmac") + ":", constraints="=s,~{memory}",
            args=[], dtype=gl.uint32, is_pure=False, pack=1)
    hcu.sched_barrier()
    acc = gl.amd.cdna4.mfma(a, b, acc)
    hcu.sched_barrier()
    if INSN_WARMUP:
        # ### ASM Note: Gluon cannot branch on EXEC to a named label; return warmup waves here.
        gl.inline_asm_elementwise(
            "s_cbranch_execz " + _icache_label(PARTITION, SLOT, "resume"),
            constraints="=s,~{memory}", args=[], dtype=gl.uint32, is_pure=False, pack=1)
    hcu.sched_barrier()
    # Persistent loop-control/address work must keep up with the peer MMAC:
    # C0 reads at priority 3, C1 at 2; compute keeps its ping/pong priorities.
    hcu.set_priority(3 - PARTITION if PERSISTENT else 0)
    hcu.sched_barrier()
    if PINGPONG and PARTITION == 1 and SLOT == 0:
        # Release C0's pong after C1's ping, without a collective rendezvous.
        _consumer_pingpong(arrive_id, False, WAVES)
    return acc


@gluon.jit
def _consumer(C, Sorted, Experts, am, bm, sam, sbm, route_mem, SHARE_ROUTES: gl.constexpr, block_m,
              block_n, M, E, N: gl.constexpr, K: gl.constexpr, TOPK: gl.constexpr, BM: gl.constexpr,
              BN: gl.constexpr, BK: gl.constexpr, WAVES: gl.constexpr, WM: gl.constexpr,
              WN: gl.constexpr, MMA_TILES_PER_WAVE_N: gl.constexpr, PARTITION: gl.constexpr,
              PINGPONG: gl.constexpr, INSN_WARMUP: gl.constexpr, first_k_tile=0, output_phase=0,
              warmup_this_tile=True, has_next_output=False,
              USE_ADDR_OFFSET_INT64_C: gl.constexpr = False):
    """Consume paired BKs, capture shared scales, and release them before output stores."""
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        4, [INST_M, INST_N, INST_K], MMAC_LAYOUT,
        [WM, WN], tiles_per_warp=[1, MMA_TILES_PER_WAVE_N])
    result: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    CM: gl.constexpr = BM
    CN: gl.constexpr = BN // CONSUMERS
    col: gl.constexpr = PARTITION * CN
    if INSN_WARMUP:
        # Instruction cache is shared across this consumer's four waves.
        # Its fixed 4-wave partition starts at a multiple of WAVES.
        if warmup_this_tile:
            wave = hcu.get_wave_id() % WAVES
            for slot in gl.static_range(2):
                if wave == slot * 2:
                    _warm_mmac_instructions(PARTITION, slot)
    acc = gl.full((CM, CN), 0, gl.int32 if bm.dtype == gl.int8 else gl.float32, result)
    if PARTITION == 0 and bm.dtype != gl.int8:
        # BF16 MMACs distinguish zero initialization from FP8 GEMM in traces.
        # M/N ownership and LTS stay identical; BF16 has instruction K=16.
        init_mma: gl.constexpr = gl.amd.AMDMFMALayout(
            4, [INST_M, INST_N, INIT_INST_K], MMAC_LAYOUT,
            [WM, WN], tiles_per_warp=[1, MMA_TILES_PER_WAVE_N])
        # Distinct zero groups prevent CSE from replacing MMACs with MOV copies.
        zero_a = gl.full((CM, INIT_INST_K), 0, gl.bfloat16, gl.DotOperandLayout(0, init_mma, INIT_K_WIDTH))
        zero_b = gl.full((INIT_INST_K, CN), 0, gl.bfloat16, hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, init_mma, INIT_K_WIDTH)))
        # ### ASM Note: Gluon zeros can be folded; opaque MOV keeps the BF16 MMAC initialization.
        zero_a = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", constraints="=v,v", args=[zero_a],
            dtype=gl.bfloat16, is_pure=False, pack=2)
        # ### ASM Note: Gluon may merge identical zeros; opaque MOV keeps B independent of A.
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
    # EB2 belongs to C0 scale READY; both readers finish before scale-slot reuse.
    # Two complete BK tiles per iteration; no remainder or separate tail.
    gl.static_assert(K % (2 * BK) == 0)
    # ### ASM Note: A normal zero may trigger unrolling; opaque MOV preserves the loop
    # to limit register lifetimes and avoid duplicating warmup labels.
    first_pair = gl.inline_asm_elementwise(
        "s_mov_b32 $0, 0", constraints="=s", args=[],
        dtype=gl.int32, is_pure=False, pack=1)
    gl.assume(first_pair < K // (2 * BK))
    for pair in range(first_pair, K // (2 * BK)):
        acc = _consume_block(
            acc, am, bm, first_k_tile + 2 * pair, sync_id, arrive_id, mma, PARTITION, 0,
            WAVES, PINGPONG, INSN_WARMUP, first_k_tile=first_k_tile)
        acc = _consume_block(
            acc, am, bm, first_k_tile + 2 * pair + 1, sync_id, arrive_id, mma, PARTITION, 1,
            WAVES, PINGPONG, INSN_WARMUP, first_k_tile=first_k_tile)
    if PARTITION == 0:
        hcu.ebarrier_sync(2, 2 * WAVES)
    else:
        hcu.ebarrier_sync(0, 2 * WAVES)
    sa = sam.load(gl.SliceLayout(1, result))
    sb = sbm.slice(col, CN).load(gl.SliceLayout(0, result))
    if has_next_output:
        # The scale slot holds old scales; notify reuse before finishing output stores.
        # Signal its last LDS read, without waiting for this output's stores.
        hcu.sched_barrier()
        hcu.wait_lgkmcnt(0)
        hcu.ebarrier_arrive(5, 2 * PRODUCER_WAVES + CONSUMERS * WAVES)
        hcu.sched_barrier()
    store_channel_output(C, acc, sa, sb, block_m, block_n, N, BN, PARTITION, WM, WN, USE_ADDR_OFFSET_INT64_C)


@gluon.jit
def _persistent_worker(A, B, SA, SB, C, Sorted, Experts, expert_mem, route_mem, am, bm, sam, sbm,
                       SHARE_ROUTES: gl.constexpr, first_tile, total_tiles, M, E, N: gl.constexpr,
                       K: gl.constexpr, TOPK: gl.constexpr, stride_bse, stride_bsn: gl.constexpr,
                       stride_asm: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                       BK: gl.constexpr, WM: gl.constexpr, WN: gl.constexpr,
                       MMA_TILES_PER_WAVE_N: gl.constexpr, PARTITION: gl.constexpr,
                       PINGPONG: gl.constexpr, INSN_WARMUP: gl.constexpr,
                       GROUP_SIZE_M: gl.constexpr, USE_ADDR_OFFSET_INT64_C: gl.constexpr,
                       USE_ADDR_OFFSET_INT64_A: gl.constexpr, num_tokens_post_padded_ptr=None):
    """Run one role using the GPU work count and overlap next copies with output tails."""
    if num_tokens_post_padded_ptr is not None:
        padded_rows = gl.load(num_tokens_post_padded_ptr)
        total_tiles = gl.minimum(total_tiles, gl.cdiv(padded_rows, BM) * gl.cdiv(N, BN))
    iteration = 0
    num_pid_n = gl.cdiv(N, BN)
    for tile_id in range(first_tile, total_tiles, gl.num_programs(0)):
        block_m, block_n = gemm1_pid_grid(tile_id, gl.cdiv(total_tiles, num_pid_n), num_pid_n, GROUP_SIZE_M)
        first_k_tile = iteration * (K // BK)
        scalar_layout: gl.constexpr = gl.SwizzledSharedLayout(1,1,1,[0])
        raw = am.index((3 * first_k_tile + 1) & 7)._reinterpret(gl.int32, [BM * BK // 4], scalar_layout)
        route_mem = raw.slice(2 * BN, BM)
        expert_mem = raw.slice(2 * BN + BM, 1)
        tail = am.index((3 * (first_k_tile + K // BK) + 2) & 7)._reinterpret(gl.float32, [BM * BK // 4], scalar_layout)
        sam = tail.slice(BN, BM)
        sbm = tail.slice(0, BN)
        next_tile = tile_id + gl.num_programs(0)
        has_next_output = next_tile < total_tiles
        if PARTITION < 2:
            if iteration == 0:
                if PARTITION == 0:
                    metadata_layout: gl.constexpr = gl.BlockedLayout([1], [WAVE_SIZE], [WAVES], [0])
                    routes = gl.load(Sorted + block_m * BM + gl.arange(0, BM, metadata_layout))
                    expert = gl.load(Experts + block_m + gl.arange(0, 1, metadata_layout))
                    route_mem.store(routes)
                    expert_mem.store(expert)
                hcu.ebarrier_sync(1, 2 * PRODUCER_WAVES)
            _producer(A, B, SA, SB, Sorted, Experts, expert_mem, am, bm, sam, sbm, route_mem,
                     SHARE_ROUTES, block_m, block_n, M, E, N, K, TOPK,
                     stride_bse, stride_bsn, stride_asm, BM, BN, BK, PRODUCER_WAVES,
                     PARTITION, first_k_tile, iteration > 0, USE_ADDR_OFFSET_INT64_A)
        else:
            _consumer(C, Sorted, Experts, am, bm, sam, sbm, route_mem, SHARE_ROUTES, block_m, block_n,
                     M, E, N, K, TOPK, BM, BN, BK, WAVES, WM, WN, MMA_TILES_PER_WAVE_N,
                     PARTITION - 2, PINGPONG, INSN_WARMUP,
                     first_k_tile, iteration & 1, iteration == 0, has_next_output, USE_ADDR_OFFSET_INT64_C)
        if has_next_output:
            if PARTITION == 0:
                next_m, next_n = gemm1_pid_grid(next_tile, gl.cdiv(total_tiles, num_pid_n), num_pid_n, GROUP_SIZE_M)
                metadata_layout: gl.constexpr = gl.BlockedLayout([1], [WAVE_SIZE], [WAVES], [0])
                routes = gl.load(Sorted + next_m * BM + gl.arange(0, BM, metadata_layout))
                expert = gl.load(Experts + next_m + gl.arange(0, 1, metadata_layout))
                next_expert = gl.maximum(0, gl.minimum(gl.sum(expert, 0), E - 1))
                # Scale and metadata slots differ. The previous operand owner
                # alone protects the next metadata stores; old output continues.
                last_ticket = 3 * (first_k_tile + K // BK) - 1
                next_meta_slot = (3 * (first_k_tile + K // BK) + 1) & 7
                last_meta = last_ticket - ((last_ticket - next_meta_slot) & 7)
                hcu.abarrier_try_wait(8 + next_meta_slot, ((last_meta >> 3) & 1) ^ 1)
                next_raw = am.index(next_meta_slot)._reinterpret(gl.int32, [BM * BK // 4], scalar_layout)
                next_raw.slice(2 * BN, BM).store(routes)
                next_raw.slice(2 * BN + BM, 1).store(expert)
                # Publish completed metadata before Bleft0: P1 can prepare A0
                # concurrently, while EB3 still orders the operand copies.
                hcu.wait_lgkmcnt(0)
                hcu.ebarrier_arrive(1, 2 * PRODUCER_WAVES)
                hcu.ebarrier_arrive(5, 2 * PRODUCER_WAVES + CONSUMERS * WAVES)
                left, source, phase, ready, empty = _prepare_b_load(
                    B, bm, next_expert, next_n, N, K, BN, BK, 0, 0, first_k_tile + K // BK)
                _load_b_tile(left, source, phase, ready, empty, N, K, BN, BK, next_n, 0)
            elif PARTITION == 1:
                hcu.ebarrier_sync(1, 2 * PRODUCER_WAVES)
        iteration += 1


@gluon.jit
def _run(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, sorted_token_ids_ptr, expert_ids_ptr, M, E,
         N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr, stride_bse,
         stride_bsn: gl.constexpr, stride_asm: gl.constexpr, TOTAL_TILES,
         GROUP_SIZE_M: gl.constexpr, PINGPONG: gl.constexpr, INSN_WARMUP: gl.constexpr,
         LDS_WRAP: gl.constexpr, COPY_BYTES: gl.constexpr, PARTITION_REGS: gl.constexpr,
         TILE_ID: gl.constexpr, USE_ADDR_OFFSET_INT64_C: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
         USE_ADDR_OFFSET_INT64_A: gl.constexpr, num_tokens_post_padded_ptr=None):
    """Eight shared slots: Bleft0 -> A0 -> Bright0, then A -> Bleft -> Bright.

    Metadata aliases the next A0 slot; scales alias the next Bright0 slot.
    Scale readers release that slot before stores.
    P0 preissues the next Bleft while P1 prepares A, overlapping the output tail.
    """
    BLOCK_SIZE_M: gl.constexpr = 128
    BLOCK_SIZE_K: gl.constexpr = 128
    CM: gl.constexpr = BLOCK_SIZE_M
    CN: gl.constexpr = BLOCK_SIZE_N // CONSUMERS
    gl.static_assert(BLOCK_SIZE_K % 64 == 0,
                     "interleave2 MLS B requires complete K tiles")
    gl.static_assert(gemm1_mls_supports_n(BLOCK_SIZE_N, N),
                     "unsupported native MLS N boundary")
    mls_tile: gl.constexpr = (64, 32)
    tiles_per_warp: gl.constexpr = (1, mls_tile[1] // INST_N)
    warps_per_cta: gl.constexpr = gemm1_mma_warps(CM, CN, WAVES, tiles_per_warp[1])
    WM: gl.constexpr = warps_per_cta[0]
    WN: gl.constexpr = warps_per_cta[1]
    pid = gl.program_id(0) + TILE_ID
    if USE_ADDR_OFFSET_INT64_A:
        al: gl.constexpr = gl.SwizzledSharedLayout(8,1,8,[1,0])
    else:
        al: gl.constexpr = _buffer_lds_layout(CM, BLOCK_SIZE_K, 1, PRODUCER_WAVES, LDS_WRAP, COPY_BYTES)
    # Copy storage is a byte view, independent of the arithmetic input type.
    # The async-copy encoding accepts FP8 backing; INT8 consumers bitcast the
    # loaded bytes without a numerical conversion or extra LDS allocation.
    storage_dtype: gl.constexpr = gl.float8e4nv if a_ptr.dtype.element_ty == gl.int8 else a_ptr.dtype.element_ty
    am = gl.allocate_shared_memory(storage_dtype, [8, CM, BLOCK_SIZE_K], al)
    bl: gl.constexpr = hcu.MlsSharedLayout(1, mls_tile, alt=1, order=(0, 1))
    # A and both B streams alias the same eight physical slots.
    b0 = am._reinterpret(b_ptr.dtype.element_ty, [8, BLOCK_SIZE_K, CM], bl)
    b1 = b0
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    tail = am.index(2)._reinterpret(gl.float32, [CM * BLOCK_SIZE_K // 4], sl)
    sam = tail.slice(BLOCK_SIZE_N, BLOCK_SIZE_M)
    sbm = tail.slice(0, BLOCK_SIZE_N)
    raw = am.index(1)._reinterpret(gl.int32, [CM * BLOCK_SIZE_K // 4], sl)
    route_mem = raw.slice(2 * BLOCK_SIZE_N, BLOCK_SIZE_M)
    expert_mem = raw.slice(2 * BLOCK_SIZE_N + BLOCK_SIZE_M, 1)
    _initialize()
    gl.thread_barrier()
    gl.warp_specialize([
        (_empty, ()),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b0, sam, sbm,
            True, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            0, PINGPONG, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_C, USE_ADDR_OFFSET_INT64_A, num_tokens_post_padded_ptr)),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b1, sam, sbm,
            True, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            1, PINGPONG, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_C, USE_ADDR_OFFSET_INT64_A, num_tokens_post_padded_ptr)),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b0, sam, sbm,
            True, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            2, PINGPONG, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_C, USE_ADDR_OFFSET_INT64_A, num_tokens_post_padded_ptr)),
        (_persistent_worker, (a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, expert_mem, route_mem, am, b1, sam, sbm,
            True, pid, TOTAL_TILES, M, E, N, K, top_k, stride_bse, stride_bsn, stride_asm,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WM, WN, tiles_per_warp[1],
            3, PINGPONG, INSN_WARMUP, GROUP_SIZE_M,
            USE_ADDR_OFFSET_INT64_C, USE_ADDR_OFFSET_INT64_A, num_tokens_post_padded_ptr)),
    ], [PRODUCER_WAVES, PRODUCER_WAVES, WAVES, WAVES], PARTITION_REGS)
    for id in gl.static_range(16):
        hcu.abarrier_invalidate(id)


@gluon.jit
def gemm_top_w8a8_channel_wasp_persistent(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    M, E, N: gl.constexpr, K: gl.constexpr, top_k: gl.constexpr,
    # Channel scales have a singleton K dimension; its strides are unused.
    stride_bse, stride_bsn: gl.constexpr, stride_bsk: gl.constexpr,
    stride_asm: gl.constexpr, stride_ask: gl.constexpr,
    # Runtime number of logical output tiles; launch grid controls persistence.
    TOTAL_TILES,
    # Autotune configuration (subject to the kernel's tile/resource constraints).
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    # Caller selects flags from tensor.numel() * tensor.element_size() >= 4 * GIB - 1.
    # B keeps the common API flag; its copy path always folds expert/N/K into
    # a 64-bit base before issuing MLS.
    USE_ADDR_OFFSET_INT64_A: gl.constexpr, USE_ADDR_OFFSET_INT64_B: gl.constexpr,
    USE_ADDR_OFFSET_INT64_C: gl.constexpr,
    ck_sorting: gl.constexpr = True,
    GROUP_SIZE_M: gl.constexpr = 1,
    A_STAGES: gl.constexpr = 2,  # Accepted for existing callers; the ring always has eight slots.
    PINGPONG: gl.constexpr = True, INSN_WARMUP: gl.constexpr = False,
    # Aligned persistent BN256/BK128: measured wrap32 wins over wrap16.
    LDS_WRAP: gl.constexpr = 32, COPY_BYTES: gl.constexpr = 16,
    # Launch/probe configuration; not autotune parameters.
    PARTITION_REGS: gl.constexpr = None,  # P0, P1, C0, C1; selected by width/copy type
    TILE_ID: gl.constexpr = 0,  # Single-TG probe offset
    num_tokens_post_padded_ptr=None,  # Optional device-side CK work count.
):
    # Autotune may override these measured defaults for either BN.
    """Run the eight-slot pipeline with four fixed producer/consumer roles.

    TOTAL_TILES bounds work; an optional CK device counter narrows that bound.
    Grid determines persistence. GROUP_SIZE_M selects tile traversal.
    Scales reuse retired operand LDS without an additional allocation."""
    partition_regs: gl.constexpr = gemm1_partition_regs(
        USE_ADDR_OFFSET_INT64_A) if PARTITION_REGS is None else PARTITION_REGS
    gl.static_assert(ck_sorting, "this kernel requires CK-packed sorted token IDs")
    gl.static_assert(0 < top_k and top_k <= 256, "CK packs the route index in eight bits")
    gl.static_assert(GROUP_SIZE_M >= 1)
    gl.static_assert(BLOCK_SIZE_M == 128 and (BLOCK_SIZE_N == 128 or BLOCK_SIZE_N == 256))
    gl.static_assert(BLOCK_SIZE_K == 128, "this kernel supports BK128")
    gl.static_assert(8 * BLOCK_SIZE_M * BLOCK_SIZE_K <= 128 * 1024)
    gl.static_assert(K >= 4 * BLOCK_SIZE_K and K % (2 * BLOCK_SIZE_K) == 0,
                     "persistent K requires at least four complete BK tiles, in ping/pong pairs")
    gl.static_assert((a_ptr.dtype.element_ty.is_fp8() and b_ptr.dtype.element_ty.is_fp8())
                     or (a_ptr.dtype.element_ty == gl.int8 and b_ptr.dtype.element_ty == gl.int8))
    gl.static_assert(a_scale_ptr.dtype.element_ty == gl.float32 and b_scale_ptr.dtype.element_ty == gl.float32)
    gl.static_assert(WAVES % PRODUCER_WAVES == 0 and (not PINGPONG or WAVES == 4))
    _run(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, M, E, N, K, top_k,
        stride_bse, stride_bsn, stride_asm, TOTAL_TILES,
        GROUP_SIZE_M, PINGPONG, INSN_WARMUP, LDS_WRAP, COPY_BYTES, partition_regs,
        TILE_ID, USE_ADDR_OFFSET_INT64_C, BLOCK_SIZE_N,
        USE_ADDR_OFFSET_INT64_A, num_tokens_post_padded_ptr)


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
    if bm != 128 or bn not in (128, 256) or bk != 128:
        return False
    # BK128 consumers process complete ping/pong pairs, with at least four BKs.
    if k < 512 or k % 256:
        return False
    if not gemm1_mls_supports_n(bn, n):
        return False
    return True
