# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent FP8/INT8 channel GEMM2 for gfx946, BM64/BN256/BK256.

CK-sorted contiguous byte A/B; FP32 channel scales; BF16/FP16 scatter output,
before the top-k reduction. Requires K256 and N divisible by 256.
Four waves per role; a CPU metadata query selects this specialization.

Roles and data flow:
    P0: A + SA/weight/routes -> C0/C1, alternating two full-A slots
    P1: full-K B-left + SB-left -> C0; full-K B-right + SB-right -> C1
    C0: left 128 columns; C1: right 128 columns -> direct routed scatter C

Persistent traversal (arrows show issue order, not completion):
    P0 each M: wait A EMPTY -> load A -> publish A and metadata -> next M.
    P1 each N: load full-K B-left/SB -> full-K B-right/SB -> next N.
    C0/C1, first N: read A/B/SB + metadata -> LDS wait -> release A/B
                   -> full-K MMAC -> scale/store two M32 halves.
    C0/C1, later N: reuse A/metadata -> read/release B/SB -> MMAC -> scatter.
    A EMPTY follows both consumers' A and metadata reads. No C LDS handoff.

Two full-A slots: 32 KiB; one [K256,N128] B slot per consumer: 64 KiB total.
Double metadata: 2 KiB; two SB generations per consumer: 2 KiB.
Total: 100 KiB LDS. All 16 abarriers initialized; ten IDs carry live handoffs.
A/metadata phases are (M epoch >> 1) & 1; B phases are N epoch & 1.
Grid is caller-supplied; M += grid. INT8 keeps INT32 accumulation through K,
then applies FP32 ((acc * SA) * SB) * weight. Instruction warmup is optional and defaults off.
Full N is the default; N_TILES_LIMIT is diagnostic only."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

# Launch options for this standalone specialization and its probes.
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
    """Initialize all 16 abarriers, rendezvous, then seed initial EMPTY.

        Buffer              Writer -> Reader    READY    EMPTY   Phase
        A slot0/1           P0 -> C0 + C1        1/2      3/4     (M epoch >> 1) & 1
        B-left, full K256   P1 -> C0             5        6       N epoch & 1
        B-right, full K256  P1 -> C1             7        8       N epoch & 1
        M metadata slot0/1  P0 -> C0 + C1        12/0     -       (M epoch >> 1) & 1

        Writer: wait EMPTY -> copy/store -> READY ---> wait READY -> read
                        ^                                              |
                        +---------------- EMPTY <----------------------+

        A EMPTY needs four credits from each consumer; B EMPTY needs four.
        Metadata lifetime is protected by the corresponding A EMPTY returns.
        Other IDs have no runtime handoff; their initialization grants no work."""
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
                         MUL_ROUTED_WEIGHT: gl.constexpr, TOP_K: gl.constexpr, CK_SORTING: gl.constexpr):
    """P0 publishes one M using alternating full-A and metadata slots.

        slot = epoch & 1; phase = (epoch >> 1) & 1
        A: READY = 1 + slot, EMPTY = 3 + slot
        Metadata: READY12 for slot0, READY0 for slot1; same phase as A
        Three [64] vectors hold SA, weight and decoded route; the fourth is unused.
        P0 folds invalid experts into route=-1. SA/weight share one LDS store.

        Both consumers finish A and metadata LDS reads before returning A EMPTY.
        This protects metadata even when P0 has advanced by multiple M blocks."""
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
    if CK_SORTING:
        route = (route & 0x00ffffff) * TOP_K + ((route >> 24) & 255)
    route = gl.where(expert >= 0, route, -1)
    saw = gl.join(sa, weight).permute((1,0))
    pair_meta_layout: gl.constexpr = gl.SharedLinearLayout([[0,4],[0,8],[0,1],[0,2],[0,16],[0,32],[1,0]], [], 16)
    ma._reinterpret(gl.float32,[4,2,BM],pair_meta_layout).index((epoch&1)*2).store(saw)
    ma.index((epoch & 1) * 4 + 2).store(route.to(gl.float32, bitcast=True))
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
        if USE_B64:
            # Keep scale offsets local to the panel, like the B descriptor.
            sb_base = (SB + gl.maximum(expert, 0).to(gl.int64) * STRIDE_SBE
                       + (n.to(gl.int64) * BN + HALF * (BN // 2)) * STRIDE_SBN)
            mb.index((epoch & 1) * 2 + HALF).store(
                gl.load(sb_base + mr.to(gl.int64) * STRIDE_SBN))
        else:
            sb_base = SB
            sb_offsets = (gl.maximum(expert, 0) * STRIDE_SBE + col * STRIDE_SBN).to(gl.uint32)
            async_copy.buffer_load_to_shared(mb.index((epoch&1)*2+HALF), sb_base, sb_offsets)
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
    if USE_B64:
        hcu.wait_lgkmcnt(0)
    else:
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
    """P0 fills alternating A/metadata slots; P1 streams the two full-K B halves.

        P0 each M: wait A EMPTY -> A READY -> metadata READY -> next M.
        P1 each N: wait/load/publish B-left + SB -> B-right + SB -> next N.
        P0 does not drain output buffers: C0/C1 scatter their results directly.
        P0 epochs count M blocks; P1 epochs count N tiles across all M blocks."""
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
                K, BM, BK, USE_A64, STRIDE_SA, MUL_ROUTED_WEIGHT, TOP_K, CK_SORTING)
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
def _scale_store_rows(acc, sa, sb, weight, route, expert, C, n, valid_routes,
                      N:gl.constexpr, BN:gl.constexpr, CN:gl.constexpr,
                      PARTITION:gl.constexpr, TOP_K:gl.constexpr, CK_SORTING:gl.constexpr,
                      USE_C64:gl.constexpr):
    """Scale one [M32,N128] register panel and scatter directly to routed C.

        acc -> ((acc * SA) * SB) * weight -> pack adjacent 16-bit values -> store
        P0 already decoded CK routes and mapped invalid experts to route=-1.
        Route/expert masks suppress padding. No shared output buffer is used."""
    linear:gl.constexpr = gl.to_linear_layout(acc.type.layout,(32,CN))
    packing:gl.constexpr = _output_pack_layout(linear)
    acc = gl.convert_layout(acc,linear)
    sa = gl.convert_layout(sa,gl.SliceLayout(1,linear))
    weight = gl.convert_layout(weight,gl.SliceLayout(1,linear))
    sb = gl.convert_layout(sb,gl.SliceLayout(0,linear))
    value = ((acc * sa[:,None]) * sb[None,:]) * weight[:,None]
    packed = gl.convert_layout(value,packing).to(C.dtype.element_ty)
    out_route = gl.convert_layout(route,gl.SliceLayout(1,packing))
    out_expert = gl.convert_layout(expert,gl.SliceLayout(1,packing))
    rn = gl.arange(0,CN,gl.SliceLayout(0,packing))
    valid = (out_route >= 0) & (out_route < valid_routes) & (out_expert >= 0)
    if USE_C64:
        offsets = (out_route.to(gl.int64)[:, None] * N + n.to(gl.int64) * BN
                   + PARTITION * CN + rn[None, :].to(gl.int64))
        gl.store(C + offsets, packed, valid[:, None])
    else:
        offsets = out_route.to(gl.uint32)[:, None] * N + n * BN + PARTITION * CN + rn[None, :]
        buffer_store(packed, C, offsets, valid[:, None])


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
              USE_C64: gl.constexpr, num_tokens_post_padded_ptr, INSN_WARMUP: gl.constexpr = False):
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
    first_n = 0
    if INSN_WARMUP:
        _warm_mmac_instructions(PARTITION, 1)
        # ### ASM Note: An opaque zero prevents loop peeling/unrolling from cloning named ASM labels.
        first_n = gl.inline_asm_elementwise(
            "s_mov_b32 $0, 0", constraints="=s", args=[], dtype=gl.int32, is_pure=False, pack=1)
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
        for n in range(first_n, N_TILES):
            hcu.abarrier_try_wait(7 if PARTITION == 1 else 5, b_epoch & 1)
            current_sb = mb.index((b_epoch&1)*2+PARTITION).load(gl.SliceLayout(0, result))
            b = bm.index(0).load(hcu.MmacInterleave2Layout(gl.DotOperandLayout(1, mma, 8)))
            if n == 0:
                hcu.abarrier_try_wait(12 if (a_epoch & 1) == 0 else 0, (a_epoch >> 1) & 1)
                sa = ma.index((a_epoch & 1) * 4).load(gl.SliceLayout(1, result))
                weight = ma.index((a_epoch & 1) * 4 + 1).load(gl.SliceLayout(1, result))
                route = ma.index((a_epoch & 1) * 4 + 2).load(gl.SliceLayout(1, result)).to(gl.int32, bitcast=True)
            hcu.wait_lgkmcnt(0)
            if n == 0:
                hcu.abarrier_arrive(3 + (a_epoch & 1))
            hcu.abarrier_arrive(8 if PARTITION == 1 else 6)
            acc = gl.full((BM, CN), 0, ACC, result)
            hcu.sched_barrier()
            if INSN_WARMUP:
                acc = _mmac(a, b, acc, PARTITION, 0, INSN_WARMUP)
            else:
                acc = gl.amd.cdna4.mfma(a, b, acc)
            hcu.sched_barrier()
            if acc.dtype == gl.int32:
                acc = acc.to(gl.float32)
            # acc[64,128] + four row vectors -> two M32 panels,
            # scaled and scattered independently without an LDS mailbox.
            x0,x1 = gl.split(gl.permute(gl.reshape(acc,(2,32,CN)),(1,2,0)))
            sa0,sa1 = gl.split(gl.permute(gl.reshape(sa,(2,32)),(1,0)))
            w0,w1 = gl.split(gl.permute(gl.reshape(weight,(2,32)),(1,0)))
            r0,r1 = gl.split(gl.permute(gl.reshape(route,(2,32)),(1,0)))
            e0,e1 = gl.split(gl.permute(gl.reshape(expert,(2,32)),(1,0)))
            _scale_store_rows(x0,sa0,current_sb,w0,r0,e0,C,n,valid_routes,N,BN,CN,PARTITION,TOP_K,CK_SORTING,USE_C64)
            hcu.sched_barrier()
            _scale_store_rows(x1,sa1,current_sb,w1,r1,e1,C,n,valid_routes,N,BN,CN,PARTITION,TOP_K,CK_SORTING,USE_C64)
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
    PARTITION_REGS: gl.constexpr = None, INSN_WARMUP: gl.constexpr = False,
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
when routed-weight multiplication is enabled. A/B/C wide addressing is constexpr.
"""
    gl.static_assert(BLOCK_SIZE_M == 64 and BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 256)
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
            GROUP_SIZE_M, CK_SORTING, 0, N_TILES, USE_ADDR_OFFSET_INT64_C, num_tokens_post_padded_ptr, INSN_WARMUP)),
        (_consumer, (
            C, SA, SB, Weights, Sorted, Experts, am, b1, co1, ma, mb, pid, TOTAL_TILES,
            num_sorted_rows, num_valid_routes, N, K, top_k, BM, BN, BK, STAGES,
            GROUP_SIZE_M, CK_SORTING, 1, N_TILES, USE_ADDR_OFFSET_INT64_C, num_tokens_post_padded_ptr, INSN_WARMUP)),
    ], [4, 4, 4, 4], regs)
    for barrier in gl.static_range(16):
        hcu.abarrier_invalidate(barrier)




def is_persistent_channel_gemm2_supported(a, b, a_scale, b_scale, *, arch, cu_count,
                                        top_k, config, c_sorted=False, a_sorted=True,
                                        ck_sorting=True, output_dtype=None,
                                        scale_recipe="fp8_channel", has_bias=False,
                                        mul_routed_weight=True):
    """CPU metadata query for the BM64/BN256/K256 pipeline; otherwise use Triton.

    Only BM is part of the sorting contract. This candidate selects its own
    BN256/BK256, launch options and row-persistent traversal. The caller checks
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
    return config.get("BLOCK_SIZE_M") == 64 and config.get("GROUP_SIZE_M", 1) == 1
