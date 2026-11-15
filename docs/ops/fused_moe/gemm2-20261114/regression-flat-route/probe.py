# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""GEMM2 PMD probe. CPU inputs/golden; one kernel dispatch; CU0 trace.

Run inside zx-perf-node60, with GPU_CHIP=sb and the paired Triton/LLVM.
--tiles=0 runs all real padded output tiles. The default 384-tile prefix gives
48 active row-persistent CTAs at BN256, eight N outputs per CTA.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from boltops.tools.gluon_moe_gemm1 import prepare_cpu, save_kernel
from boltops.fused_moe.gluon.gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128_256 import (
    gemm_bottom_w8a8_channel_wasp_persistent, DEFAULT_LAUNCH_OPTIONS,
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=('gluon', 'triton'), default='gluon')
    p.add_argument('--bn', type=int, choices=(128, 256), default=256)
    p.add_argument('--bk', type=int, choices=(128, 256), default=256)
    p.add_argument('--a-buffer', action='store_true')
    p.add_argument('--tiles', type=int, default=384)
    p.add_argument('--grid', type=int, default=48)
    p.add_argument('--compile-only', action='store_true')
    args = p.parse_args()
    root = Path.cwd()
    case = dict(M=4096, E=256, N=256, K=2048, topk=8)
    M, E, N, K, TOPK, BM = 4096, 256, 2048, 256, 8, 128
    t, _, _, _, routing = prepare_cpu(case, 'fp8_channel', BM, args.bn, 0, 3101, gemm='gemm2')
    encoded = t['Sorted'].long()
    routes = (encoded & 0xffffff) * TOPK + ((encoded >> 24) & 255)
    valid = routes < M * TOPK
    source = routes.clamp_max(M * TOPK - 1)
    # FP8 transport is a byte gather, preserving E4M3FN values exactly.
    sorted_a = t['A'].view(torch.uint8)[source].clone()
    sorted_a[~valid] = 0
    t['A'] = sorted_a.view(t['A'].dtype)
    t['SA'] = t['SA'][source].clone()
    t['SA'][~valid] = 0
    rows = routes.numel()
    full_tiles = rows // BM * (N // args.bn)
    total = full_tiles if args.tiles == 0 else min(args.tiles, full_tiles)
    assert total % (N // args.bn) == 0, 'diagnostic tiles must cover whole row blocks'
    assert total >= args.grid > 0
    t["Sorted"] = routes.to(torch.int32)
    routing["sorting_format"] = "flat route index; padding=M*topk"
    inputs_sha = {n: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                  for n, v in t.items()}
    gpu = {n: v.cuda() for n, v in t.items()}
    c = torch.full((M * TOPK, N), float('nan'), dtype=torch.bfloat16).cuda()
    if args.mode == 'gluon':
        kernel = gemm_bottom_w8a8_channel_wasp_persistent
        options = dict(DEFAULT_LAUNCH_OPTIONS, N=N, K=K, top_k=TOPK,
                       BLOCK_SIZE_M=BM, BLOCK_SIZE_N=args.bn, BLOCK_SIZE_K=args.bk,
                       USE_MLS_LOAD_A=not args.a_buffer, CK_SORTING=False)
        arguments = (gpu['A'], gpu['B'], gpu['SA'], gpu['SB'], c,
                     gpu['RouteWeights'], gpu['Sorted'], gpu['Experts'], rows, M * TOPK, total)
    else:
        from boltops.fused_moe.triton.moe_op import fused_moe_persistent_kernel
        kernel = fused_moe_persistent_kernel
        # Exact M4096 entry in the user-selected gfx938 bottom JSON.
        config_path = Path(__file__).resolve().parents[1] / 'fused_moe/triton/configs/triton_3_6' / (
            'E=256,N=256,K=2048,topk=8,arch=gfx938,dtype=fp8_w8a8,is_bottom=True.json')
        config = json.loads(config_path.read_text())['4096']
        assert args.bn == config['BLOCK_SIZE_N'] and args.bk == config['BLOCK_SIZE_K']
        padded_rows = total // (N // args.bn) * BM
        padded = torch.tensor([padded_rows], dtype=torch.int32).cuda()
        a, b, sa, sb = gpu['A'], gpu['B'], gpu['SA'], gpu['SB']
        arguments = (a, b, c, gpu['RouteWeights'], padded, gpu['Experts'], gpu['Sorted'],
                     gpu['SortedWeights'], sa, sb, N, K, rows, M * TOPK,
                     *a.stride(), b.stride(0), b.stride(2), b.stride(1), N, 1,
                     *sa.stride(), sb.stride(0), sb.stride(2), sb.stride(1), rows)
        options = dict(BLOCK_SIZE_M=BM, BLOCK_SIZE_N=args.bn, BLOCK_SIZE_K=args.bk,
                       GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False,
                       MUL_ROUTED_WEIGHT=True, USE_ADDR_OFFSET_INT64_A=False,
                       USE_ADDR_OFFSET_INT64_B=False, USE_ADDR_OFFSET_INT64_C=False,
                       top_k=1, compute_type=__import__('triton').language.bfloat16,
                       use_fp8_w8a8=True, use_int8_w8a8=False, use_int8_w8a16=False,
                       per_channel_quant=True, c_sorted=False, a_sorted=True,
                       bottom_a_use_mls_load=False, ck_sorting=True, ck_topk=TOPK,
                       NUM_SMS=args.grid, NUM_XCDS=1, group_n=0, group_k=0,
                       num_warps=4, num_stages=1, kpack=2, use_block_pingpong=False,
                       buffer_cache_swizzle=True)
    a_reuse = args.mode == 'gluon' and getattr(sys.modules[kernel.fn.__module__], 'ROW_PERSISTENT_A', False)
    record = dict(case=case, actual_N=N, actual_K=K, grid=args.grid,
                  a_reuse_across_n=a_reuse,
                  active_ctas=min(args.grid, total // (N // args.bn) if a_reuse else total),
                  full_logical_tiles=full_tiles, measured_logical_tiles=total,
                  diagnostic_only=total != full_tiles, sorted_rows=rows,
                  input_sha256=inputs_sha, routing=routing, mode=args.mode,
                  options={k: str(v) if k == 'compute_type' else v for k, v in options.items()})
    (root / 'launch.json').write_text(json.dumps(record, indent=2))
    (root / 'gemm').mkdir(exist_ok=True)
    os.chdir(root / 'gemm')
    compiled = kernel.warmup(*arguments, grid=(args.grid,), **options)
    save_kernel(compiled)
    os.chdir(root)
    if args.compile_only:
        print('COMPILE ONLY', flush=True)
        return
    print(f'LAUNCH {args.mode} {total}/{full_tiles} output tiles, grid={args.grid}', flush=True)
    kernel[(args.grid,)](*arguments, **options)
    torch.cuda.synchronize()
    actual = c.cpu()
    errors = checked = 0
    for m in range(total // (N // args.bn)):
        route = routes[m * BM:(m + 1) * BM]
        mask = route < M * TOPK
        route = route[mask]
        expert = int(t['Experts'][m])
        a = t['A'][m * BM:(m + 1) * BM].float()[mask]
        expected = ((a @ t['B'][expert].float().T)
                    * t['SA'][m * BM:(m + 1) * BM][mask]
                    * t['SB'][expert, :, 0][None, :] * t['RouteWeights'][route, None]).bfloat16()
        got = actual[route]
        errors += int((~torch.isclose(got.float(), expected.float(), rtol=.02, atol=.002)).sum())
        checked += expected.numel()
    result = dict(mismatches=errors, checked_elements=checked, shared=compiled.metadata.shared,
                  scope='full' if total == full_tiles else 'diagnostic_prefix')
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
    assert errors == 0


if __name__ == '__main__':
    main()
