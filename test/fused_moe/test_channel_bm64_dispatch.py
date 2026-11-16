# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Check BM64 routing ownership and the ordinary Triton fallback on CPU metadata."""
import importlib
import inspect

import pytest
import torch


def tensor(shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device="meta")


class Capture:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append((grid, args, kwargs))
        return run


@pytest.mark.parametrize("bm,bn,bk", [(64, 256, 256), (64, 512, 64),
                                       (128, 256, 128), (128, 512, 64)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int8])
def test_gemm2_selects_sorting_block_and_launch_geometry(monkeypatch, bm, bn, bk, dtype):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info",
                        lambda *_: DeviceInfo(0, "gfx946", 48))
    mod = moe_op._gluon_channel_gemm2_module(bn, bm)
    assert mod.__name__.endswith(f"bm{bm}xbn{bn}xbk{bk}")
    defaults = inspect.signature(mod.gemm_bottom_w8a8_channel_wasp_persistent.fn).parameters
    capture = Capture()
    monkeypatch.setattr(mod, "gemm_bottom_w8a8_channel_wasp_persistent", capture)
    a, b = tensor((3 * bm, 256), dtype), tensor((2, bn * 3, 256), dtype)
    sa, sb = tensor((3 * bm, 1)), tensor((2, bn * 3, 1))
    c = tensor((194, bn * 3), torch.bfloat16)
    ids, owners, count = tensor((3 * bm,), torch.int32), tensor((3,), torch.int32), tensor((1,), torch.int32)
    config = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)
    fp8 = dtype != torch.int8
    assert moe_op._gluon_gemm2_prefers_sorted_layout(
        b, sb, top_k=2, config=config, output_dtype=c.dtype,
        use_fp8_w8a8=fp8, use_int8_w8a8=not fp8, per_channel_quant=True,
        block_shape=None, ck_sorting=True, has_bias=False)
    assert moe_op._try_gluon_channel_gemm2(
        a, b, c, sa, sb, None, ids, owners, count, 1, config,
        use_fp8_w8a8=fp8, use_int8_w8a8=not fp8, per_channel_quant=True,
        block_shape=None, c_sorted=False, a_sorted=True, ck_sorting=True,
        ck_topk=2, naive_block_assignment=False, mul_routed_weight=False,
        B_bias=None, B_zp=None)
    grid, args, opts = capture.launches[0]
    assert grid == (48,) and args[-1] == 9
    assert opts.get("BLOCK_SIZE_K", defaults["BLOCK_SIZE_K"].default) == bk
    assert opts["top_k"] == 2
    assert opts["num_tokens_post_padded_ptr"] is count
    assert config == dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int8])
def test_gemm1_bm64_launch_counts_sorted_blocks(monkeypatch, dtype):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info",
                        lambda *_: DeviceInfo(0, "gfx946", 96))
    mod = importlib.import_module(
        "boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm64xbn512xbk256")
    capture = Capture()
    monkeypatch.setattr(mod, "gemm_top_w8a8_channel_wasp_persistent", capture)
    a, b = tensor((1024, 2048), dtype), tensor((2, 1024, 2048), dtype)
    c, sa, sb = tensor((192, 1024), torch.bfloat16), tensor((1024, 1)), tensor((2, 1024, 1))
    ids, owners, count = tensor((192,), torch.int32), tensor((3,), torch.int32), tensor((1,), torch.int32)
    assert moe_op._try_gluon_channel_gemm1(
        a, b, c, sa, sb, ids, owners, count, 2, dict(BLOCK_SIZE_M=64),
        use_fp8_w8a8=dtype != torch.int8, use_int8_w8a8=dtype == torch.int8,
        per_channel_quant=True, block_shape=None, c_sorted=True, a_sorted=False,
        ck_sorting=True, ck_topk=2, naive_block_assignment=False,
        mul_routed_weight=False, B_bias=None, B_zp=None)
    grid, args, opts = capture.launches[0]
    assert grid == (96,) and args[5] is ids and args[6] is owners
    assert opts["PERSISTENT_GRID"] == 96
    assert opts["TOTAL_TILES"] == 6
    assert (opts["BLOCK_SIZE_M"], opts["BLOCK_SIZE_N"], opts["BLOCK_SIZE_K"]) == (64, 512, 256)
    assert opts["num_tokens_post_padded_ptr"] is count


@pytest.mark.parametrize("bn", [256, 512])
@pytest.mark.parametrize("wide", ["A", "B", "C"])
def test_bm64_gemm2_independent_wide_addresses(monkeypatch, bn, wide):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info",
                        lambda *_: DeviceInfo(0, "gfx946", 48))
    mod = moe_op._gluon_channel_gemm2_module(bn, 64)
    capture = Capture()
    monkeypatch.setattr(mod, "gemm_bottom_w8a8_channel_wasp_persistent", capture)
    rows = 32 * 1024**2 if wide == "A" else 192
    experts = 65536 if wide == "B" else 2
    output_rows = 16 * 1024**2 if wide == "C" else 194
    assert moe_op._try_gluon_channel_gemm2(
        tensor((rows, 256), torch.int8), tensor((experts, bn, 256), torch.int8),
        tensor((output_rows, bn), torch.bfloat16), tensor((rows, 1)), tensor((experts, bn, 1)),
        None, tensor((192,), torch.int32), tensor((3,), torch.int32), tensor((1,), torch.int32),
        1, dict(BLOCK_SIZE_M=64), use_fp8_w8a8=False, use_int8_w8a8=True,
        per_channel_quant=True, block_shape=None, c_sorted=False, a_sorted=True,
        ck_sorting=True, ck_topk=2, naive_block_assignment=False,
        mul_routed_weight=False, B_bias=None, B_zp=None)
    _, _, opts = capture.launches[0]
    assert {name: opts["USE_ADDR_OFFSET_INT64_" + name] for name in "ABC"} == {
        name: name == wide for name in "ABC"}


@pytest.mark.parametrize("bn", [256, 512])
def test_bm64_gemm2_rejects_unsupported_group_before_launch(bn):
    from boltops.fused_moe.triton import moe_op
    mod = moe_op._gluon_channel_gemm2_module(bn, 64)
    assert not mod.is_persistent_channel_gemm2_supported(
        tensor((192, 256), torch.int8), tensor((2, bn, 256), torch.int8),
        tensor((192, 1)), tensor((2, bn, 1)), arch="gfx946", cu_count=48, top_k=2,
        config=dict(BLOCK_SIZE_M=64, GROUP_SIZE_M=2), scale_recipe="int8_channel")


@pytest.mark.parametrize("bn", [256, 512])
def test_bm64_gemm2_scale_stride_selects_wide_b(monkeypatch, bn):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info",
                        lambda *_: DeviceInfo(0, "gfx946", 48))
    mod = moe_op._gluon_channel_gemm2_module(bn, 64)
    capture = Capture()
    monkeypatch.setattr(mod, "gemm_bottom_w8a8_channel_wasp_persistent", capture)
    # Even one adjacent channel is 4 GiB away: tensor numel alone misses this.
    stride = 1024**3
    sb = torch.empty_strided((2, bn), (bn * stride, stride), device="meta")
    assert moe_op._try_gluon_channel_gemm2(
        tensor((192, 256), torch.int8), tensor((2, bn, 256), torch.int8),
        tensor((194, bn), torch.bfloat16), tensor((192, 1)), sb,
        None, tensor((192,), torch.int32), tensor((3,), torch.int32), tensor((1,), torch.int32),
        1, dict(BLOCK_SIZE_M=64), use_fp8_w8a8=False, use_int8_w8a8=True,
        per_channel_quant=True, block_shape=None, c_sorted=False, a_sorted=True,
        ck_sorting=True, ck_topk=2, naive_block_assignment=False,
        mul_routed_weight=False, B_bias=None, B_zp=None)
    _, _, opts = capture.launches[0]
    assert opts["USE_ADDR_OFFSET_INT64_B"] and opts["stride_bsn"] == stride
