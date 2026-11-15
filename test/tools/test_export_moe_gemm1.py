# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU goldens for standalone GEMM1 exports; GPU execution uses AIDSL file_runner."""

import pytest
import torch

from boltops.tools.export_moe_gemm1 import prepare_case, tensor_file


@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_routed_golden_and_raw_storage(tmp_path, input_dtype, output_dtype):
    m, n, k, experts, topk = 7, 128, 512, 3, 2
    tensors, golden = prepare_case(
        m, n, k, experts, topk, input_dtype, output_dtype, 3197
    )
    ids = tensors["Sorted"].to(torch.int64)
    tokens, routes = ids & 0x00FFFFFF, (ids >> 24) & 255
    owners = tensors["Experts"].repeat_interleave(128)
    valid = (tokens < m) & (owners >= 0)
    # Every original token/route appears once across expert blocks.
    assert sorted((tokens[valid] * topk + routes[valid]).tolist()) == list(
        range(m * topk)
    )
    assert tensors["WorkCount"].item() == ids.numel()
    assert torch.all(golden[~valid] == 0)
    assert torch.all(owners[-128:] == -1)
    assert tokens[-128] == 0  # Invalid expert zeroing is independent of token padding.
    # Independent per-row K reduction, separate from export's dense GEMM/gather.
    for row in torch.nonzero(valid).flatten().tolist():
        token, expert = int(tokens[row]), int(owners[row])
        product = tensors["A"][token].float() * tensors["B"][expert].float()
        expected = (
            product.sum(-1) * tensors["SA"][token, 0] * tensors["SB"][expert]
        ).to(output_dtype)
        torch.testing.assert_close(golden[row], expected, rtol=0, atol=0)
    blob = tensor_file(tmp_path, "golden.bin", golden)
    raw = torch.frombuffer(
        bytearray((tmp_path / blob["file"]).read_bytes()), dtype=output_dtype
    )
    torch.testing.assert_close(raw.view_as(golden), golden, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        tensor_file(tmp_path, "golden.bin", golden)
    assert (tmp_path / "golden.bin").stat().st_size == blob["bytes"]


@pytest.mark.parametrize("change", [dict(k=640), dict(topk=4), dict(m=0)])
def test_invalid_export_shapes(change):
    options = dict(
        m=7,
        n=128,
        k=512,
        experts=3,
        topk=2,
        input_dtype=torch.int8,
        output_dtype=torch.bfloat16,
        seed=3197,
    )
    options.update(change)
    with pytest.raises(ValueError):
        prepare_case(**options)


def test_frozen_case_detects_changed_routing(tmp_path):
    import json
    from boltops.tools.export_moe_gemm1 import load_frozen_case

    ids = torch.tensor([0, 1 << 24, 7], dtype=torch.int32)
    desc = dict(
        role="input",
        logical_dtype="torch.int32",
        logical_shape=[3],
        **tensor_file(tmp_path, "Sorted.bin", ids),
    )
    (tmp_path / "launch.json").write_text(
        json.dumps(
            dict(operator="moe_gemm1", schema_version=4, buffers=dict(Sorted=desc))
        )
    )
    _, tensors = load_frozen_case(tmp_path)
    torch.testing.assert_close(tensors["Sorted"], ids, rtol=0, atol=0)
    data = bytearray((tmp_path / "Sorted.bin").read_bytes())
    data[0] ^= 1
    (tmp_path / "Sorted.bin").write_bytes(data)
    with pytest.raises(ValueError, match="Frozen input/golden changed"):
        load_frozen_case(tmp_path)


def test_export_preserves_existing_case(tmp_path):
    from boltops.tools.export_moe_gemm1 import export_case

    (tmp_path / "Sorted.bin").write_bytes(b"fixed-routing")
    with pytest.raises(FileExistsError, match="Frozen case already exists"):
        export_case(
            tmp_path,
            m=7,
            n=128,
            k=512,
            experts=3,
            topk=2,
            bn=128,
            grid=1,
            input_dtype=torch.int8,
            output_dtype=torch.bfloat16,
            seed=3197,
            readobj="unused",
        )
    assert (tmp_path / "Sorted.bin").read_bytes() == b"fixed-routing"
