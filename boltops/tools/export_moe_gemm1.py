# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Export standalone MoE GEMM1 cases for AIDSL's schema-4 C++ file runner.

Build side needs Torch/Triton; the resulting directory needs only file_runner
and HIP on gfx946. Inputs and full-output goldens are generated on the CPU.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import torch


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def tensor_file(directory, name, tensor):
    """Write raw little-endian storage, preserving FP8 and BF16 bit patterns."""
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    with (directory / name).open("xb") as stream:
        stream.write(data)
    return dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def prepare_case(m, n, k, experts, topk, input_dtype, output_dtype, seed):
    """Generate CK-packed routing and an independent dense CPU reference.

    Each token chooses distinct experts. Expert blocks pad to128 rows; an
    additional invalid-expert block tests zero output. Golden uses decoded
    quantized operands, FP32 accumulation and the requested output cast.
    """
    if min(m, n, k, experts, topk) <= 0 or not topk <= min(experts, 256):
        raise ValueError("Require positive dimensions and topk<=min(experts,256)")
    if k < 512 or k % 256:
        raise ValueError("GEMM1 requires K>=512 divisible by256")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(-8, 9, (m, k), generator=rng).float()
    b = torch.randint(-8, 9, (experts, n, k), generator=rng).float()
    if input_dtype != torch.int8:
        a, b = a / 8, b / 8
    a, b = a.to(input_dtype), b.to(input_dtype)
    sa = torch.randint(1, 5, (m, 1), generator=rng).float() / 8
    sb = torch.randint(1, 5, (experts, n), generator=rng).float() / 8
    assignments = torch.stack(
        [torch.randperm(experts, generator=rng)[:topk] for _ in range(m)]
    )
    routes, expert_blocks, flat_routes = [], [], []
    for expert in range(experts):
        flat = torch.nonzero(assignments.flatten() == expert).flatten()
        if not flat.numel():
            continue
        padded = ((flat.numel() + 127) // 128) * 128
        packed = ((flat // topk) | ((flat % topk) << 24)).to(torch.int32)
        routes.append(
            torch.cat(
                (packed, torch.full((padded - flat.numel(),), m, dtype=torch.int32))
            )
        )
        flat_routes.append(torch.cat((flat, torch.full((padded - flat.numel(),), -1))))
        expert_blocks.extend([expert] * (padded // 128))
    routes.append(torch.full((128,), m, dtype=torch.int32))
    routes[-1][0] = 0  # A valid token under an invalid expert must also produce zero.
    flat_routes.append(torch.full((128,), -1))
    expert_blocks.append(-1)
    sorted_ids, flat_ids = torch.cat(routes), torch.cat(flat_routes)
    expert_ids = torch.tensor(expert_blocks, dtype=torch.int32)
    golden = torch.zeros((sorted_ids.numel(), n), dtype=output_dtype)
    for expert in range(experts):
        # Compute dense MxN first, then gather routed rows independently of GPU tiles.
        dense = ((a.float() @ b[expert].float().T) * sa * sb[expert]).to(output_dtype)
        for block, owner in enumerate(expert_blocks):
            if owner != expert:
                continue
            row = block * 128
            ids = flat_ids[row : row + 128]
            valid = ids >= 0
            golden[row : row + 128][valid] = dense[ids[valid] // topk]
    tensors = dict(
        A=a,
        B=b,
        SA=sa,
        SB=sb,
        Sorted=sorted_ids,
        Experts=expert_ids,
        WorkCount=torch.tensor([sorted_ids.numel()], dtype=torch.int32),
    )
    return tensors, golden


def load_frozen_case(directory):
    """Load AIDSL inputs without RNG, rejecting any changed raw-input/golden bytes."""
    launch = json.loads((directory / "launch.json").read_text())
    if launch.get("operator") != "moe_gemm1" or launch.get("schema_version") != 4:
        raise ValueError("Expected a schema-4 MoE GEMM1 case")
    tensors = {}
    for name, description in launch["buffers"].items():
        output = description["role"] == "output"
        blob = description["golden"] if output else description
        data = (directory / blob["file"]).read_bytes()
        if (
            len(data) != blob["bytes"]
            or hashlib.sha256(data).hexdigest() != blob["sha256"]
        ):
            raise ValueError(f"Frozen input/golden changed: {directory / blob['file']}")
        dtype_name = (
            blob["storage_dtype"]
            if output
            else description["logical_dtype"].split(".")[-1]
        )
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_name)
        if dtype is None:
            dtype = getattr(torch, dtype_name)
        tensors[name] = (
            torch.frombuffer(bytearray(data), dtype=torch.uint8)
            .view(dtype)
            .reshape(description["logical_shape"])
        )
    return launch, tensors


def export_case(
    directory,
    *,
    m,
    n,
    k,
    experts,
    topk,
    bn,
    grid,
    input_dtype,
    output_dtype,
    seed,
    readobj,
    group_size_m=1,
):
    """Compile the library kernel's default path and bind its actual code-object ABI."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Frozen case already exists: {directory}; use a new output directory"
        )
    import yaml
    from boltops.fused_moe.gluon import (
        gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 as module,
    )

    directory.mkdir(parents=True, exist_ok=True)
    tensors, golden = prepare_case(
        m, n, k, experts, topk, input_dtype, output_dtype, seed
    )
    total_tiles = tensors["Experts"].numel() * ((n + bn - 1) // bn)
    if grid <= 0 or group_size_m <= 0:
        raise ValueError("Grid and GROUP_SIZE_M must be positive")
    kernel = module.gemm_top_w8a8_channel_wasp_persistent
    output = torch.empty_like(golden)
    # warmup() compiles with mock pointers; no GPU input generation or compute launch.
    values = dict(
        zip(
            kernel.arg_names[:17],
            (
                tensors["A"],
                tensors["B"],
                tensors["SA"],
                tensors["SB"],
                output,
                tensors["Sorted"],
                tensors["Experts"],
                m,
                experts,
                n,
                k,
                topk,
                n,
                1,
                0,
                1,
                0,
            ),
        )
    )
    opts = dict(
        module.DEFAULT_LAUNCH_OPTIONS,
        TOTAL_TILES=total_tiles,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=128,
        USE_ADDR_OFFSET_INT64_A=tensors["A"].numel() >= 2**32 - 1,
        USE_ADDR_OFFSET_INT64_B=tensors["B"].numel() >= 2**32 - 1,
        USE_ADDR_OFFSET_INT64_C=output.numel() * output.element_size() >= 2**32 - 1,
        GROUP_SIZE_M=group_size_m,
        num_tokens_post_padded_ptr=tensors["WorkCount"],
    )
    compiled = kernel.warmup(*values.values(), grid=(grid,), **opts)
    asm = compiled.asm["amdgcn"]
    if ".Lmoe_iwarm_c" in asm:
        raise RuntimeError("Export must use the default disabled instruction warmup")
    for suffix, data in compiled.asm.items():
        if isinstance(data, (str, bytes)):
            path = directory / ("kernel." + suffix)
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
    notes = subprocess.check_output(
        [readobj, "--notes", str(directory / "kernel.hsaco")], text=True
    )
    (directory / "kernel.metadata.txt").write_text(notes)
    metadata = yaml.safe_load(
        notes.split("AMDGPU Metadata: ", 1)[1].split("\n...", 1)[0]
    )
    (code,) = metadata["amdhsa.kernels"]
    for field in (
        ".sgpr_spill_count",
        ".vgpr_spill_count",
        ".private_segment_fixed_size",
    ):
        if code.get(field, 0):
            raise RuntimeError(f"Unexpected spill/scratch: {field}={code[field]}")
    if code[".wavefront_size"] != 64 or compiled.metadata.shared != 131072:
        raise RuntimeError("Unexpected GEMM1 wave size or LDS allocation")
    constants = {kernel.arg_names[index[0]] for index in compiled.src.constants}
    runtime = [name for name in kernel.arg_names if name not in constants]
    abi = [arg for arg in code[".args"] if not arg[".value_kind"].startswith("hidden_")]
    # Triton appends global/profile scratch pointers; neither is used by this kernel.
    names = runtime + ["global_scratch", "profile_scratch"]
    if len(names) != len(abi):
        raise RuntimeError(f"ABI differs from JIT signature: {len(abi)} vs {names}")
    pointer_names = dict(
        zip(kernel.arg_names[:7], ("A", "B", "SA", "SB", "C", "Sorted", "Experts"))
    )
    pointer_names["num_tokens_post_padded_ptr"] = "WorkCount"
    values.update(opts)
    args, bindings, scalars = [], [], {}
    for name, arg in zip(names, abi):
        kind, size = arg[".value_kind"], arg[".size"]
        if name in pointer_names or name.endswith("_scratch"):
            if kind != "global_buffer" or size != 8:
                raise RuntimeError(f"Invalid pointer ABI for {name}")
            binding = (
                {"buffer": pointer_names[name]}
                if name in pointer_names
                else {"null": True}
            )
        else:
            if kind != "by_value" or size not in (4, 8):
                raise RuntimeError(f"Invalid scalar ABI for {name}")
            scalars[name] = dict(
                dtype="int32" if size == 4 else "int64", value=values[name]
            )
            binding = {"scalar": name}
        args.append(dict(name=name, kind=kind, size=size, offset=arg[".offset"]))
        bindings.append(binding)
    buffers = {}
    for name, tensor in tensors.items():
        buffers[name] = dict(
            role="input",
            **tensor_file(directory, name + ".bin", tensor),
            logical_shape=list(tensor.shape),
            logical_dtype=str(tensor.dtype),
        )
    dtype_name = "bf16" if output_dtype == torch.bfloat16 else "fp16"
    buffers["C"] = dict(
        role="output",
        bytes=golden.numel() * golden.element_size(),
        storage_dtype=dtype_name,
        logical_shape=list(golden.shape),
        golden=dict(
            **tensor_file(directory, "golden.bin", golden),
            storage_dtype=dtype_name,
            comparison=dict(type="allclose", rtol=0.02, atol=0.002),
        ),
    )
    variant = dict(
        schema_version=1,
        operator="moe_gemm1",
        arch="gfx946",
        variant=directory.name,
        kernels=[
            dict(
                id="gemm1",
                name=code[".name"],
                file="kernel.hsaco",
                sha256=hashlib.sha256(compiled.asm["hsaco"]).hexdigest(),
                block=[1024, 1, 1],
                shared_bytes=compiled.metadata.shared,
                args=args,
            )
        ],
    )
    write_json(directory / "variant.json", variant)
    source = Path(inspect.getfile(kernel.fn))
    write_json(
        directory / "compile.json",
        dict(
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            triton=__import__("triton").__version__,
            options={
                k: v for k, v in opts.items() if k != "num_tokens_post_padded_ptr"
            },
            instruction_warmup_default=inspect.signature(kernel.fn)
            .parameters["INSN_WARMUP"]
            .default,
            kernel_metadata=compiled.metadata._asdict(),
        ),
    )
    launch = dict(
        schema_version=4,
        data_mode="files",
        operator="moe_gemm1",
        arch="gfx946",
        variant=directory.name,
        variant_manifest="variant.json",
        seed=seed,
        parameters=dict(
            m=m, n=n, k=k, experts=experts, topk=topk, bn=bn, group_size_m=group_size_m
        ),
        buffers=buffers,
        scalars=scalars,
        launches=[dict(kernel="gemm1", grid=[grid, 1, 1], arguments=bindings)],
    )
    write_json(directory / "launch.json", launch)
    return dict(
        case=directory.name,
        grid=grid,
        tiles=total_tiles,
        checked_elements=golden.numel(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m", type=int, default=97)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--group-size-m", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3197)
    parser.add_argument("--bn", choices=("128", "256", "all"), default="all")
    parser.add_argument("--input-dtype", choices=("fp8", "int8", "all"), default="all")
    parser.add_argument("--output-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--readobj",
        default=shutil.which("llvm-readobj") or "/opt/rocm/llvm/bin/llvm-readobj",
    )
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.output_dtype == "bf16" else torch.float16
    for kind in ("fp8", "int8") if args.input_dtype == "all" else (args.input_dtype,):
        for bn in (128, 256) if args.bn == "all" else (int(args.bn),):
            name = f"{kind}_bn{bn}_{args.output_dtype}_m{args.m}_n{args.n}_k{args.k}_grid{args.grid}"
            result = export_case(
                args.out / name,
                m=args.m,
                n=args.n,
                k=args.k,
                experts=args.experts,
                topk=args.top_k,
                bn=bn,
                grid=args.grid,
                input_dtype=torch.int8 if kind == "int8" else torch.float8_e4m3fn,
                output_dtype=dtype,
                seed=args.seed,
                readobj=args.readobj,
                group_size_m=args.group_size_m,
            )
            print(json.dumps(result), flush=True)
    # Preserve the index when another geometry/output dtype is added later.
    cases = []
    for path in sorted(args.out.glob("*/launch.json")):
        launch = json.loads(path.read_text())
        if launch.get("operator") != "moe_gemm1":
            continue
        params = launch["parameters"]
        cases.append(
            dict(
                case=path.parent.name,
                grid=launch["launches"][0]["grid"][0],
                tiles=launch["buffers"]["Experts"]["bytes"]
                // 4
                * ((params["n"] + params["bn"] - 1) // params["bn"]),
                checked_elements=launch["buffers"]["C"]["bytes"] // 2,
            )
        )
    write_json(args.out / "cases.json", cases)


if __name__ == "__main__":
    main()
